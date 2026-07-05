#!/usr/bin/env python
"""phosbench part 2 (STRETCH) - batched CUDA graph: N small cells in ONE graph.

The per-step graph floor at 140 atoms is ~1.9 ms, almost all of it the single
host->device replay launch.  If a workload has many small independent cells (an
ensemble, replica exchange, a batch of candidate structures), we can put N=8
replicas of the 140-atom cell into ONE torch_geometric Batch and evaluate them in
a single graph replay - amortising that one launch across 8 systems.  This probes
the below-break-even regime the main sweep flagged (host-bound small cells).

Arms compared, all cuEq/fp32, on the 140-atom cell replicated 8x:
  eager-batched-synced : eager forward of the 8-graph batch, per-call synced.
  graph-batched        : ONE captured graph over the 8-graph batch (replayed).
  graph-single x8      : the single-cell graph from 71's path, replayed 8 times
                         back to back (the "no batching" baseline for 8 cells).

Parity: batched-graph forces vs batched-eager forces, max|dF| < tol.

Usage (GPU box, via jobq):
    python scripts/73_cudagraph_batched.py --n-replicas 8 --nx 5 --ny 7 \
        --out results/raw/cudagraph/batched_cueq_fp32.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent


def _load_gate():
    path = REPO / "scripts" / "70_cudagraph_gate.py"
    spec = importlib.util.spec_from_file_location("cudagraph_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stats(laps_s):
    ms = 1e3 * np.asarray(laps_s)
    return {"ms_per_step_median": float(np.median(ms)),
            "ms_per_step_p10": float(np.percentile(ms, 10)),
            "ms_per_step_p90": float(np.percentile(ms, 90))}


def _time_synced(fn, geoms, n_warmup, n_steps):
    import torch
    gi = 0
    for _ in range(n_warmup):
        fn(geoms[gi]); gi += 1
    torch.cuda.synchronize()
    laps = []
    for _ in range(n_steps):
        t = time.perf_counter()
        fn(geoms[gi]); gi += 1
        torch.cuda.synchronize()
        laps.append(time.perf_counter() - t)
    return laps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cueq", choices=["cueq", "e3nn"])
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-replicas", type=int, default=8)
    ap.add_argument("--nx", type=int, default=5)
    ap.add_argument("--ny", type=int, default=7)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-warmup", type=int, default=15)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/cudagraph/batched.json")
    args = ap.parse_args()

    import torch
    from phosbench.common import (env_metadata, gpu_clock_temp, make_calc,
                                  make_supercell)
    gate = _load_gate()

    N = args.n_replicas
    atoms = make_supercell(args.nx, args.ny)
    natoms1 = len(atoms)
    print(f"[batched] {N}x {natoms1}-atom cells = {N*natoms1} atoms total, "
          f"{args.backend}/{args.dtype}", flush=True)
    calc = make_calc(args.backend, args.dtype, model=args.model, device=args.device)

    # single-cell evaluator (baseline) and N-replica batched evaluator
    fe1 = gate.FixedTopoForceEval(calc, atoms, device=args.device, n_replicas=1)
    feN = gate.FixedTopoForceEval(calc, atoms, device=args.device, n_replicas=N)
    print(f"[batched] single: {fe1.n_atoms} atoms {fe1.n_edges} edges | "
          f"batched: {feN.n_atoms} atoms {feN.n_edges} edges", flush=True)

    rng = np.random.default_rng(args.seed)
    base1 = atoms.get_positions().astype(np.float64)
    baseN = np.tile(base1, (N, 1))          # block layout matches Batch node order
    total = args.n_warmup + args.n_steps
    geomsN = [baseN + rng.normal(scale=0.01, size=baseN.shape) for _ in range(total)]
    geoms1 = [g[:natoms1] for g in geomsN]  # first block, for the single-cell arm

    rec = {
        "config": {"backend": args.backend, "dtype": args.dtype,
                   "model": args.model, "n_replicas": N,
                   "nx": args.nx, "ny": args.ny},
        "natoms_per_cell": natoms1, "natoms_total": feN.n_atoms,
        "n_edges_batched": feN.n_edges, "r_max": feN.r_max,
        "n_warmup": args.n_warmup, "n_steps": args.n_steps,
        "clocks_start": gpu_clock_temp() if args.device == "cuda" else None,
    }

    # --- eager batched reference ---
    e_e, f_e = feN.eager(geomsN[0]); f_e = f_e.clone()

    # --- capture both graphs ---
    capture_error = None
    try:
        fe1.capture(n_warmup=3)
        feN.capture(n_warmup=3)
        _, f_g = feN.replay(geomsN[0]); f_g = f_g.clone()
        max_dF = float((f_g - f_e).abs().max().item())
        rec["parity_max_dF_eV_per_A"] = max_dF
        rec["parity_pass"] = bool(max_dF < args.tol)
        print(f"[batched] parity (batched graph vs batched eager) "
              f"max|dF|={max_dF:.3e} eV/A "
              f"({'PASS' if rec['parity_pass'] else 'FAIL'})", flush=True)
    except Exception as exc:  # noqa: BLE001
        capture_error = f"{type(exc).__name__}: {exc}"
        rec["capture_error"] = capture_error[:2000]
        rec["parity_pass"] = False
        print(f"[batched] CAPTURE FAILED: {capture_error}", flush=True)

    if rec.get("parity_pass"):
        torch.cuda.reset_peak_memory_stats()
        # eager batched, synced
        rec["eager_batched"] = _stats(
            _time_synced(lambda g: feN.eager(g), geomsN, args.n_warmup, args.n_steps))
        # one graph over the batch
        rec["graph_batched"] = _stats(
            _time_synced(lambda g: feN.replay(g), geomsN, args.n_warmup, args.n_steps))
        rec["peak_vram_mib_batched"] = float(torch.cuda.max_memory_allocated() / 2**20)

        # baseline: N sequential single-cell graph replays per "step"
        def eight_singles(gN):
            for k in range(N):
                fe1.replay(gN[k * natoms1:(k + 1) * natoms1])
        rec["graph_single_x{}".format(N)] = _stats(
            _time_synced(eight_singles, geomsN, args.n_warmup, args.n_steps))

        gb = rec["graph_batched"]["ms_per_step_median"]
        gs = rec["graph_single_x{}".format(N)]["ms_per_step_median"]
        eb = rec["eager_batched"]["ms_per_step_median"]
        rec["ms_per_cell_batched_graph"] = gb / N
        rec["ms_per_cell_single_graph"] = gs / N
        rec["speedup_batched_vs_Nsingles"] = gs / gb if gb else None
        rec["speedup_batched_graph_vs_batched_eager"] = eb / gb if gb else None
        print(f"[batched] eager-batched {eb:.2f} ms | graph-batched {gb:.2f} ms "
              f"({gb/N:.3f} ms/cell) | {N}x single-graph {gs:.2f} ms "
              f"({gs/N:.3f} ms/cell) | batched vs {N}-singles "
              f"x{rec['speedup_batched_vs_Nsingles']:.2f}", flush=True)

    rec["clocks_end"] = gpu_clock_temp() if args.device == "cuda" else None
    rec["env"] = env_metadata()

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    print(f"[batched] wrote {out}", flush=True)
    return 0 if rec.get("parity_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
