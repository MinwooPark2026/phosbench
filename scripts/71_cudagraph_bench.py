#!/usr/bin/env python
"""phosbench part 2 - CUDA-graph vs eager force-eval benchmark.

Quantifies how much of the host / launch-latency overhead identified in README
finding 3 a CUDA graph reclaims.  For each system size we time the frozen-topology
MACE force evaluation three ways, all on IDENTICAL small position perturbations
(0.01 A rms, neighbour list held valid - see 70_cudagraph_gate.py):

  eager-synced   : eager forward, torch.cuda.synchronize() after EVERY step.
                   This is the sweep harness's number - it exposes full per-step
                   launch latency (the pessimistic, launch-latency-bound case).
  eager-free     : eager forward, NO per-step sync (sync only at the end of the
                   lap window).  cuEq pipelines launches ahead, so this hides some
                   launch latency - the README's 31 ms vs 78 ms free/synced gap.
  graph-replay   : replay the captured CUDA graph, per-step synced (the graph
                   collapses the whole launch sequence into one host call, so the
                   synced number is the fair comparison against eager-synced).

Timing discipline mirrors phosbench.common: warmup >= 10, median + p10/p90 of
per-step laps, GPU clock/temp logged at start and end via nvidia-smi.

The graph path REUSES FixedTopoForceEval from 70_cudagraph_gate.py and re-runs the
parity gate inline before trusting any graph number.

Usage (on the GPU box, via jobq):
    python scripts/71_cudagraph_bench.py --backend cueq --dtype float32 \
        --sizes 140,512,2944 --out results/raw/cudagraph/bench_cueq_fp32.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

# Size -> (nx, ny) chosen so 4*nx*ny hits the three headline sizes exactly.
#   140  -> 5 x 7  (=140)   512 -> 8 x 11 (=352? no) -> use exact factorisations
# The canonical cell is 4 atoms; supercell natoms = 4*nx*ny.  Pick (nx,ny) that
# multiply to the target/4 and stay near-square in Cartesian terms (ny ~ 1.4 nx).
SIZE_TO_NXNY = {
    140: (5, 7),      # 4*5*7   = 140
    512: (8, 16),     # 4*8*16  = 512
    2944: (16, 46),   # 4*16*46 = 2944
}


def _resolve_nxny(size):
    if size in SIZE_TO_NXNY:
        return SIZE_TO_NXNY[size]
    # fall back: factor size/4 into a near-square (nx, ny) with ny >= nx
    q = size // 4
    best = None
    for nx in range(1, int(q ** 0.5) + 2):
        if q % nx == 0:
            ny = q // nx
            best = (nx, ny)
    if best is None or 4 * best[0] * best[1] != size:
        raise ValueError(f"size {size} is not 4*nx*ny for integer nx,ny")
    return best


# --------------------------------------------------------------------------- #
# Timing primitives (phosbench discipline)
# --------------------------------------------------------------------------- #

def _stats_from_laps(laps_s, natoms):
    laps_ms = 1e3 * np.asarray(laps_s)
    return {
        "n_steps": int(len(laps_ms)),
        "ms_per_step_median": float(np.median(laps_ms)),
        "ms_per_step_p10": float(np.percentile(laps_ms, 10)),
        "ms_per_step_p90": float(np.percentile(laps_ms, 90)),
        "ms_per_step_mean": float(np.mean(laps_ms)),
        "us_per_atom_step": float(1e3 * np.median(laps_ms) / natoms),
    }


def time_eager_synced(fe, geoms, n_warmup, n_steps):
    """Eager forward, sync after every step. geoms: (n_warmup+n_steps, natoms, 3)."""
    import torch
    gi = 0
    for _ in range(n_warmup):
        fe.eager(geoms[gi]); gi += 1
    torch.cuda.synchronize()
    laps = []
    for _ in range(n_steps):
        t = time.perf_counter()
        fe.eager(geoms[gi]); gi += 1
        torch.cuda.synchronize()
        laps.append(time.perf_counter() - t)
    return laps


def time_eager_free(fe, geoms, n_warmup, n_steps):
    """Eager forward, NO per-step sync; the lap window is timed wall-to-wall and
    divided by n_steps (the free-running throughput the README reports)."""
    import torch
    gi = 0
    for _ in range(n_warmup):
        fe.eager(geoms[gi]); gi += 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        fe.eager(geoms[gi]); gi += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    # Per-step laps are not individually measurable without syncing (which would
    # defeat the point), so free-running reports a single per-step number from the
    # window; we replicate it n_steps times so the stats helper still works and
    # p10/p90 collapse onto the mean (disclosed: free-running has no per-lap spread).
    per = wall / n_steps
    return [per] * n_steps


def time_graph_synced(fe, geoms, n_warmup, n_steps):
    """Replay the captured graph, sync after every step (fair vs eager-synced)."""
    import torch
    gi = 0
    for _ in range(n_warmup):
        fe.replay(geoms[gi]); gi += 1
    torch.cuda.synchronize()
    laps = []
    for _ in range(n_steps):
        t = time.perf_counter()
        fe.replay(geoms[gi]); gi += 1
        torch.cuda.synchronize()
        laps.append(time.perf_counter() - t)
    return laps


def time_graph_free(fe, geoms, n_warmup, n_steps):
    """Replay the graph, no per-step sync (host launches replays back to back)."""
    import torch
    gi = 0
    for _ in range(n_warmup):
        fe.replay(geoms[gi]); gi += 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        fe.replay(geoms[gi]); gi += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    per = wall / n_steps
    return [per] * n_steps


# --------------------------------------------------------------------------- #
# Per-size benchmark
# --------------------------------------------------------------------------- #

_GATE_MOD = None


def _load_gate():
    """Load 70_cudagraph_gate.py by path (its name starts with a digit)."""
    global _GATE_MOD
    if _GATE_MOD is None:
        import importlib.util
        path = REPO / "scripts" / "70_cudagraph_gate.py"
        spec = importlib.util.spec_from_file_location("cudagraph_gate", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GATE_MOD = mod
    return _GATE_MOD

def bench_size(backend, dtype, size, model, device, n_warmup, n_steps, tol, seed):
    import torch
    from phosbench.common import gpu_clock_temp, make_calc, make_supercell
    # reuse the frozen-topology evaluator + parity from the gate script (module
    # name starts with a digit, so load it by file path rather than import_module)
    gate = _load_gate()

    nx, ny = _resolve_nxny(size)
    atoms = make_supercell(nx, ny)
    natoms = len(atoms)
    print(f"\n[bench] === {size} atoms ({nx}x{ny}={natoms}) {backend}/{dtype} ===",
          flush=True)

    calc = make_calc(backend, dtype, model=model, device=device)
    fe = gate.FixedTopoForceEval(calc, atoms, device=device)
    model_dtype = fe.model_dtype
    print(f"[bench] {fe.n_atoms} atoms, {fe.n_edges} edges, r_max={fe.r_max} A",
          flush=True)

    rng = np.random.default_rng(seed)
    base = atoms.get_positions().astype(np.float64)
    total = n_warmup + n_steps
    # Pre-generate all perturbed geometries as fp32/fp64 device-ready arrays so the
    # timing loop measures the force eval, not numpy RNG.  Small rattle keeps the
    # frozen neighbour list valid (max disp << r_max skin).
    geoms = [base + rng.normal(scale=0.01, size=base.shape) for _ in range(total)]
    max_disp = float(max(np.abs(g - base).max() for g in geoms))

    rec = {
        "config": {"backend": backend, "dtype": dtype, "model": model,
                   "device": device, "nx": nx, "ny": ny},
        "natoms": natoms, "n_edges": fe.n_edges, "r_max": fe.r_max,
        "n_warmup": n_warmup, "n_steps": n_steps,
        "max_perturb_disp_A": max_disp,
    }

    # --- eager arms (always available) ---
    clocks0 = gpu_clock_temp() if device == "cuda" else None
    torch.cuda.reset_peak_memory_stats()
    rec["eager_synced"] = _stats_from_laps(
        time_eager_synced(fe, geoms, n_warmup, n_steps), natoms)
    rec["eager_free"] = _stats_from_laps(
        time_eager_free(fe, geoms, n_warmup, n_steps), natoms)
    rec["peak_vram_mib_eager"] = float(torch.cuda.max_memory_allocated() / 2**20)

    # --- capture + inline parity gate ---
    capture_error = None
    try:
        fe.capture(n_warmup=3)
        # inline parity on a fresh geometry
        pg = base + rng.normal(scale=0.01, size=base.shape)
        _, f_e = fe.eager(pg); f_e = f_e.clone()
        _, f_g = fe.replay(pg); f_g = f_g.clone()
        max_dF = float((f_g - f_e).abs().max().item())
        rec["parity_max_dF_eV_per_A"] = max_dF
        rec["parity_pass"] = bool(max_dF < tol)
        print(f"[bench] inline parity max|dF|={max_dF:.3e} eV/A "
              f"({'PASS' if rec['parity_pass'] else 'FAIL'})", flush=True)
    except Exception as exc:  # noqa: BLE001
        capture_error = f"{type(exc).__name__}: {exc}"
        rec["capture_error"] = capture_error[:2000]
        rec["parity_pass"] = False
        print(f"[bench] CAPTURE FAILED: {capture_error}", flush=True)

    if rec.get("parity_pass"):
        torch.cuda.reset_peak_memory_stats()
        rec["graph_synced"] = _stats_from_laps(
            time_graph_synced(fe, geoms, n_warmup, n_steps), natoms)
        rec["graph_free"] = _stats_from_laps(
            time_graph_free(fe, geoms, n_warmup, n_steps), natoms)
        rec["peak_vram_mib_graph"] = float(torch.cuda.max_memory_allocated() / 2**20)

        es = rec["eager_synced"]["ms_per_step_median"]
        ef = rec["eager_free"]["ms_per_step_median"]
        gs = rec["graph_synced"]["ms_per_step_median"]
        rec["speedup_graph_vs_eager_synced"] = es / gs if gs else None
        rec["speedup_graph_vs_eager_free"] = ef / gs if gs else None

    rec["clocks_start"] = clocks0
    rec["clocks_end"] = gpu_clock_temp() if device == "cuda" else None

    _print_row(rec)
    # free the model / context before the next size
    del fe, calc
    torch.cuda.empty_cache()
    return rec


def _print_row(rec):
    es = rec.get("eager_synced", {}).get("ms_per_step_median")
    ef = rec.get("eager_free", {}).get("ms_per_step_median")
    gs = rec.get("graph_synced", {}).get("ms_per_step_median")
    parity = rec.get("parity_max_dF_eV_per_A")
    su_s = rec.get("speedup_graph_vs_eager_synced")
    su_f = rec.get("speedup_graph_vs_eager_free")
    def f(x, fmt="{:.2f}"):
        return fmt.format(x) if isinstance(x, (int, float)) else "n/a"
    print(f"[bench] ROW n={rec['natoms']:>5} | eager-sync {f(es)} ms | "
          f"eager-free {f(ef)} ms | graph {f(gs)} ms | "
          f"x{f(su_s)} vs sync | x{f(su_f)} vs free | "
          f"parity {f(parity, '{:.1e}')} eV/A", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cueq", choices=["cueq", "e3nn"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--sizes", default="140,512,2944")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-warmup", type=int, default=15)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/cudagraph/bench.json")
    args = ap.parse_args()

    from phosbench.common import env_metadata
    sizes = [int(s) for s in args.sizes.split(",")]
    t0 = time.time()
    rows = []
    for size in sizes:
        try:
            rows.append(bench_size(args.backend, args.dtype, size, args.model,
                                   args.device, args.n_warmup, args.n_steps,
                                   args.tol, args.seed))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows.append({"config": {"backend": args.backend, "dtype": args.dtype,
                                    "size": size},
                         "error": f"{type(exc).__name__}: {exc}"[:1000]})

    payload = {
        "kind": "cudagraph_bench",
        "config": {"backend": args.backend, "dtype": args.dtype,
                   "model": args.model, "device": args.device},
        "rows": rows,
        "wall_s": time.time() - t0,
        "env": env_metadata(),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n[bench] wrote {out} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
