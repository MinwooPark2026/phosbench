#!/usr/bin/env python
"""Stage B - throughput sweep over backend x dtype x model x system size.

Each (backend, dtype, model) configuration runs in a fresh subprocess so that an
OOM cannot poison the CUDA context of later configs; within a config, sizes run
small to large and the first OOM skips the remaining (larger) sizes.

Parent mode (default):
    python scripts/10_sweep_throughput.py --out results/raw/sweep_cuda.jsonl
Child modes:
    --config e3nn float32 medium md     # one (backend,dtype,model,mode), ALL sizes,
                                        # model loaded once; one RESULT_JSON line per size
    --single e3nn float64 medium 8 11 md  # one size only (used by nsys profiling wrappers)
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent


def _measure(atoms, calc, backend, dtype, model, mode, device, n_steps, n_warmup,
             nx, ny, env):
    from phosbench.common import time_force_calls, time_md

    timer = time_md if mode == "md" else time_force_calls
    stats = timer(atoms, calc, device, n_warmup=int(n_warmup), n_steps=int(n_steps))
    rec = {
        "config": {"backend": backend, "dtype": dtype, "model": model,
                   "device": device, "nx": int(nx), "ny": int(ny), "mode": mode},
        "natoms": len(atoms),
        **stats,
        "env": env,
    }
    print("RESULT_JSON " + json.dumps(rec), flush=True)


def run_single(backend, dtype, model, nx, ny, mode, device, n_steps, n_warmup):
    from phosbench.common import env_metadata, make_calc, make_supercell

    atoms = make_supercell(int(nx), int(ny))
    calc = make_calc(backend, dtype, model=model, device=device)
    _measure(atoms, calc, backend, dtype, model, mode, device, n_steps, n_warmup,
             nx, ny, env_metadata())
    return 0


def run_config(backend, dtype, model, mode, device, n_steps, n_warmup, max_atoms):
    """All sizes for one (backend, dtype, model, mode): the model loads once.

    OOM at a given size is caught, recorded, and ends the ladder (sizes are
    monotonic). Earlier RESULT_JSON lines survive even if the process dies.
    """
    from phosbench.common import (env_metadata, make_calc, make_supercell,
                                  size_ladder)

    calc = make_calc(backend, dtype, model=model, device=device)
    env = env_metadata()
    for nx, ny, natoms in size_ladder(int(max_atoms)):
        try:
            atoms = make_supercell(nx, ny)
            _measure(atoms, calc, backend, dtype, model, mode, device,
                     n_steps, n_warmup, nx, ny, env)
        except Exception as exc:  # noqa: BLE001 - record and stop the ladder
            is_oom = "out of memory" in str(exc).lower() or "OutOfMemory" in type(exc).__name__
            print("RESULT_JSON " + json.dumps({
                "config": {"backend": backend, "dtype": dtype, "model": model,
                           "device": device, "nx": nx, "ny": ny, "mode": mode},
                "natoms": natoms,
                "error": "oom" if is_oom else f"crash: {exc!r}"[:500],
            }), flush=True)
            if device == "cuda":
                import torch

                torch.cuda.empty_cache()
            break
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--single", nargs=6, metavar=("BACKEND", "DTYPE", "MODEL", "NX", "NY", "MODE"))
    p.add_argument("--config", nargs=4, metavar=("BACKEND", "DTYPE", "MODEL", "MODE"))
    p.add_argument("--backends", default="e3nn,cueq")
    p.add_argument("--dtypes", default="float32")
    p.add_argument("--models", default="medium")
    p.add_argument("--device", default="cuda")
    p.add_argument("--modes", default="md,force_call")
    p.add_argument("--max-atoms", type=int, default=40000)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--timeout", type=int, default=7200, help="per-config timeout (s)")
    p.add_argument("--out", default="results/raw/sweep.jsonl")
    args = p.parse_args()

    if args.single:
        backend, dtype, model, nx, ny, mode = args.single
        return run_single(backend, dtype, model, nx, ny, mode,
                          args.device, args.n_steps, args.n_warmup)
    if args.config:
        backend, dtype, model, mode = args.config
        return run_config(backend, dtype, model, mode, args.device,
                          args.n_steps, args.n_warmup, args.max_atoms)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    configs = list(itertools.product(
        args.models.split(","), args.backends.split(","),
        args.dtypes.split(","), args.modes.split(",")))
    t0 = time.time()

    with open(out, "a") as fh:
        for i, (model, backend, dtype, mode) in enumerate(configs, 1):
            tag = f"{model}/{backend}/{dtype}/{mode}"
            print(f"[{i}/{len(configs)}] CONFIG {tag} (elapsed {time.time()-t0:.0f}s)",
                  flush=True)
            cmd = [sys.executable, __file__,
                   "--config", backend, dtype, model, mode,
                   "--device", args.device,
                   "--max-atoms", str(args.max_atoms),
                   "--n-steps", str(args.n_steps),
                   "--n-warmup", str(args.n_warmup)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=args.timeout)
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                n = _harvest(stdout, fh)
                fh.write(json.dumps({"config": {
                    "backend": backend, "dtype": dtype, "model": model,
                    "device": args.device, "mode": mode},
                    "error": f"config timeout after {n} sizes"}) + "\n")
                fh.flush()
                continue
            n = _harvest(proc.stdout, fh)
            if n == 0:
                fh.write(json.dumps({"config": {
                    "backend": backend, "dtype": dtype, "model": model,
                    "device": args.device, "mode": mode},
                    "error": f"config crash rc={proc.returncode}",
                    "stderr_tail": (proc.stderr or "")[-2000:]}) + "\n")
                fh.flush()
            print(f"    -> {n} measurements", flush=True)
    print(f"[phosbench] sweep finished in {time.time()-t0:.0f}s -> {out}")
    return 0


def _harvest(stdout: str, fh) -> int:
    """Copy every RESULT_JSON line from a child's stdout into the JSONL sink."""
    n = 0
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT_JSON "):
            fh.write(line[len("RESULT_JSON "):] + "\n")
            n += 1
    fh.flush()
    return n


if __name__ == "__main__":
    raise SystemExit(main())
