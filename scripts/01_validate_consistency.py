#!/usr/bin/env python
"""Stage A - numerical consistency gates before any expensive run (3-cell design).

Cells: e3nn/float64 (reference), e3nn/float32, cueq/float32 (production candidate).

  GATE 1 (hard stop)  cueq/fp32 vs e3nn/fp32: same math, different kernels -
          |dE| < 1 meV/atom and max|dF| < 1 meV/A on a ~140-atom supercell.
  GATE 2  e3nn/fp64 path is real: model parameter dtypes are actually float64.
  PROBE   cueq/float64 is documented broken upstream (MACE #1203/#1298; cuEq
          changelog removed fp32-math/fp64-IO). We attempt it EXPECTING failure
          and record the outcome - works / raises / runs-with-wrong-numbers.
          Outcome is a deployment finding, not a gate.
  REPORT  precision deltas fp32-vs-fp64 (e3nn): small-scale preview of Stage D.
  REPORT  load / first-call (JIT, autotune) / steady-call latencies per cell.
"""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phosbench.common import (env_metadata, make_calc, make_supercell,
                              write_json)

GATE_E_MEV_PER_ATOM = 1.0    # cueq-fp32 vs e3nn-fp32
GATE_F_MEV_PER_A = 1.0


def eval_config(backend, dtype, model="medium", device="cuda"):
    atoms = make_supercell(5, 7)  # 140 atoms
    t0 = time.perf_counter()
    calc = make_calc(backend, dtype, model=model, device=device)
    t_load = time.perf_counter() - t0
    atoms.calc = calc
    t0 = time.perf_counter()
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    t_first = time.perf_counter() - t0
    atoms.rattle(stdev=1e-4, seed=1)
    t0 = time.perf_counter()
    atoms.get_forces()
    t_steady = time.perf_counter() - t0
    # verify the requested precision actually propagated to the weights
    import torch

    dtypes = {str(p.dtype) for p in calc.models[0].parameters()}
    return {"energy_eV": e, "forces": f, "natoms": len(atoms),
            "param_dtypes": sorted(dtypes), "t_load_s": t_load,
            "t_first_call_s": t_first, "t_steady_call_s": t_steady}


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="medium")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = {"env": env_metadata(), "model": args.model,
           "gates": {}, "probe": {}, "reports": {}}
    results = {}
    for backend, dtype in (("e3nn", "float64"), ("e3nn", "float32"),
                           ("cueq", "float32")):
        key = f"{backend}/{dtype}"
        print(f"--- evaluating {key}", flush=True)
        r = eval_config(backend, dtype, model=args.model, device=args.device)
        results[key] = r
        print(f"    E={r['energy_eV']:.8f} eV  params={r['param_dtypes']}  "
              f"load={r['t_load_s']:.1f}s first={r['t_first_call_s']:.2f}s "
              f"steady={r['t_steady_call_s']:.3f}s")
        out["reports"][key] = {k: v for k, v in r.items() if k != "forces"}

    natoms = results["e3nn/float64"]["natoms"]

    def compare(a, b):
        de = abs(results[a]["energy_eV"] - results[b]["energy_eV"]) / natoms * 1e3
        df = float(np.abs(results[a]["forces"] - results[b]["forces"]).max()) * 1e3
        return {"dE_meV_per_atom": de, "max_dF_meV_per_A": df}

    # GATE 1: backend parity at matched precision
    g = compare("cueq/float32", "e3nn/float32")
    g["pass"] = (g["dE_meV_per_atom"] < GATE_E_MEV_PER_ATOM
                 and g["max_dF_meV_per_A"] < GATE_F_MEV_PER_A)
    out["gates"]["cueq_vs_e3nn_fp32"] = g
    print(f"GATE1 cueq-vs-e3nn @fp32: dE={g['dE_meV_per_atom']:.2e} meV/at "
          f"max dF={g['max_dF_meV_per_A']:.2e} meV/A -> "
          f"{'PASS' if g['pass'] else 'FAIL'}")

    # GATE 2: fp64 reference path is real
    g2 = {"param_dtypes": results["e3nn/float64"]["param_dtypes"],
          "pass": results["e3nn/float64"]["param_dtypes"] == ["torch.float64"]}
    out["gates"]["fp64_path_real"] = g2
    print(f"GATE2 e3nn fp64 weights: {g2['param_dtypes']} -> "
          f"{'PASS' if g2['pass'] else 'FAIL'}")

    # REPORT: precision cost preview (not a gate - this is a Stage D measurement)
    out["reports"]["precision_e3nn"] = compare("e3nn/float32", "e3nn/float64")
    r = out["reports"]["precision_e3nn"]
    print(f"REPORT fp32-vs-fp64 (e3nn): dE={r['dE_meV_per_atom']:.3e} meV/at "
          f"max dF={r['max_dF_meV_per_A']:.3e} meV/A")

    # PROBE: cueq/float64 - expected broken upstream
    print("--- probing cueq/float64 (expected to fail or fall back)", flush=True)
    try:
        rp = eval_config("cueq", "float64", model=args.model, device=args.device)
        d = compare_probe(rp, results["e3nn/float64"], natoms)
        out["probe"]["cueq_float64"] = {
            "outcome": "ran", **d, "param_dtypes": rp["param_dtypes"],
            "verdict": ("numbers match fp64 reference - possibly silent e3nn "
                        "fallback, check nsys kernel names" if d["max_dF_meV_per_A"] < 1.0
                        else "ran but numbers deviate from fp64 reference"),
        }
    except Exception as exc:
        out["probe"]["cueq_float64"] = {
            "outcome": "raised", "error": repr(exc),
            "traceback_tail": traceback.format_exc()[-1500:],
            "verdict": "cueq+fp64 unsupported, as documented (MACE #1203)",
        }
    print(f"PROBE cueq/fp64: {out['probe']['cueq_float64']['outcome']} - "
          f"{out['probe']['cueq_float64']['verdict']}")

    write_json(Path(__file__).resolve().parent.parent
               / "results/raw/stage_a_consistency.json", out)
    ok = all(v.get("pass") for v in out["gates"].values())
    print(f"STAGE_A_CONSISTENCY: {'GO' if ok else 'NO-GO'}")
    return 0 if ok else 1


def compare_probe(rp, ref, natoms):
    de = abs(rp["energy_eV"] - ref["energy_eV"]) / natoms * 1e3
    df = float(np.abs(rp["forces"] - ref["forces"]).max()) * 1e3
    return {"dE_meV_per_atom": de, "max_dF_meV_per_A": df}


if __name__ == "__main__":
    raise SystemExit(main())
