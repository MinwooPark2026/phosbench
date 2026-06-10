#!/usr/bin/env python
"""Recompute elastic constants with the energy-curvature path as primary.

scripts/90_diag_stress_hf.py shows MACE 0.3.16 analytic stress on this slab is
~17.8x smaller than the Hellmann-Feynman dE/deps of its own energy surface, so
the stress-slope constants in results/raw/elastic.json are systematically low.
The energies in the saved stress_strain_table are unaffected; this script
re-fits them (no GPU needed) and writes results/raw/elastic_corrected.json:

  C11_Nm, C22_Nm   energy-curvature fits, 2*c2/A0 (primary, assumption-free)
  C12_Nm, C21_Nm   stress cross-slopes scaled by the measured HF factor
                   (no uniaxial-energy equivalent exists; flagged with caveat)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EV_PER_A2_TO_N_PER_M = 16.0218
HF_FACTOR_DEFAULT = 17.806   # measured by scripts/90_diag_stress_hf.py


def quad_c(eps, en):
    return np.polyfit(eps, en, 2)


def refit(cell: dict, hf: float) -> dict:
    A0 = cell["area_A2"]
    rows = cell["stress_strain_table"]
    out = {}
    for axis in ("xx", "yy"):
        pts = [(r["eps"], r["energy_eV"]) for r in rows
               if r["strain_axis"] in (axis, "0")]
        eps, en = map(np.asarray, zip(*sorted(set(pts))))
        coef = quad_c(eps, en)
        pred = np.polyval(coef, eps)
        ss = float(np.sum((en - en.mean()) ** 2))
        r2 = 1.0 - float(np.sum((en - pred) ** 2)) / ss if ss else float("nan")
        out[axis] = {"C_Nm": 2.0 * coef[0] / A0 * EV_PER_A2_TO_N_PER_M, "R2": r2,
                     "n_points": len(eps)}
    c12 = cell["fits"]["C12"]["slope_eV_A2"] * EV_PER_A2_TO_N_PER_M * hf
    c21 = cell["fits"]["C21"]["slope_eV_A2"] * EV_PER_A2_TO_N_PER_M * hf
    return {
        "ok": True,
        "C11_Nm": out["xx"]["C_Nm"],
        "C22_Nm": out["yy"]["C_Nm"],
        "C12_Nm": c12,
        "C21_Nm": c21,
        "anisotropy_C22_over_C11": out["yy"]["C_Nm"] / out["xx"]["C_Nm"],
        "R2": {"xx": out["xx"]["R2"], "yy": out["yy"]["R2"]},
        "method": {
            "C11_C22": "energy curvature 2*c2/A0 (relaxed-ion, primary)",
            "C12_C21": f"stress cross-slope x HF factor {hf} "
                       "(MACE slab stress bug workaround - see "
                       "scripts/90_diag_stress_hf.py)",
        },
        "stress_path_raw_Nm": {k: cell.get(k) for k in
                               ("C11_Nm", "C22_Nm", "C12_Nm", "C21_Nm")},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default=str(REPO / "results/raw/elastic.json"))
    p.add_argument("--out", default=str(REPO / "results/raw/elastic_corrected.json"))
    p.add_argument("--hf-factor", type=float, default=HF_FACTOR_DEFAULT)
    args = p.parse_args()

    src = json.load(open(args.inp))
    out = {"source": args.inp, "hf_factor": args.hf_factor,
           "note": "see module docstring; energies are bug-free, stress is not",
           "lit_reference_PBE": src.get("lit_reference_PBE"),
           "env": src.get("env"), "cells": {}}
    print("cell              C11(N/m)  C22(N/m)  C12*  C22/C11   R2(xx/yy)")
    for name, cell in src.get("cells", {}).items():
        if not isinstance(cell, dict) or not cell.get("ok"):
            out["cells"][name] = cell
            continue
        r = refit(cell, args.hf_factor)
        out["cells"][name] = r
        print("%-16s %9.2f %9.2f %5.1f   %6.2f   %.5f/%.5f" % (
            name, r["C11_Nm"], r["C22_Nm"], r["C12_Nm"],
            r["anisotropy_C22_over_C11"], r["R2"]["xx"], r["R2"]["yy"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[phosbench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
