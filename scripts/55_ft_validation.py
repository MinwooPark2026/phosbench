#!/usr/bin/env python
"""'After the model fix' validation column: phonons + elastic for the
GAP-20-fine-tuned model, on its own relaxed structure.

Deliberately self-consistent with the study's own findings:
  - displacement amplitude per cell follows the hybrid policy this study
    derived (fp64 -> 0.01 A, cuEq/fp32 -> 0.05 A);
  - elastic constants use energy-curvature fits only (slab-stress bug);
  - outputs land in separate files (phonons_ft/, ft_validation.json) so the
    zero-shot study artifacts stay untouched.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phosbench.common import env_metadata, make_calc, write_json
from phosbench.phonons import band_path_SXGYS, build_phonon, min_freq_near_gamma

REPO = Path(__file__).resolve().parent.parent
EV_PER_A2_TO_N_PER_M = 16.0218
LIT = {"a": 4.62, "b": 3.30, "C11": 24.0, "C22": 103.0}


def elastic_energy_curvature(atoms0, calc, strains, fmax=2e-3):
    """C11/C22 (N/m) from relaxed-ion energy curvature on a 2x3 supercell."""
    from ase.optimize import FIRE

    base = atoms0.repeat((2, 3, 1))
    base.calc = calc
    FIRE(base, logfile=None).run(fmax=fmax, steps=1000)
    e0 = float(base.get_potential_energy())
    cell0 = np.asarray(base.get_cell()).copy()
    A0 = float(abs(np.linalg.det(cell0[:2, :2])))

    out = {}
    eps_pts = sorted({s * sgn for s in strains for sgn in (1.0, -1.0)})
    for axis, name in ((0, "C11"), (1, "C22")):
        eps_l, en_l = [0.0], [e0]
        for eps in eps_pts:
            a = base.copy()
            m = np.eye(3)
            m[axis, axis] += eps
            a.set_cell(cell0 @ m, scale_atoms=True)
            a.calc = calc
            FIRE(a, logfile=None).run(fmax=fmax, steps=1000)
            eps_l.append(eps)
            en_l.append(float(a.get_potential_energy()))
        c2 = np.polyfit(eps_l, en_l, 2)[0]
        out[name] = 2.0 * c2 / A0 * EV_PER_A2_TO_N_PER_M
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="finetune/phos_ft2_stagetwo.model")
    p.add_argument("--structure", default="structures/phosphorene_ft2.extxyz")
    p.add_argument("--strains", default="0.005,0.01")
    args = p.parse_args()

    from ase.io import read

    model = str(REPO / args.model)
    if not Path(model).exists():
        model = str(REPO / "finetune/phos_ft2.model")
    atoms0 = read(REPO / args.structure)
    a0, b0 = atoms0.cell.lengths()[:2]
    print(f"FT structure: a={a0:.4f} b={b0:.4f} "
          f"(lit {LIT['a']}/{LIT['b']}; da={100*(a0/LIT['a']-1):+.2f}%)")

    summary = {"model": model, "lattice": {"a_A": a0, "b_A": b0,
               "da_pct": 100 * (a0 / LIT["a"] - 1),
               "db_pct": 100 * (b0 / LIT["b"] - 1)},
               "cells": {}, "lit": LIT, "env": env_metadata()}

    # displacement per cell follows the hybrid policy derived in Stage D
    for backend, dtype, disp in (("e3nn", "float64", 0.01),
                                 ("cueq", "float32", 0.05)):
        key = f"{backend}/{dtype}"
        print(f"--- {key} (displacement {disp} A)", flush=True)
        calc = make_calc(backend, dtype, model=model, device="cuda")
        rec = {"displacement_A": disp}
        try:
            ph = build_phonon(atoms0.copy(), calc, supercell=(4, 6, 1),
                              displacement=disp)
            band = band_path_SXGYS(ph, npoints=51)
            np.savez(REPO / f"results/raw/phonons_ft/band_{backend}_{dtype}.npz",
                     **{k: np.asarray(v) for k, v in band.items()
                        if k in ("distances", "frequencies_THz", "label_distances")},
                     labels=np.array(band["labels"]))
            rec["min_freq_near_gamma_THz"] = float(min_freq_near_gamma(ph))
            rec["max_freq_THz"] = float(np.max(band["frequencies_THz"]))
            rec["n_imag_band_points"] = int(np.sum(
                np.asarray(band["frequencies_THz"]) < -1e-3))
            print(f"    phonons: min {rec['min_freq_near_gamma_THz']:+.4f} THz, "
                  f"max {rec['max_freq_THz']:.3f} THz, "
                  f"imag pts {rec['n_imag_band_points']}")
        except Exception as exc:  # noqa: BLE001
            rec["phonon_error"] = repr(exc)[:300]
            print(f"    phonons FAILED: {exc!r}")
        try:
            el = elastic_energy_curvature(
                atoms0.copy(), calc,
                [float(s) for s in args.strains.split(",")])
            rec.update(el)
            print(f"    elastic: C11={el['C11']:.1f} C22={el['C22']:.1f} N/m "
                  f"(lit {LIT['C11']}/{LIT['C22']})")
        except Exception as exc:  # noqa: BLE001
            rec["elastic_error"] = repr(exc)[:300]
            print(f"    elastic FAILED: {exc!r}")
        summary["cells"][key] = rec

    (REPO / "results/raw/phonons_ft").mkdir(parents=True, exist_ok=True)
    write_json(REPO / "results/raw/ft_validation.json", summary)
    print("FT_VALIDATION_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
