#!/usr/bin/env python
"""Stage D2 - anisotropic in-plane elastic constants per backend/dtype cell.

Phosphorene's hinge-like pucker makes the armchair direction (x) anomalously
soft: PBE literature puts C11_2D near 24 N/m against ~103 N/m along zigzag
(y), an anisotropy of ~4. The soft direction is the discriminating signal
here - its stresses at 0.25-1% strain are tiny (~1e-3 eV/A^2), so a numerical
cell that corrupts small stresses shows up directly as a wrong C11 or a
ragged fit, long before it would show in lattice constants.

Method per cell: the canonical relaxed monolayer as a 2x3 (24-atom)
supercell. A perfect supercell has exactly the primitive cell's stress in
exact arithmetic, but the extra atoms average fp32 kernel noise in the virial
and give FIRE a better-conditioned landscape once strain breaks symmetry, at
negligible cost at this size. Each strain eps is applied along xx and yy
separately (cell' = cell @ (I + eps_matrix), scale_atoms=True), ions are
re-relaxed at fixed cell (FIRE to --fmax, pucker height free to respond), and
the analytic stress + total energy are recorded.

Primary constants: linear fits of sigma*Lz vs eps (relaxed-ion 2D constants;
multiplying by the box height Lz removes the arbitrary vacuum from the 3D
stress; eV/A^2 -> N/m via 16.0218). C12 is the cross slope sigma_yy*Lz vs
eps_xx; its symmetry partner (sigma_xx*Lz vs eps_yy) is recorded as C21 - the
C12/C21 gap is a free internal consistency check. Secondary crosscheck:
energy-curvature fits E = c0 + c1*eps + c2*eps^2 with C_2D = 2*c2/A0.

The lattice is held at the canonical fp64 geometry for every cell so configs
are compared at identical geometry; any residual eps=0 stress (e.g. when
--model differs from the structure's relaxation model) is absorbed by the
fit intercepts and recorded explicitly.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phosbench.common import env_metadata, make_calc, make_supercell, write_json

REPO = Path(__file__).resolve().parent.parent

EV_PER_A2_TO_N_PER_M = 16.0218

LITERATURE = {
    # monolayer phosphorene, relaxed-ion 2D constants, common PBE values
    # (see docs/case-study.md for refs)
    "C11_Nm": 24.0,    # armchair (x): soft - strain unfolds the pucker
    "C22_Nm": 103.0,   # zigzag  (y): stiff covalent backbone
    "C12_Nm": 18.0,
}

MODEL_URLS = {
    # known release assets, used when the short tag is not yet known to the
    # installed mace-torch (tags newer than the pinned 0.3.16 can 404)
    "medium-mpa-0": "https://github.com/ACEsuit/mace-foundations/releases/"
                    "download/mace_mpa_0/mace-mpa-0-medium.model",
    "medium-omat-0": "https://github.com/ACEsuit/mace-foundations/releases/"
                     "download/mace_omat_0/mace-omat-0-medium.model",
}


def default_model() -> str:
    """Stage-A winner from configs/model_choice.json if present, else mpa-0."""
    try:
        payload = json.loads((REPO / "configs" / "model_choice.json").read_text())
        model = payload if isinstance(payload, str) else payload.get("model")
        if isinstance(model, str) and model:
            return model
    except (OSError, ValueError):
        pass
    return "medium-mpa-0"


def make_calc_resolved(backend, dtype, model, device):
    """make_calc, falling back to the pinned release URL if the tag fails."""
    try:
        return make_calc(backend, dtype, model=model, device=device), model
    except Exception as exc:
        url = MODEL_URLS.get(model)
        if url is None:
            raise
        print(f"[phosbench] model tag {model!r} failed ({exc!r}); "
              f"retrying release URL", flush=True)
        return make_calc(backend, dtype, model=url, device=device), url


def relax_ions(atoms, fmax, steps=1000):
    """FIRE on positions only (cell fixed); returns (converged, residual fmax)."""
    from ase.optimize import FIRE

    converged = FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)
    fres = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    return bool(converged), fres


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "R2": r2}


def quadfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    coef = np.polyfit(x, y, 2)
    pred = np.polyval(coef, x)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"c2": float(coef[0]), "c1": float(coef[1]), "c0": float(coef[2]),
            "R2": r2}


def _row(axis, eps, stress_voigt, energy, fres, converged):
    return {"strain_axis": axis, "eps": float(eps),
            "stress_voigt_eV_A3": np.asarray(stress_voigt).tolist(),
            "energy_eV": float(energy), "fmax_residual_eV_A": fres,
            "converged": bool(converged)}


def measure_cell(backend, dtype, model, device, strains, fmax):
    calc, resolved = make_calc_resolved(backend, dtype, model, device)
    base = make_supercell(2, 3)
    base.calc = calc
    conv0, fres0 = relax_ions(base, fmax)
    e0 = float(base.get_potential_energy())
    s0 = base.get_stress(voigt=True)            # eV/A^3, (xx,yy,zz,yz,xz,xy)
    cell0 = np.asarray(base.get_cell()).copy()
    Lz = float(cell0[2, 2])
    A0 = float(abs(np.linalg.det(cell0[:2, :2])))
    print(f"    eps=0: E={e0:.6f} eV  sxx*Lz={s0[0] * Lz:+.2e}  "
          f"syy*Lz={s0[1] * Lz:+.2e} eV/A^2  fres={fres0:.1e}"
          f"{'' if conv0 else '  NOT CONVERGED'}", flush=True)

    eps_points = sorted({sgn * s for s in strains for sgn in (1.0, -1.0)})
    table = [_row("0", 0.0, s0, e0, fres0, conv0)]
    series = {}
    for axis, name in ((0, "xx"), (1, "yy")):
        # the relaxed eps=0 point is shared by both strain directions
        eps_l, sxx_l, syy_l, en_l = [0.0], [s0[0] * Lz], [s0[1] * Lz], [e0]
        for eps in eps_points:
            atoms = base.copy()                 # start from relaxed eps=0 ions
            emat = np.zeros((3, 3))
            emat[axis, axis] = eps
            atoms.set_cell(cell0 @ (np.eye(3) + emat), scale_atoms=True)
            atoms.calc = calc
            conv, fres = relax_ions(atoms, fmax)
            e = float(atoms.get_potential_energy())
            s = atoms.get_stress(voigt=True)
            print(f"    eps_{name}={eps:+.4f}: sxx*Lz={s[0] * Lz:+.5f}  "
                  f"syy*Lz={s[1] * Lz:+.5f} eV/A^2  E={e:.6f} eV"
                  f"{'' if conv else '  NOT CONVERGED'}", flush=True)
            table.append(_row(name, eps, s, e, fres, conv))
            eps_l.append(eps)
            sxx_l.append(s[0] * Lz)
            syy_l.append(s[1] * Lz)
            en_l.append(e)
        series[name] = tuple(np.asarray(v) for v in (eps_l, sxx_l, syy_l, en_l))

    eps, sxx, syy, en = series["xx"]
    fits = {"C11": linfit(eps, sxx), "C12": linfit(eps, syy)}
    e_xx = quadfit(eps, en)
    eps, sxx, syy, en = series["yy"]
    fits["C22"] = linfit(eps, syy)
    fits["C21"] = linfit(eps, sxx)
    e_yy = quadfit(eps, en)

    consts = {k: f["slope"] * EV_PER_A2_TO_N_PER_M for k, f in fits.items()}
    return {
        "ok": True,
        "model_resolved": resolved,
        "natoms": len(base),
        "Lz_A": Lz,
        "area_A2": A0,
        "C11_Nm": consts["C11"],
        "C22_Nm": consts["C22"],
        "C12_Nm": consts["C12"],
        "C21_Nm": consts["C21"],
        "anisotropy_C22_over_C11": consts["C22"] / consts["C11"],
        "fits": {k: {"slope_eV_A2": f["slope"], "intercept_eV_A2": f["intercept"],
                     "R2": f["R2"]} for k, f in fits.items()},
        "energy_crosscheck": {
            "C11_Nm": 2.0 * e_xx["c2"] / A0 * EV_PER_A2_TO_N_PER_M,
            "R2_xx": e_xx["R2"],
            "C22_Nm": 2.0 * e_yy["c2"] / A0 * EV_PER_A2_TO_N_PER_M,
            "R2_yy": e_yy["R2"],
        },
        "residual_stress_eps0_voigt_eV_A3": s0.tolist(),
        "eps0_converged": conv0,
        "stress_strain_table": table,
    }


def _print_summary(out):
    print("\n== elastic summary (N/m, relaxed-ion 2D constants) ==")
    print(f"{'cell':<16}{'C11':>8}{'C22':>8}{'C12':>8}{'C21':>8}"
          f"{'C22/C11':>9}{'R2min':>8}  E-xcheck C11/C22")
    for key, c in out["cells"].items():
        if not c.get("ok"):
            print(f"{key:<16}FAILED  {c.get('error', '')[:70]}")
            continue
        r2min = min(f["R2"] for f in c["fits"].values())
        ec = c["energy_crosscheck"]
        print(f"{key:<16}{c['C11_Nm']:>8.2f}{c['C22_Nm']:>8.2f}"
              f"{c['C12_Nm']:>8.2f}{c['C21_Nm']:>8.2f}"
              f"{c['anisotropy_C22_over_C11']:>9.2f}{r2min:>8.4f}"
              f"  {ec['C11_Nm']:.2f}/{ec['C22_Nm']:.2f}")
    lit = LITERATURE
    print(f"{'lit (PBE)':<16}{lit['C11_Nm']:>8.1f}{lit['C22_Nm']:>8.1f}"
          f"{lit['C12_Nm']:>8.1f}{'-':>8}"
          f"{lit['C22_Nm'] / lit['C11_Nm']:>9.2f}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage D2 - in-plane elastic constants per cell config")
    p.add_argument("--cells", default="e3nn/float64,e3nn/float32,cueq/float32")
    p.add_argument("--model", default=default_model())
    p.add_argument("--strains", default="0.0025,0.005,0.0075,0.01",
                   help="comma list; each is applied +/- along xx and yy")
    p.add_argument("--fmax", type=float, default=2e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true",
                   help="single strain 0.005 - <2 min sanity run")
    args = p.parse_args()

    strains = [0.005] if args.smoke else [float(s) for s in args.strains.split(",")]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    out = {"model": args.model, "device": args.device, "fmax": args.fmax,
           "strains": strains, "smoke": args.smoke, "supercell": [2, 3, 1],
           "lit_reference_PBE": LITERATURE, "cells": {},
           "env": env_metadata()}

    t0 = time.time()
    for key in cells:
        print(f"--- cell {key} (model={args.model})", flush=True)
        t_cell = time.time()
        try:
            parts = key.split("/")
            if len(parts) != 2:
                raise ValueError(f"cell must be 'backend/dtype', got {key!r}")
            res = measure_cell(*parts, args.model, args.device, strains, args.fmax)
            res["wall_s"] = time.time() - t_cell
            out["cells"][key] = res
        except Exception as exc:
            # e.g. cueq/float64 (broken upstream, MACE #1203/#1298) or a
            # backend without analytic stress - record, keep other cells alive
            out["cells"][key] = {"ok": False, "error": repr(exc),
                                 "traceback_tail": traceback.format_exc()[-1500:]}
            print(f"    FAILED: {exc!r}", flush=True)

    write_json(REPO / "results/raw/elastic.json", out)
    _print_summary(out)
    n_ok = sum(1 for c in out["cells"].values() if c.get("ok"))
    print(f"[phosbench] elastic done in {time.time() - t0:.0f}s "
          f"({n_ok}/{len(cells)} cells ok)")
    return 0 if n_ok == len(cells) else (1 if n_ok else 2)


if __name__ == "__main__":
    raise SystemExit(main())
