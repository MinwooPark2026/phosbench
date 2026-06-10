#!/usr/bin/env python
"""Build and relax the canonical phosphorene monolayer used by every other script.

The reference numerical path is fixed here once: e3nn backend, float64. The model
is selectable because Stage A (scripts/02_gate_physics.py) picks the foundation
model by physics gates and re-runs this with the winner. Cell (in-plane only) and
positions are relaxed to fmax, written to structures/phosphorene_relaxed.extxyz,
and the relaxed lattice constants are printed against literature DFT as a first
sanity gate.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phosbench.common import (CANONICAL_XYZ, env_metadata, make_calc,
                              phosphorene_unit_cell, write_json)

LITERATURE = {
    # monolayer phosphorene, common PBE values (see docs/case-study.md for refs)
    "a_armchair_A": 4.62,
    "b_zigzag_A": 3.30,
}


def build_and_relax(model="medium", device="cuda", fmax=1e-3,
                    backend="e3nn", dtype="float64", logfile="-"):
    """Relax positions + in-plane cell of the 4-atom monolayer; returns Atoms."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = phosphorene_unit_cell()
    atoms.calc = make_calc(backend, dtype, model=model, device=device)
    ecf = FrechetCellFilter(atoms, mask=[True, True, False, False, False, False])
    FIRE(ecf, logfile=logfile).run(fmax=fmax, steps=500)
    return atoms


def coordination_ok(atoms) -> bool:
    """Every P must stay 3-fold coordinated (~2.2-2.3 A bonds) after relaxation."""
    from ase.neighborlist import neighbor_list

    i = neighbor_list("i", atoms, cutoff=2.6)
    return bool(np.all(np.bincount(i, minlength=len(atoms)) == 3))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="medium")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fmax", type=float, default=1e-3)
    p.add_argument("--out", default=str(CANONICAL_XYZ))
    args = p.parse_args()

    from ase.io import write as ase_write

    atoms = build_and_relax(model=args.model, device=args.device, fmax=args.fmax)
    a, b = atoms.cell.lengths()[:2]
    e0 = atoms.get_potential_energy()
    print(f"relaxed: a(armchair)={a:.4f} A  b(zigzag)={b:.4f} A  E={e0:.6f} eV")
    print(f"literature (PBE): a={LITERATURE['a_armchair_A']}  b={LITERATURE['b_zigzag_A']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atoms_save = atoms.copy()  # detach calculator before writing
    ase_write(out, atoms_save)
    write_json(out.with_suffix(".json"), {
        "a_armchair_A": a,
        "b_zigzag_A": b,
        "energy_eV": e0,
        "natoms": len(atoms),
        "fmax_target": args.fmax,
        "relaxation": {"model": args.model, "backend": "e3nn", "dtype": "float64"},
        "literature": LITERATURE,
        "env": env_metadata(),
    })
    print(f"[phosbench] canonical structure -> {out}")

    if not coordination_ok(atoms):
        print("WARNING: coordination changed during relaxation (want all 3-fold)")
        return 1
    print("coordination check: all atoms 3-fold - OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
