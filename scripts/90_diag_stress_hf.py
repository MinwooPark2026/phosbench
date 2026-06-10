#!/usr/bin/env python
"""Minimal reproduction: MACE analytic stress vs Hellmann-Feynman dE/deps on a slab.

On the phosphorene monolayer (pbc=[True, True, False], ~20 A vacuum) MACE 0.3.16
returns an analytic virial stress ~17.8x smaller than the finite-difference
strain derivative of its own energy, independent of backend/precision/device
(reproduced with e3nn/float64 on CPU - no cuEquivariance involved):

    sigma_yy finite-diff : 3.217e-03 eV/A^3
    sigma_yy analytic    : 1.807e-04 eV/A^3   (ratio 17.81)

Consequence for this study: stress-slope elastic constants are unusable on this
geometry; energy-curvature fits (scripts/21_elastic.py crosscheck path) are the
primary numbers, and the NPT barostat relaxes ~18x slower than its nominal taup
(fixed point unchanged: zero reported stress is zero true stress).

Run: python scripts/90_diag_stress_hf.py [--model medium-omat-0] [--device cpu]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phosbench.common import load_canonical, make_calc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="medium-omat-0")
    p.add_argument("--device", default="cpu")
    p.add_argument("--eps", type=float, default=0.010)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--control", action="store_true",
                   help="fully-periodic control (bulk Si): expect ratio ~1.0, "
                        "isolating the slab geometry as the trigger")
    args = p.parse_args()

    calc = make_calc("e3nn", "float64", model=args.model, device=args.device)
    if args.control:
        from ase.build import bulk

        base = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
        print("CONTROL: fully periodic bulk Si, pbc =", base.pbc.tolist())
    else:
        base = load_canonical().repeat((2, 3, 1))
    cell0 = base.get_cell().array.copy()
    V = abs(np.linalg.det(cell0))
    print(f"natoms={len(base)}  V={V:.2f} A^3  Lz={cell0[2, 2]:.3f} A")

    def at_eps(e):
        a = base.copy()
        M = np.eye(3)
        M[1, 1] += e
        a.set_cell(cell0 @ M, scale_atoms=True)
        a.calc = calc
        return a

    energies, sigma = {}, None
    for e in (args.eps - args.delta, args.eps, args.eps + args.delta):
        a = at_eps(e)
        energies[e] = a.get_potential_energy()
        if abs(e - args.eps) < 1e-12:
            sigma = a.get_stress(voigt=True)

    dEde = (energies[args.eps + args.delta]
            - energies[args.eps - args.delta]) / (2 * args.delta)
    fd = dEde / V
    print(f"sigma_yy finite-diff : {fd: .6e} eV/A^3")
    print(f"sigma_yy analytic    : {sigma[1]: .6e} eV/A^3")
    ratio = fd / sigma[1] if sigma[1] else float("inf")
    print(f"ratio FD/analytic    : {ratio:.4f}")
    consistent = abs(ratio - 1.0) < 0.05
    print(f"HELLMANN_FEYNMAN: {'CONSISTENT' if consistent else 'INCONSISTENT'}")
    return 0 if consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
