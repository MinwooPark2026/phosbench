**Title**: Analytic stress on partially periodic (slab) systems is ~18x smaller
than the Hellmann-Feynman strain derivative of the energy

## Summary

For a 2D slab (`pbc=[True, True, False]`, vacuum along z) the ASE calculator's
analytic `get_stress()` is inconsistent with the finite-difference strain
derivative of the calculator's own potential energy: on a phosphorene
monolayer we measure

```
sigma_yy (finite difference dE/d_eps / V) :  3.217e-03 eV/A^3
sigma_yy (analytic get_stress)            :  1.807e-04 eV/A^3
ratio FD / analytic                       :  17.81
```

A fully periodic control with the **same calculator and the same script**
(bulk Si, `pbc=[True, True, True]`) is consistent to 4 decimal places
(ratio 1.0000). That isolates the inconsistency to the partially periodic
geometry rather than the model or the FD setup.

Because the energy surface itself is fine, every energy-derived quantity is
unaffected, but everything that consumes the analytic virial silently
breaks on slabs (details below).

## Environment

- mace-torch 0.3.16, e3nn 0.4.4, ase 3.28.0, numpy 2.4.6
- torch 2.11.0+cu128, though the repro below runs on CPU with plain e3nn
  at float64 (no cuEquivariance involved)
- model: `mace_mp(model="medium-omat-0")`; we expect the effect to be
  model-independent (it sits at the calculator level) and can re-test other
  models on request
- Linux x86_64 (Ubuntu 24.04)

## Minimal reproduction (self-contained, CPU, ~2 min)

```python
import numpy as np
from ase import Atoms
from mace.calculators import mace_mp

calc = mace_mp(model="medium-omat-0", device="cpu", default_dtype="float64",
               enable_cueq=False, dispersion=False)

def phosphorene(vacuum=20.0):
    a, b, dz = 4.62, 3.30, 2.14   # approximate monolayer black phosphorene
    z0 = vacuum / 2
    return Atoms("P4",
                 positions=[(0.000 * a, 0.0, z0), (0.367 * a, b / 2, z0),
                            (0.500 * a, b / 2, z0 + dz), (0.867 * a, 0.0, z0 + dz)],
                 cell=[a, b, vacuum + dz], pbc=[True, True, False])

base = phosphorene().repeat((2, 3, 1))
cell0 = base.get_cell().array.copy()
V = abs(np.linalg.det(cell0))

EPS, DELTA = 0.010, 0.001
def at_eps(e):                      # affine strain along yy, clamped ions
    a = base.copy()
    M = np.eye(3); M[1, 1] += e
    a.set_cell(cell0 @ M, scale_atoms=True)
    a.calc = calc
    return a

energies = {e: at_eps(e).get_potential_energy()
            for e in (EPS - DELTA, EPS, EPS + DELTA)}
sigma = at_eps(EPS).get_stress(voigt=True)

fd = (energies[EPS + DELTA] - energies[EPS - DELTA]) / (2 * DELTA) / V
print(f"sigma_yy finite-diff : {fd: .6e} eV/A^3")
print(f"sigma_yy analytic    : {sigma[1]: .6e} eV/A^3")
print(f"ratio FD/analytic    : {fd / sigma[1]:.4f}")
# observed: ratio ~ 17.8  (expected ~ 1.0)
# control: replace the slab with ase.build.bulk("Si","diamond",a=5.43).repeat((2,2,2))
#          -> ratio 1.0000 with the same calculator
```

At any configuration, sigma_ij = (1/V) dE/d_eps_ij under affine scaling of the
positions, so the two numbers must agree regardless of model quality; on the
slab they disagree by ~18x.

## Additional observations

- The ratio is approximately, but maybe not exactly, constant: independent
  strain-direction fits on the same slab give 17.8-17.9 (xx vs yy), which may
  point to a missing/partial virial contribution rather than a pure
  volume-normalization slip. We did not attempt to localize the mechanism.
- Downstream effects we measured before finding the root cause:
  - stress-slope elastic constants come out ~18x too small, while
    energy-curvature fits on the same relaxed configurations give values in
    the expected range (C11 ~ 33, C22 ~ 105 N/m for the monolayer model);
  - `Inhomogeneous_NPTBerendsen` (z-masked) runs away: the barostat balances
    the *unscaled* kinetic pressure against the ~18x-undersized virial, so the
    in-plane cell inflates monotonically (~+20 % over 50 ps at 300 K),
    identically for e3nn/float64, e3nn/float32 and cuEq/float32, so the
    failure is independent of precision and backend.
- Possibly related (but, as far as we can tell, distinct) reports: #980
  (per-atom stress), #294/#222/#395 (training-data virial formats).

Happy to provide the full measurement data (the repro above is extracted from
a benchmarking study of MACE+cuEquivariance on consumer GPUs) or to test
candidate fixes.
