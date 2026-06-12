# MACE 업스트림 이슈 초안 — 제출 전 확인용

> **제출 위치**: https://github.com/ACEsuit/mace/issues/new
> **제출 전 체크리스트**
> 1. ~~중복 검색~~ — 2026-06-13 검색 완료: 동일 이슈 없음. 인접 이슈들은 학습 데이터의
>    virial 포맷(#294, #222, #395)이나 per-atom stress(#980), cuEq+fp64(#1203, #1298)로
>    전부 별개.
> 2. (선택, 2분) workstation에서 `python scripts/90_diag_stress_hf.py --model medium`으로
>    MP-0 medium에서도 재현되는지 확인 → 재현되면 아래 본문의 "expected to be
>    model-independent"를 "reproduced with MACE-MP-0 medium as well"로 교체.
> 3. (선택) 제출 시점의 최신 mace-torch 버전에서 한 번 더 돌려 버전 갱신.
> 4. 제출 후: 이슈 번호를 README.md의 "Upstream issue ... in preparation" 문장에 반영.
>
> 아래 가로줄부터 끝까지가 그대로 붙여넣을 영문 본문입니다.

---

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
(ratio 1.0000), isolating the inconsistency to the partially periodic
geometry rather than the model or the FD setup.

Because the energy surface itself is fine, every energy-derived quantity is
unaffected — but everything that consumes the analytic virial silently
breaks on slabs (details below).

## Environment

- mace-torch 0.3.16, e3nn 0.4.4, ase 3.28.0, numpy 2.4.6
- torch 2.11.0+cu128 — but the repro below runs on **CPU**, plain e3nn,
  float64 (no cuEquivariance involved)
- model: `mace_mp(model="medium-omat-0")`; the effect is expected to be
  model-independent (calculator-level), happy to re-test others on request
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
  volume-normalization slip — we did not attempt to localize the mechanism.
- Downstream effects we measured before finding the root cause:
  - stress-slope elastic constants come out ~18x too small, while
    energy-curvature fits on the same relaxed configurations give values in
    the expected range (C11 ~ 33, C22 ~ 105 N/m for the monolayer model);
  - `Inhomogeneous_NPTBerendsen` (z-masked) runs away: the barostat balances
    the *unscaled* kinetic pressure against the ~18x-undersized virial, so the
    in-plane cell inflates monotonically (~+20 % over 50 ps at 300 K),
    identically for e3nn/float64, e3nn/float32 and cuEq/float32 — i.e. the
    failure is precision/backend-independent.
- Possibly related (but, as far as we can tell, distinct) reports: #980
  (per-atom stress), #294/#222/#395 (training-data virial formats).

Happy to provide the full measurement data (the repro above is extracted from
a benchmarking study of MACE+cuEquivariance on consumer GPUs) or to test
candidate fixes.
