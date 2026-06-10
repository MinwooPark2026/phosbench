# phosbench

**Production MLIP molecular dynamics on the GPUs university labs already own —
an end-to-end deployment study on black phosphorus (RTX 3080 Ti, 12 GB).**

Machine-learned interatomic potentials (MACE foundation models) promise
DFT-quality MD at classical cost — but the published acceleration numbers
(NVIDIA's cuEquivariance-in-LAMMPS results, Oct 2025; arXiv:2510.23621) are
water benchmarks on datacenter A100/H100s, throughput-first. The hardware a
typical university lab actually owns is a workstation RTX card, where fp64
runs at 1/64 of fp32 throughput and VRAM is 12 GB. This study measures, on
monolayer phosphorene (a maximally anisotropic 2D semiconductor), what it
takes to run **trustworthy production MD** on that hardware: when
cuEquivariance pays off and when it does not, what fp32 does to *physical
observables* (phonon dispersions, anisotropic elastic constants, energy
drift — not force RMSE), and where the time actually goes (Nsight). Every
figure caption states the deployment decision it informs.

> **Scope discipline**: this extends the published datacenter throughput
> results to the consumer break-even boundaries and solid-state observables
> that drive real lab deployment decisions. It is *not* the first consumer
> MACE benchmark (arXiv:2510.23621 includes an RTX 2080 Ti) — it is, to our
> knowledge, the first break-even/OOM characterization and the first
> precision-vs-solid-state-observable study for the MACE + cuEquivariance
> stack.

## TL;DR — the recommendation matrix

For a lab running MACE-class potentials on a 12 GB RTX workstation:

| system size | backend & precision | why (measured) |
|---|---|---|
| < ~350–500 atoms | e3nn / fp32 (cuEq off) | step time is host-bound (GPU kernels ≤ 8% of step); cuEq kernels make it *slower* (×0.74–0.96) |
| ~500 – 3,000 atoms | **cuEq / fp32** | ×1.4 (small model) to ×5.0 (large model) per-step speedup at 2,944 atoms |
| 3,000 – 23,000 atoms | **cuEq / fp32** (only option) | e3nn OOMs at 2,944–5,760 atoms; cuEq's ~5× smaller activation memory reaches 11,520 (medium) – 23,040 (small) atoms |
| any size, fp64 | don't — use e3nn/fp64 *sparingly* for reference data | fp64 costs ×3.2 (64 at.) to ×10 (1,408 at.) on GA102 and OOMs at 1,408 atoms |
| property workflows (phonons, elastic) | **hybrid**: displaced-force evaluations on e3nn/fp64, production MD on cuEq/fp32 | fp32 finite-difference noise pollutes small displacements; at the standard 0.01 Å amplitude it creates spurious imaginary acoustic modes (tiny, ~0.01 cm⁻¹ RMSE — but they flip stability verdicts) |
| CPU (8-core Ryzen 5800X) | never for this workload | GPU wins ×13 even at 64 atoms, ×64 at 1,408 |

**The error budget that matters**: across every observable we measured,
|fp32 − fp64| and |cuEq − e3nn| are **≤ 0.2 %** of |model − DFT-literature|.
Numerical precision is not the error you should pay to reduce — model choice
is (see *Zero-shot validation failure*, below).

## Headline figures

| | |
|---|---|
| ![break-even](results/figures/speedup_breakeven.png) | ![vram](results/figures/vram_oom.png) |
| ![time share](results/figures/time_share.png) | ![phonons](results/figures/phonon_dispersion.png) |

## Key findings

1. **Break-even is real and model-size-dependent.** cuEquivariance kernel
   crossover sits at ~313 atoms (MACE-MP-0 large), ~373–454 (medium), ~947–982
   (small); wall-clock crossover lands slightly later than the bare force-call
   crossover — ASE host overhead delays it. Below break-even cuEq *costs*
   up to 26 % (and 60 % for OMAT-medium at 64 atoms).
2. **cuEquivariance's bigger gift on 12 GB is memory, not speed.** At 2,944
   atoms (medium model) cuEq peaks at ~1.5 GiB where e3nn needs 7.5 GiB; the
   reachable system size grows ×4 (2,944 → 11,520 atoms). The OOM boundary
   per config is tabulated in `results/figures/oom_boundary.csv`.
3. **After acceleration, the bottleneck is the host.** Nsight: at 140 atoms the
   GPU is busy 1 % of the step (cuEq) — faster kernels cannot help; at 2,944
   atoms e3nn keeps the GPU 60 % busy vs cuEq's 10 %. cuEq shifts the limiter
   from kernels to the Python/ASE loop: the production path beyond this study
   is LAMMPS ML-IAP (Kokkos) or CUDA-graph-style batching, per NVIDIA's
   datacenter results.
4. **fp64 is effectively unavailable on consumer Ampere.** ×3.2–×10 measured
   cost (the fp64 GEMMs dominate the timeline: `cutlass...d884gemm` kernels),
   ×2 memory, OOM at 1,408 atoms. Consumer deployment *forces* the fp32
   question — which is exactly why the accuracy gates below matter. (On
   A100/H100 fp64 is 1:2, so the datacenter column of the matrix differs.)
5. **fp32/cuEq preserve the physics — measured, not assumed.**
   - Parity gate (cuEq vs e3nn @ fp32, 140 atoms): ΔE = 0.000 meV/atom,
     max |ΔF| = 5×10⁻⁴ meV/Å.
   - Phonon dispersion: per-branch RMSE ≤ 0.08 cm⁻¹ (fp32 vs fp64), invisible
     against the ~10–20 cm⁻¹ model-vs-experiment scale.
   - Anisotropic elastic constants: C11/C22/C12 = 33/105/30 N/m identical to
     0.2 % across e3nn-fp64 / e3nn-fp32 / cuEq-fp32.
   - NVE drift & NPT lattice: [overnight run — results land here]
6. **…but fp32 changes *how you must measure*.** Finite-difference phonons at
   the standard 0.01 Å displacement pick up fp32 force noise: spurious
   (≈ −0.01 THz) imaginary acoustic artifacts that vanish at 0.05 Å — while
   fp64 prefers small displacements (0.05 Å adds anharmonic contamination).
   Hence the hybrid policy in the matrix.
7. **Zero-shot validation failure caught before deployment.** All three
   foundation models tested (MACE-MP-0, MACE-MPA-0, MACE-OMAT-0) reproduce the
   zigzag lattice constant within ~2 % but compress the soft armchair axis by
   **7–10 %** (4.30 Å vs 4.62 Å DFT) — exactly the direction whose stiffness
   is 3–4× lower (C11 ≈ 24 N/m vs C22 ≈ 103 N/m in DFT). Anisotropy survives
   zero-shot (C22/C11 = 3.2 vs ~4.3 DFT) but absolute armchair elasticity is
   ~35 % high against literature. A potential you have not validated on *your*
   material's soft direction is not production-ready; fine-tuning on the
   open GAP-20 phosphorus dataset is the documented next step.
8. **Found upstream: MACE analytic stress is ×17.8 off on this slab.**
   Hellmann-Feynman check (`scripts/90_diag_stress_hf.py`): analytic
   `get_stress()` is 17.8× smaller than dE/dε of the same energy surface
   (e3nn/fp64/CPU — independent of cuEquivariance). Stress-slope elastic
   constants are unusable on pbc=[T,T,F] geometries with mace-torch 0.3.16;
   this repo's constants use energy-curvature fits (R² > 0.9997), and the NPT
   barostat note in `scripts/22_md_stability.py` documents the consequence.

## Known-issues table (what we worked around, honestly)

| issue | consequence here |
|---|---|
| cuEq + fp64 unsupported/broken upstream (MACE #1203, #1298) | matrix is 3 cells (e3nn/fp64 reference, e3nn/fp32, cuEq/fp32), not 2×2; cuEq/fp64 probe recorded |
| SM86 is not a tuned cuEq target (kernels are SM80/90/100+) | speedups here are a *lower bound*; verified real cuEq kernels run via Nsight (`segmented_polynomial_*`) — no silent fallback |
| torch.compile × cuEq zero-gradient bug (cuEq #77) | torch.compile disabled everywhere |
| MACE slab stress ×17.8 (this work, `90_diag_stress_hf.py`) | elastic constants via energy curvature; NPT relaxes ~18× slower than nominal |
| consumer boost clocks drift | no root to lock clocks → SM clock/temp/power logged per measurement, medians of per-step laps reported |
| ncu hardware counters need `NVreg_RestrictProfilingToAdminUsers=0` | kernel analysis via nsys timelines only (sufficient for time-share) |

## Reproduce

```bash
# one-time: structure + gates (fails loudly if your stack is broken)
bash scripts/stage_a.sh
# sweeps / accuracy arms / profiling (hours; queue-friendly, resumable)
python scripts/10_sweep_throughput.py --backends e3nn,cueq --dtypes float32 \
    --models small,medium,large,medium-omat-0 --modes md,force_call
python scripts/20_phonons.py --displacements 0.01,0.03,0.05
python scripts/21_elastic.py && python scripts/23_elastic_recompute.py
python scripts/22_md_stability.py
bash scripts/30_profile_nsys.sh
python scripts/40_make_plots.py
```

Pinned stack (verified): mace-torch 0.3.16 · cuequivariance(-torch/-ops) 0.10.0
· torch 2.11.0+cu128 · e3nn 0.4.4 · ase 3.28.0 · phonopy 4.1.0 · driver 610.43
(CUDA 13.3) · Ubuntu 24.04 · RTX 3080 Ti 12 GB · Ryzen 7 5800X.

Protocol design, gate thresholds and the schedule that produced this in
~2 days: [PROTOCOL.md](PROTOCOL.md). The 2-page consulting-style summary for
a lab deciding *today*: [docs/engagement-memo.md](docs/engagement-memo.md).

## Where this goes next

- **Datacenter column**: the sweep harness is config-driven and re-runs
  unmodified on A100/H100 (fp64 at 1:2 changes the precision economics;
  SM80-tuned kernels should raise the cuEq column). Published water numbers
  suggest ×3–5 at scale — consistent with what we measure at 2,944+ atoms.
- **Scaling out**: LAMMPS ML-IAP-Kokkos is the supported multi-GPU path
  (cuEq-converted models are not LAMMPS-exportable today — another deployment
  fact a lab needs to know before building a workflow on ASE).
- **Model fix**: time-boxed fine-tune on GAP-20 (Zenodo 10.5281/zenodo.4003703)
  to repair the armchair axis, then re-run *only* the accuracy arms (the
  benchmark numbers are model-fidelity-independent by construction).
