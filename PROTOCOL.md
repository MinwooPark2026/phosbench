# phosbench — Measurement Protocol (frozen 2026-06-10)

**Production MLIP MD on consumer GPUs — a deployment case
study on black phosphorus.**

Single material (monolayer phosphorene + bulk-BP Γ modes as context), full depth.
Every figure answers a deployment decision, not a curiosity. Target hardware is a
workstation RTX 3080 Ti (GA102, SM86, 12 GB) — deliberately: this is the consumer
GPU most researchers actually run on. Published A100/H100 numbers are imported as the
"datacenter column" with explicit caveats.

## Prior art and claims discipline

- NVIDIA blog (Oct 2025): cuEq-MACE in LAMMPS, A100/H100, water, throughput only.
- arXiv 2510.23621: e3nn-vs-cuEq precision arms, water thermodynamics, incl. RTX 2080 Ti.
- **Unclaimed cells we occupy**: (1) solid-state physical observables (phonon
  dispersion, anisotropic elastic constants) under precision/backend change;
  (2) break-even + OOM boundary maps on consumer SM86; (3) Nsight kernel-level
  decomposition of where time goes. Never claim "first consumer GPU MACE benchmark".

## The 3-cell matrix (NOT 2×2 — by design)

| cell | role |
|---|---|
| e3nn / float64 | accuracy reference (never a speed denominator) |
| e3nn / float32 | precision cost, measured |
| cueq / float32 | production candidate; backend speedup = vs e3nn/float32 ONLY |

cueq/float64 is documented broken upstream (MACE #1203, #1298; cuEq changelog
removed fp32-math/fp64-IO). Stage A probes it expecting failure or silent
fallback; the outcome is itself a deployment finding ("the supported production
path is cueq-fp32"). GA102 runs fp64 at 1:64 of fp32 — consumer deployment
*forces* the fp32 question; that asymmetry (A100 = 1:2) is the story's engine.

Known-issues guardrails: single-head models only (#1298), no torch.compile
(cuEq #77), pinned stack: mace-torch 0.3.16 / cuequivariance 0.10.0 +
ops-torch-cu12 / e3nn (env) / torch 2.11.0+cu128 (verified working on this box).

## Stage A — gates (hour 1 on GPU; all must pass before sweeps)

1. **Kernel-truth check**: 10-step nsys trace with `enable_cueq=True` →
   cuEquivariance kernel names visibly on the GPU timeline (rules out silent
   e3nn fallback on SM86; SM86 is not a tuned cuEq target — disclose in captions).
2. **Parity gate (hard stop)**: cueq/fp32 vs e3nn/fp32, MACE medium, 140-atom
   phosphorene supercell: |ΔE| < 1 meV/atom AND max|ΔF| < 1 meV/Å.
3. **fp64 path**: e3nn/fp64 runs end-to-end (verify parameter dtypes really are
   float64); cueq/fp64 probe → record failure mode or fallback.
4. **Physics gate (model selection)**: for `medium-mpa-0` and `medium-omat-0`
   (zero-shot, e3nn/fp64): relax monolayer → puckered geometry preserved,
   a, b within ~3% of DFT (a≈4.62, b≈3.30 Å); quick 4×6×1 phonopy check →
   quadratic ZA branch, no significant imaginary modes (tolerance: |ω_imag| <
   0.3 THz near Γ, standard for 2D flexural numerics). Winner becomes THE model;
   plain MP-0 is recorded for context only. If both fail → pivot: the deliverable
   becomes "zero-shot validation failure caught before deployment" + (stretch,
   time-boxed) fine-tune on GAP-20 dataset (Zenodo 10.5281/zenodo.4003703).
5. **Profiling gate**: nsys works rootless (CUDA + NVTX + osrt traces); note
   whether ncu hardware counters need NVreg flag (don't block on it).
6. **VRAM bisect**: max atoms before OOM, medium model, e3nn vs cueq @ fp32
   (log-scale bisect) — sizes the sweep ladder; cueq's memory saving extends
   the boundary (report as data).

## Stage B — throughput sweep (≤ ~40 runs)

- Axes: {e3nn, cueq} × fp32 × {small, medium, large or MPA/OMAT medium} ×
  size ladder (~64 → OOM); PLUS one e3nn/fp64 medium column (reference cost);
  PLUS CPU (Ryzen 5800X, e3nn/fp32, sizes ≤ ~2k atoms) for the GPU-vs-CPU
  break-even.
- Two timing modes per config (this separation is a headline deliverable):
  - `force_call`: bare calculator on rattled copies — kernel-level speedup;
  - `md`: real VelocityVerlet loop — end-to-end ns/day with ASE host overhead.
  "Kernel crossover" ≠ "wall-clock crossover"; publish both maps.
- Methodology: subprocess isolation per config (OOM cannot poison later runs),
  warmup ≥ 10 steps (JIT/autotune amortized; first-call latency recorded
  separately), explicit cuda.synchronize, median + p10/p90 of per-step laps,
  peak VRAM, **GPU clock + temperature logged per data point** (consumer boost
  variance; clock locking needs sudo → we log instead, methodology disclosed).

## Stage C — Nsight (nsys-first)

- nsys on representative configs: {small ~140, mid ~2k, large ~10k atoms} ×
  {e3nn, cueq} @ fp32 + medium e3nn/fp64: kernel time breakdown (top-10 kernels),
  NVTX `force_eval` vs host time share, H2D/D2H transfer share, achieved occupancy
  proxy from timeline. NVTX ranges instrumented in phosbench.common.
- ncu deep-dive on the single dominant kernel ONLY if perf-counter permission is
  painless; otherwise skip (documented).
- Output: stacked time-share bars vs system size — the "where did the time go"
  figure that explains WHY break-even sits where it sits.

## Stage D — physical accuracy (three-tier error budget)

Tier definitions per observable: T1 = |fp32 − fp64| (same model, e3nn);
T1' = |cueq-fp32 − e3nn-fp32| (backend); T2 = |model(fp64) − DFT literature|;
T3 = |DFT − experiment|. Claim format: "T1 is X% of T2" — precision is (or is
not) the error worth paying to reduce.

1. **Phonon dispersion** (phonopy, finite displacement, 4×6×1 supercell of the
   relaxed monolayer, path S–X–Γ–Y–S): per-branch RMSE between cells; ZA branch
   quadratic fit near Γ; **displacement-amplitude sweep 0.01/0.03/0.05 Å ×
   {fp64, fp32}** — fp32 finite-difference noise vs amplitude is a core finding
   (expected outcome: hybrid policy — fp64/e3nn for displaced-force property
   evaluations, cueq-fp32 for production MD). Bulk-BP Γ Raman modes
   (exp: Ag¹ 362 / B₂g 439 / Ag² 467 cm⁻¹; bulk = mp-157) as T2/T3 context.
2. **Anisotropic elastic constants**: in-plane strain–stress fits (±0.25/0.5/
   0.75/1.0% on εxx, εyy; ion positions relaxed at each strain; 2D constants =
   slope × Lz, N/m). Targets: C11(armchair) ≈ 24 N/m vs C22(zigzag) ≈ 103 N/m,
   anisotropy ≈ 4 (DFT lit.). Report C11, C22 (+C12 if clean) per cell.
3. **NVE drift**: ~500-atom supercell, 1 fs, 300 K; ≥25 ps per cell (extend
   fp32 arms to 100 ps if queue time allows); report drift slope μeV/atom/ps.
4. **NPT in-plane lattice** (monolayer only — bulk interlayer is vdW/D3-bound,
   demoted to a model-limitation note even though torch-dftd is available):
   anisotropic in-plane barostat (Inhomogeneous_NPTBerendsen, z masked), 300 K,
   ≥50 ps: ⟨a⟩, ⟨b⟩ per cell vs fp64 reference.

## Stage E — deliverables

1. `README.md` — case study with decision-captioned figures (each caption states
   the deployment decision it informs, e.g. "below ~N atoms stay on e3nn —
   host overhead dominates").
2. `docs/engagement-memo.md` — 1-2 page mock SA engagement memo addressed to "a
   university lab with RTX workstations": workload characterization table (data
   types, IO, CPU↔GPU split), bottleneck quantification, recommendation matrix
   (model × precision × backend × system size) incl. datacenter column from
   published numbers. Highest JD-signal artifact.
3. Figures: break-even heatmap + decision flowchart (GTC-style summary), 3-tier
   error budget bars, time-share stacks, OOM boundary curve, phonon overlays,
   displacement-noise curves, drift traces.
4. Honesty markers: known-issues table (MACE #1203/#1298, cuEq #77, SM86 status),
   parity gates printed before every speedup table, clock/thermal methodology,
   half-page "why plane-wave DFT stays on the cluster" appendix (no new QE runs).

## Schedule (compressed; jobq serializes GPU work)

- **Tonight (D0)**: Stage A gates → launch Stage B sweep on queue.
- **D1**: Stage D fp64 reference physics + precision arms; Stage C traces between.
- **D2**: analysis, figures, README + memo, repo polish, (stretch) 5-min GTC-style
  slide/video.
