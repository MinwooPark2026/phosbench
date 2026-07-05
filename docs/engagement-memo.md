# Engagement memo — accelerating MLIP molecular dynamics on lab workstations

**To**: a university 2D-materials group running MACE-class potentials on RTX
workstations
**Re**: workload characterization, bottleneck analysis, and a deployment
recommendation for production MD on black-phosphorus systems
**Basis**: measured on RTX 3080 Ti 12 GB / Ryzen 7 5800X, mace-torch 0.3.16 +
cuEquivariance 0.10.0 (full data: this repo)

---

## 1. Workload characterization

| dimension | finding |
|---|---|
| kernel profile (e3nn baseline) | at 2,944 atoms GPU kernels occupy 60 % of each MD step's wall-time (kernel share = summed kernel durations / NVTX force_eval span on the nsys timeline — a wall-time ratio, *not* SM occupancy; hardware counters were permission-blocked); kernel time is spread across elementwise tensor ops and SGEMMs, with index-backward (gather/scatter) 6th at 6.5 % — on the cuEq build that same gather/scatter rises to the top (30.7 %) because the tensor-product kernels shrink |
| precision | model weights/activations fp32 by default; fp64 runs at 1/64 FLOP rate on GA102 → measured ×3.2–×10 step-time penalty and ×2 memory; fp64 is for reference data only on this hardware |
| CPU↔GPU traffic | negligible (< 0.25 % of step time): positions down / forces up per step; this workload is **not** transfer-bound on a workstation |
| host vs device | the real split: at 140 atoms, 92 % (e3nn/fp32) to 99 % (cuEq) of step time is host-side Python/ASE/launch overhead — the GPU idles; fp64 is the exception (50 % host, because its kernels are so slow) |
| MPI share | Measured, methodology demo (classical LJ, not MACE): strong-scaling a fixed 500k-atom LJ melt on 1→8 CPU cores, Comm rises 0.7 %→9.0 % of loop time while Pair falls 84.6 %→70.1 % (wall 135 s→25 s, 5.4×). This is how you locate the rank count where the comm tax overtakes the marginal pair-time win — the number to fix before any multi-node/GPU commitment. Same LAMMPS Pair/Comm breakdown applies to MACE via LAMMPS ML-IAP (pair_style mace). See results/figures/mpi_comm_share.png |
| memory scaling | activation memory grows steeply with neighbors; e3nn/medium OOMs at ~3k atoms on 12 GB |
| IO / data types | extxyz trajectories, float32 tensors; checkpoint-restart cost negligible vs step time |

## 2. Bottleneck analysis → what to change

1. **≥ ~500 atoms: kernels are the bottleneck → switch them.** Enabling
   cuEquivariance (`enable_cueq=True`, one line) delivers ×1.4–×5.0 at 2,944
   atoms (bigger models gain more) and cuts peak VRAM ~5×, raising your
   reachable system size from ~3k to ~11.5k atoms (medium model).
2. **Below break-even: the host is the bottleneck → cuEq won't help.** Below
   the measured crossover (~310–450 atoms for medium/large models, ~950–980
   for small) cuEq is *slower* (×0.74–0.96, down to ×0.40 for OMAT-medium at
   64 atoms). Keep e3nn, or batch many small systems.
3. **After cuEq, you are host-bound again** (GPU kernels occupy ≤ 10 % of step wall-time at 3k atoms).
   The next step is not a faster kernel — it is the integration layer:
   LAMMPS ML-IAP (Kokkos) for production runs, or batching/CUDA-graph
   approaches. Plan the workflow migration before scaling up.

## 3. Accuracy gates before you trust fp32 production MD

Run these once per material (scripts in this repo; minutes of GPU time):

- **Backend parity**: cuEq vs e3nn at fp32 — here ΔE below fp32
  representation resolution (bitwise-identical fp32 energies), max |ΔF| =
  5×10⁻⁴ meV/Å. Verify cuEq kernels actually run (Nsight;
  `segmented_polynomial_*`) — we caught a parser blind spot this way.
- **Observable-level precision check**: phonons, elastic constants, NVE drift
  vs an e3nn/fp64 reference. Here fp32 errors are ≤ 3 % (typically ≤ 1 %) of
  the model-vs-DFT error on every observable, and NVE drift is
  ≤ 0.012 µeV/atom/ps in every cell → fp32 production MD is safe *for this
  system*. One hard exclusion: do **not** run barostatted (NPT) MD on
  partially periodic systems with mace-torch 0.3.16 — the slab-stress bug
  (§5) inflates the cell ~20 % in 50 ps regardless of precision/backend.
- **Protocol caveat**: fp32 force noise corrupts finite-difference property
  workflows at small displacements (spurious imaginary acoustic points at
  0.01 Å that shrink ~4× or change sign at 0.05 Å; fp64 shows the opposite,
  anharmonic trend). Policy: displaced-force property evaluations on
  e3nn/fp64 (or CPU), production MD on cuEq/fp32, displacement amplitude
  chosen per precision.
- **Model validation is the real risk, not precision.** All current MACE
  foundation models compress phosphorene's soft armchair axis by 7–10 %.
  Zero-shot ≠ production-ready: validate the soft direction of *your*
  material. If it fails, budget ~one workstation-GPU-day for an
  energy-weighted fine-tune (measured training wall-time here: ~2 h) —
  demonstrated here on the open GAP-20 dataset,
  bringing every observable within ~5 % of DFT (and note the trap we hit
  first: a forces-weighted fine-tune leaves the lattice broken; the soft
  axis lives in the energy landscape).

## 4. Recommendation matrix

| your run | do this | measured throughput (1 fs steps, medium model) |
|---|---|---|
| screening, below break-even (~0.4k–1k at.) | e3nn/fp32 on the workstation GPU (never CPU: GPU wins ×13 even at 64 atoms) | 2.0 ns/day @ 64 at. |
| production MD, break-even–11k atoms | cuEq/fp32 on the workstation | 1.3 ns/day @ 1.4k at., 0.57 @ 2.9k, 0.25 @ 11.5k |
| > 11k atoms, multi-ns trajectories, or multi-GPU | datacenter / LAMMPS-Kokkos path; A100 fp64 at 1:2 also reopens fp64 if you need it | above ~5k atoms this card is for relaxations and snapshots, not multi-ns MD — that is the datacenter handoff point |
| phonons / elastic / stability verdicts | hybrid: e3nn/fp64 displaced forces, cuEq/fp32 dynamics | phonon set: minutes |
| geometry relaxation reference | e3nn/fp64, small cells only (last working rung 1.4k atoms; OOM at 2.9k) | — |

## 5. Effort & risk

- Adoption cost of cuEq on an existing MACE/ASE workflow: **one flag**, plus
  the parity gate above. Known sharp edges we hit and documented: cuEq+fp64
  unsupported (MACE #1203/#1298), torch.compile incompatibility (cuEq #77),
  MACE slab-stress inconsistency (×17.8, found here — use energy-curvature
  fits for elastic constants until fixed upstream).
- Everything in this memo is reproducible from the repo in about one GPU-day
  (the long pole is the MD stability arms); the harness re-runs unmodified on
  datacenter hardware for procurement comparisons.

## 6. Cost framing

Everything below is in this card's own currency — GPU-days on the single
RTX 3080 Ti — derived straight from the measured throughput in §4 (a campaign
of *T* ns costs *T* / (ns/day) GPU-days). Nothing is extrapolated.

| campaign | 64 at. (2.0 ns/day) | 1.4k at. (1.3) | 2.9k at. (0.57) | 11.5k at. (0.25) |
|---|---|---|---|---|
| 1 ns trajectory | 0.5 GPU-day | 0.8 GPU-day | 1.8 GPU-days | 4.0 GPU-days |
| 5 ns trajectory | 2.5 GPU-days | 3.8 GPU-days | 8.8 GPU-days | 20 GPU-days |

- **The one-time model-fix tax is small and already paid.** Repairing the
  soft-axis failure cost ~2 h of measured training wall-time on this same card
  (both fine-tune stages; budget ~1 GPU-day end-to-end with data prep and
  validation iterations). Amortized over any real MD campaign it is noise —
  validate and fine-tune once per material, then run.
- **Stay on the RTX** for screening and for single trajectories up to a few ns
  below ~3k atoms: a 5 ns run there is a long weekend, not a cluster request.
- **A second RTX card pays off** only for *throughput-parallel* work — many
  independent small/medium systems (screening ensembles, replica sweeps) — not
  for making one trajectory faster; these runs don't communicate, so two cards
  ≈ 2× jobs/day with no scaling loss.
- **Request A100/datacenter time** at the measured handoff: **> 11k atoms or
  multi-ns trajectories**, where a single 5 ns run already costs ~20 GPU-days
  (three weeks of wall-clock) on the RTX and above ~5k atoms this card is for
  relaxations and snapshots, not multi-ns MD. The A100 also reopens fp64 at
  1:2 if a reference trajectory needs it.
- **Vs DFT-AIMD (order-of-magnitude context only — not measured here):** a
  first-principles MD trajectory of these system sizes is categorically more
  expensive than any row above, by many orders of magnitude in core-hours; the
  point of MLIP deployment is that these GPU-day campaigns exist *at all* where
  AIMD would be infeasible. No specific DFT timing is claimed.

---

## Addendum (Part 2, 2026-07-06) — the host-gap fix, measured

Section 2's item 3 recommended "batching/CUDA-graph approaches" for the
host-bound regime. We ran that recommendation. Capturing the MACE force
evaluation into a CUDA graph (fixed topology, numerically exact — parity
max |ΔF| ≤ 8×10⁻⁷ eV/Å):

- ≤ ~512 atoms: **×4–9 per-step speedup** — and the §4 "below break-even, cuEq
  off" row becomes an *eager-only* rule: with graph capture the crossover
  disappears and cuEq wins at every size (×7.1 over e3nn+graph at 140 atoms).
- Many small cells (screening/ensembles): pack N cells into one captured graph
  — 8×140 atoms ran at **1.04 ms/cell** (×2.1 vs eager).
- ≥ ~3,000 atoms: kernels dominate; a graph adds ≈×1.1 — not worth the wiring.
- Caveat: per-step ceiling at frozen topology. Production MD needs padded
  capture (`mace.tools` padding utilities) or periodic recapture; budget that
  engineering before quoting end-to-end numbers. Full data:
  [cudagraph-study.md](cudagraph-study.md).
