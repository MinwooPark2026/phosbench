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
| kernel profile (e3nn baseline) | equivariant tensor products + scatter/gather dominate; at 2,944 atoms the GPU is busy 60 % of each MD step, top kernel is index-backward (gather/scatter), then SGEMMs |
| precision | model weights/activations fp32 by default; fp64 runs at 1/64 FLOP rate on GA102 → measured ×3.2–×10 step-time penalty and ×2 memory; fp64 is for reference data only on this hardware |
| CPU↔GPU traffic | negligible (< 0.2 % of step time): positions down / forces up per step; this workload is **not** transfer-bound on a workstation |
| host vs device | the real split: at ≤ 140 atoms, ≥ 92 % of step time is host-side Python/ASE/launch overhead — the GPU idles regardless of backend |
| memory scaling | activation memory grows steeply with neighbors; e3nn/medium OOMs at ~3k atoms on 12 GB |
| IO / data types | extxyz trajectories, float32 tensors; checkpoint-restart cost negligible vs step time |

## 2. Bottleneck analysis → what to change

1. **≥ ~500 atoms: kernels are the bottleneck → switch them.** Enabling
   cuEquivariance (`enable_cueq=True`, one line) delivers ×1.4–×5.0 at 2,944
   atoms (bigger models gain more) and cuts peak VRAM ~5×, raising your
   reachable system size from ~3k to ~11.5k atoms (medium model).
2. **≤ ~500 atoms: the host is the bottleneck → cuEq won't help.** Below the
   measured break-even (~310–980 atoms depending on model size) cuEq is
   *slower* (×0.74–0.96). Keep e3nn, or batch many small systems.
3. **After cuEq, you are host-bound again** (GPU ≤ 10 % busy at 3k atoms).
   The next step is not a faster kernel — it is the integration layer:
   LAMMPS ML-IAP (Kokkos) for production runs, or batching/CUDA-graph
   approaches. Plan the workflow migration before scaling up.

## 3. Accuracy gates before you trust fp32 production MD

Run these once per material (scripts in this repo; minutes of GPU time):

- **Backend parity**: cuEq vs e3nn at fp32 — here ΔE = 0.000 meV/atom,
  max |ΔF| = 5×10⁻⁴ meV/Å. Verify cuEq kernels actually run (Nsight;
  `segmented_polynomial_*`) — we caught a parser blind spot this way.
- **Observable-level precision check**: phonons, elastic constants, NVE drift
  vs an e3nn/fp64 reference. Here fp32 errors are ≤ 0.2 % of the
  model-vs-DFT error on every observable → fp32 production MD is safe
  *for this system*.
- **Protocol caveat**: fp32 force noise corrupts finite-difference property
  workflows at small displacements (spurious imaginary acoustic modes at
  0.01 Å that vanish at 0.05 Å). Policy: displaced-force property
  evaluations on e3nn/fp64 (or CPU), production MD on cuEq/fp32.
- **Model validation is the real risk, not precision.** All current MACE
  foundation models compress phosphorene's soft armchair axis by 7–10 %.
  Zero-shot ≠ production-ready: validate the soft direction of *your*
  material; budget a fine-tune (open GAP-20 dataset) if it fails.

## 4. Recommendation matrix

| your run | do this |
|---|---|
| screening, < 500 atoms | e3nn/fp32 on the workstation GPU (never CPU: GPU wins ×13 even at 64 atoms) |
| production MD, 0.5k–11k atoms | cuEq/fp32 on the workstation |
| > 11k atoms or multi-GPU | datacenter / LAMMPS-Kokkos path; A100 fp64 at 1:2 also reopens fp64 if you need it |
| phonons / elastic / stability verdicts | hybrid: e3nn/fp64 displaced forces, cuEq/fp32 dynamics |
| geometry relaxation reference | e3nn/fp64, small cells only (OOM at 1.4k atoms) |

## 5. Effort & risk

- Adoption cost of cuEq on an existing MACE/ASE workflow: **one flag**, plus
  the parity gate above. Known sharp edges we hit and documented: cuEq+fp64
  unsupported (MACE #1203/#1298), torch.compile incompatibility (cuEq #77),
  MACE slab-stress inconsistency (×17.8, found here — use energy-curvature
  fits for elastic constants until fixed upstream).
- Everything in this memo is reproducible from the repo in ~half a GPU-day;
  the harness re-runs unmodified on datacenter hardware for procurement
  comparisons.
