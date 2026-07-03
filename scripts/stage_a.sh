#!/usr/bin/env bash
# Stage A gate runner. Submit on the workstation as ONE jobq job:
#   jobq submit "cd ~/phosbench && bash scripts/stage_a.sh"
# Order matters: 00 builds the canonical structure 01/02 depend on; 02 may
# rebuild it with the physics-gate winner model.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-scicomp}"  # adjust to your environment name
conda activate "$CONDA_ENV"
mkdir -p results/logs results/raw/nsys
RC=0

echo "=== [A0] canonical structure (medium baseline)"
python scripts/00_make_structure.py --model medium 2>&1 \
  | tee results/logs/00_structure.log || RC=1

echo "=== [A1] numerical consistency gates (3-cell parity + cueq-fp64 probe)"
python scripts/01_validate_consistency.py 2>&1 \
  | tee results/logs/01_consistency.log || RC=1

echo "=== [A2] kernel-truth nsys micro-trace (silent-fallback detector)"
# trace name must match 31_parse_nsys.py's <model>_<backend>_<dtype>_<nx>x<ny>
# tag so the KERNEL_TRUTH gate recognises it as a cueq trace and can FAIL.
nsys profile --trace=cuda,nvtx --force-overwrite true \
  -o results/raw/nsys/medium_cueq_float32_5x7 \
  python scripts/10_sweep_throughput.py --single cueq float32 medium 5 7 force_call --n-steps 10 \
  2>&1 | tee results/logs/02_kernel_truth.log || RC=1
python scripts/31_parse_nsys.py 2>&1 \
  | tee -a results/logs/02_kernel_truth.log || RC=1

echo "=== [A3] physics gate + foundation-model selection"
python scripts/02_gate_physics.py 2>&1 \
  | tee results/logs/03_gate_physics.log || RC=1

echo "STAGE_A_EXIT rc=$RC"
exit $RC
