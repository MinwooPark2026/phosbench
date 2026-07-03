#!/usr/bin/env bash
# Fine-tune retry with stage-two (SWA) energy-weighted refit.
# Attempt 1 (scripts/51_finetune.sh) underfit energies (valid RMSE_E
# ~106 meV/atom) and left the armchair axis unrepaired - the soft-direction
# lattice constant is an energy-landscape property, so stage two re-weights
# energies 1000:100 per standard MACE practice. One retry, then we accept
# whatever the data says.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-scicomp}"  # adjust to your environment name
conda activate "$CONDA_ENV"
mkdir -p results/logs finetune

(
  cd finetune
  mace_run_train \
    --name phos_ft2 \
    --foundation_model ~/.cache/mace/maceomat0mediummodel \
    --multiheads_finetuning False \
    --train_file ../data/gap20_train.xyz \
    --valid_fraction 0.05 \
    --energy_key REF_energy \
    --forces_key REF_forces \
    --E0s average \
    --loss weighted --energy_weight 10.0 --forces_weight 100.0 \
    --lr 3e-4 \
    --scaling rms_forces_scaling \
    --batch_size 4 --valid_batch_size 8 \
    --max_num_epochs 48 \
    --swa --start_swa 30 \
    --swa_energy_weight 1000.0 --swa_forces_weight 100.0 --swa_lr 1e-4 \
    --ema --ema_decay 0.999 \
    --amsgrad \
    --default_dtype float32 \
    --device cuda \
    --seed 7 \
    --restart_latest
) 2>&1 | tee -a results/logs/53_finetune_swa.log
RC=${PIPESTATUS[0]}
echo "FT2_TRAIN_EXIT rc=$RC"
[ "$RC" -ne 0 ] && exit "$RC"

echo "=== FT2 armchair gate (stagetwo model preferred if produced)"
MODEL=finetune/phos_ft2_stagetwo.model
[ -f "$MODEL" ] || MODEL=finetune/phos_ft2.model
python scripts/00_make_structure.py \
  --model "$MODEL" \
  --out structures/phosphorene_ft2.extxyz \
  2>&1 | tee results/logs/54_ft2_gate.log
echo "FT2_GATE_EXIT rc=$?"
