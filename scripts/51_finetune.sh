#!/usr/bin/env bash
# Stretch - fine-tune OMAT-0 medium on GAP-20 P data, then re-run the
# armchair-axis geometry gate with the fine-tuned model.
# Submit on the workstation:  jobq submit "cd ~/phosbench && bash scripts/51_finetune.sh"
# fp32 training on purpose (GA102 fp64 is 1:64); E/F only - no virials, the
# slab-stress finding makes them untrustworthy anyway. --restart_latest keeps
# the run resumable across the box's dual-boot reboots (jobq re-enqueues).
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-scicomp}"  # adjust to your environment name
conda activate "$CONDA_ENV"
mkdir -p results/logs finetune

(
  cd finetune
  mace_run_train \
    --name phos_ft \
    --foundation_model ~/.cache/mace/maceomat0mediummodel \
    --multiheads_finetuning False \
    --train_file ../data/gap20_train.xyz \
    --valid_fraction 0.05 \
    --energy_key REF_energy \
    --forces_key REF_forces \
    --E0s average \
    --loss weighted --energy_weight 1.0 --forces_weight 100.0 \
    --lr 1e-4 \
    --scaling rms_forces_scaling \
    --batch_size 4 --valid_batch_size 8 \
    --max_num_epochs 40 \
    --ema --ema_decay 0.999 \
    --amsgrad \
    --default_dtype float32 \
    --device cuda \
    --seed 7 \
    --restart_latest
) 2>&1 | tee -a results/logs/51_finetune.log
RC=${PIPESTATUS[0]}
echo "FINETUNE_TRAIN_EXIT rc=$RC"
[ "$RC" -ne 0 ] && exit "$RC"

echo "=== FT armchair gate: relax monolayer with the fine-tuned model"
python scripts/00_make_structure.py \
  --model finetune/phos_ft.model \
  --out structures/phosphorene_ft.extxyz \
  2>&1 | tee results/logs/52_ft_gate.log
echo "FINETUNE_GATE_EXIT rc=$?"
