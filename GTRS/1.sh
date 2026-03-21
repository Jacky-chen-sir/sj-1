#!/bin/bash

set -euo pipefail

export HYDRA_FULL_ERROR=1

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/GTRS"

# Dataset root used by hydra configs (default_dataset_paths.yaml uses OPENSCENE_DATA_ROOT)
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/bigdisk/GTRS/download}"
export NAVSIM_TRAJPDM_ROOT="$HOME/navsim_workspace/dataset/traj_pdm_v2"

# ----------------- user knobs -----------------
split=${1:-navhard_two_stage}
experiment_name=${2:-guided_dp_eval}

# 节点传入位置
DP_CKPT_DEFAULT="/mnt/bigdisk/cache_GTRS/gtrs_dp.ckpt"
DENSE_CKPT_DEFAULT="/mnt/bigdisk/cache_GTRS/gtrs_dense_vov.ckpt"

# Allow override via environment variables
export DP_CKPT="${DP_CKPT:-$DP_CKPT_DEFAULT}"
export DENSE_CKPT="${DENSE_CKPT:-$DENSE_CKPT_DEFAULT}"

CACHE_DIR="${CACHE_DIR:-/mnt/bigdisk/cache_GTRS}"
MODEL_DIR="$OPENSCENE_DATA_ROOT/models"
VOV_CKPT_PATH="${MODEL_DIR}/dd3d_det_final.pth"
# ------------------------------------------------

required_paths=(
  "$DP_CKPT"
  "$DENSE_CKPT"
  "$VOV_CKPT_PATH"
  "$NAVSIM_DEVKIT_ROOT/traj_final/16384.npy"
)

for p in "${required_paths[@]}"; do
  if [ ! -e "$p" ]; then
    echo "[FATAL] Missing required path: $p" >&2
    exit 1
  fi
done

cd "$NAVSIM_DEVKIT_ROOT"

# Inference/eval is configured similarly to docs/gtrs_inference.md
python navsim/planning/script/run_pdm_score_gpu_v2.py \
  --config-name default_run_pdm_score_gpu \
  agent=gtrs_guided_dp \
  experiment_name="$experiment_name" \
  train_test_split="$split" \
  dataloader.params.batch_size=16 \
  dataloader.params.num_workers=2 \
  +dataloader.params.persistent_workers=false \
  dataloader.params.pin_memory=false \
  trainer.params.precision=16-mixed \
  +agent.dp_checkpoint_path="$DP_CKPT" \
  +agent.dense_checkpoint_path="$DENSE_CKPT" \
  agent.dense_config.vov_ckpt="$VOV_CKPT_PATH" \
  agent.dense_config.vocab_path="$NAVSIM_DEVKIT_ROOT/traj_final/16384.npy" \
  metric_cache_path="$NAVSIM_EXP_ROOT/${split}_metric_cache" \
  +cache_path="$CACHE_DIR"
