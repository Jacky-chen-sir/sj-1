#!/bin/bash

set -euo pipefail
export HYDRA_FULL_ERROR=1

# ---------------- env ----------------
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/GTRS"

# Dataset root used by hydra configs (default_dataset_paths.yaml uses OPENSCENE_DATA_ROOT).
# Must contain: navsim_logs/<split> and sensor_blobs/<split>
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/bigdisk/GTRS/download}"

# ---------------- knobs ----------------
# MODE=train: default training (no cache)
MODE=${MODE:-train}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}

config=${config:-competition_training}
agent=${agent:-gtrs_diffusion_policy}
experiment_name=${experiment_name:-train_dp_guided_anchor}

bs=${bs:-6}
max_epochs=${max_epochs:-17}

# -------- LR schedule --------
# Enable OneCycle dynamic LR with warmup by default.
# We use DPAgent scheduler='cycle' and its config keys:
#   onecycle_max_lr, onecycle_total_steps, onecycle_pct_start, onecycle_div_factor
# Target behavior: max_lr=2e-3 -> final_lr=3e-4 (via div_factor), warmup=5%.
# Set USE_ONECYCLE_LR=0 to fall back to the original fixed lr (agent.lr).
USE_ONECYCLE_LR=${USE_ONECYCLE_LR:-1}
MAX_LR=${MAX_LR:-0.002}
FINAL_LR=${FINAL_LR:-0.0003}
WARMUP_PCT=${WARMUP_PCT:-0.05}

# Base lr for Adam (also used when USE_ONECYCLE_LR=0)
lr=${lr:-0.0002}

# Cache (disabled by default)
USE_CACHE=${USE_CACHE:-0}
DISABLE_CACHE=${DISABLE_CACHE:-1}

# Pretrained checkpoints
DP_INIT_CKPT_DEFAULT="/mnt/bigdisk/cache_GTRS/gtrs_dp.ckpt"
DENSE_CKPT_DEFAULT="/mnt/bigdisk/cache_GTRS/gtrs_dense_vov.ckpt"
VOV_CKPT_DEFAULT="$OPENSCENE_DATA_ROOT/models/dd3d_det_final.pth"

DP_INIT_CKPT=${DP_INIT_CKPT:-$DP_INIT_CKPT_DEFAULT}
DENSE_CKPT=${DENSE_CKPT:-$DENSE_CKPT_DEFAULT}
VOV_CKPT=${VOV_CKPT:-$VOV_CKPT_DEFAULT}

# Guidance
ANCHOR_TOPK=${ANCHOR_TOPK:-1}

# DataLoader (3 GPUs -> total workers ~= NUM_WORKERS * 3)
NUM_WORKERS=${NUM_WORKERS:-4}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-true}
PIN_MEMORY=${PIN_MEMORY:-true}

export DP_PREDS=${DP_PREDS:-none}
RESUME_CKPT_PATH=${RESUME_CKPT_PATH:-}

if [ "$DISABLE_CACHE" = "1" ]; then
  USE_CACHE=0
fi

echo "[INFO] MODE=${MODE}"
echo "[INFO] OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] USE_CACHE=${USE_CACHE} (cache_path=null)"
echo "[INFO] USE_ONECYCLE_LR=${USE_ONECYCLE_LR} MAX_LR=${MAX_LR} FINAL_LR=${FINAL_LR} WARMUP_PCT=${WARMUP_PCT}"
echo "[INFO] NUM_WORKERS=${NUM_WORKERS} PERSISTENT_WORKERS=${PERSISTENT_WORKERS} PIN_MEMORY=${PIN_MEMORY}"

required_paths=("$DP_INIT_CKPT" "$DENSE_CKPT" "$VOV_CKPT")
for p in "${required_paths[@]}"; do
  if [ ! -e "$p" ]; then
    echo "[FATAL] Missing required path: $p" >&2
    exit 1
  fi
done

cd "$NAVSIM_DEVKIT_ROOT"

# OneCycleLR: final_lr ~= max_lr / div_factor  => div_factor = max_lr / final_lr
# Use bash+python with proper variable expansion.
if [ "$USE_ONECYCLE_LR" = "1" ]; then
  DIV_FACTOR=$(python - <<PY
max_lr=float("$MAX_LR")
final_lr=float("$FINAL_LR")
if final_lr <= 0 or max_lr <= 0:
    raise ValueError("MAX_LR and FINAL_LR must be > 0")
print(max_lr/final_lr)
PY
  )
else
  DIV_FACTOR="25.0"
fi

# Total steps: approximate steps_per_epoch * max_epochs.
ONECYCLE_TOTAL_STEPS=${ONECYCLE_TOTAL_STEPS:-0}

LR_OVERRIDES=""
if [ "$USE_ONECYCLE_LR" = "1" ]; then
  # NOTE: DPAgent expects scheduler name 'cycle'
  if [ "$ONECYCLE_TOTAL_STEPS" = "0" ]; then
    ONECYCLE_TOTAL_STEPS=$(( max_epochs * 200 ))
  fi
  LR_OVERRIDES="+agent.config.scheduler=cycle +agent.config.onecycle_max_lr=${MAX_LR} +agent.config.onecycle_total_steps=${ONECYCLE_TOTAL_STEPS} +agent.config.onecycle_pct_start=${WARMUP_PCT} +agent.config.onecycle_div_factor=${DIV_FACTOR}"
else
  LR_OVERRIDES=""
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_dense.py" \
  --config-name "${config}" \
  agent=${agent} \
  experiment_name=${experiment_name} \
  train_test_split=navtrain \
  dataloader.params.batch_size=${bs} \
  dataloader.params.num_workers=${NUM_WORKERS} \
  ++dataloader.params.persistent_workers=${PERSISTENT_WORKERS} \
  ++dataloader.params.pin_memory=${PIN_MEMORY} \
  ~trainer.params.strategy \
  trainer.params.max_epochs=${max_epochs} \
  trainer.params.precision=32 \
  agent.lr=${lr} \
  ${LR_OVERRIDES} \
  agent.checkpoint_path="${DP_INIT_CKPT}" \
  agent.config.ckpt_path="${experiment_name}" \
  +agent.config.guidance.enable=true \
  +agent.config.guidance.dense_checkpoint_path="${DENSE_CKPT}" \
  +agent.config.guidance.vov_ckpt="${VOV_CKPT}" \
  +agent.config.guidance.anchor_topk=${ANCHOR_TOPK} \
  cache_path=null \
  ${RESUME_CKPT_PATH:+resume_ckpt_path="${RESUME_CKPT_PATH}"}
