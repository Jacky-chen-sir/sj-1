#!/bin/bash

set -euo pipefail

# Ensure we run under the correct conda env (needed for hydra/torch/lightning)
# If already activated, this is a no-op.
if [ "${CONDA_DEFAULT_ENV-}" != "conda_gtrs" ]; then
  # Try common conda init locations
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "[FATAL] conda not found. Please install conda or source conda.sh before running." >&2
    exit 1
  fi

  conda activate conda_gtrs
fi

echo "[INFO] Using python: $(which python)" 
python -c "import hydra; import torch; import pytorch_lightning as pl; print('[INFO] env ok:', hydra.__version__)" >/dev/null

export HYDRA_FULL_ERROR=1

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/GTRS"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
export NAVSIM_TRAJPDM_ROOT="$HOME/navsim_workspace/dataset/traj_pdm_v2"

NUM_NODES=1
MASTER_ADDR=127.0.0.1
NODE_RANK=0
config="competition_training"
experiment_name=train_dense
agent=gtrs_dense_vov
lr=0.0002

# Required by navsim/planning/training/agent_lightning_module.py
# Use the provided dp predictions pkl in this repo by default.
export DP_PREDS="${NAVSIM_DEVKIT_ROOT}/data/models/dp_preds.pkl"

# ---- performance knobs (tune here) ----
# per-GPU batch. start from 12 on 3090 (24GB). if OOM -> 10/8.
bs=24
max_epochs=17
# NOTE: /dev/shm bus error is common on multi-worker dataloaders. Use workers=0 for stability.
workers=0
prefetch=1
# --------------------------------------

# Dataloader worker debugging (more actionable stack traces)
export TORCH_SHOW_CPP_STACKTRACES=1

CACHE_DIR="/mnt/bigdisk/cache_GTRS"
GT_DIR="$HOME/navsim_workspace/dataset/traj_pdm_v2/ori"

# Prefer official locations
MODEL_DIR="$OPENSCENE_DATA_ROOT/models"
VOV_CKPT_PATH="${MODEL_DIR}/dd3d_det_final.pth"

# Dense ckpt: use the local cached ckpt if present, otherwise download to CACHE_DIR
BEV_CKPT_PATH="${CACHE_DIR}/gtrs_dense_vov.ckpt"

mkdir -p "$CACHE_DIR" "$MODEL_DIR"

if [ ! -f "$VOV_CKPT_PATH" ]; then
  echo "[INFO] dd3d_det_final.pth not found, downloading to $VOV_CKPT_PATH"
  wget -O "$VOV_CKPT_PATH" "https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/dd3d_det_final.pth"
fi

if [ ! -f "$BEV_CKPT_PATH" ]; then
  echo "[INFO] gtrs_dense_vov.ckpt not found, downloading to $BEV_CKPT_PATH"
  wget -O "$BEV_CKPT_PATH" "https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/gtrs_dense_vov.ckpt"
fi

# Sanity checks (fail fast with clear messages)
required_paths=(
  "$CACHE_DIR"
  "$BEV_CKPT_PATH"
  "$VOV_CKPT_PATH"
  "$GT_DIR/navtrain_16384.pkl"
  "$NAVSIM_DEVKIT_ROOT/traj_final/16384.npy"
)

for p in "${required_paths[@]}"; do
  if [ ! -e "$p" ]; then
    echo "[FATAL] Missing required path: $p" >&2
    exit 1
  fi
done

echo "[INFO] All required paths exist."

# NCCL stability/perf (helps with multi-GPU on some rigs)
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

# Reduce DataLoader shared-memory pressure (/dev/shm) when using many workers.
# This prevents "Unexpected bus error" from DataLoader workers.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CUDA_VISIBLE_DEVICES=0,1 \
MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
    python -c "import torch.multiprocessing as mp; mp.set_sharing_strategy('file_system')" >/dev/null 2>&1 || true

CUDA_VISIBLE_DEVICES=0,1 \
MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_dense.py \
        --config-name ${config} \
        trainer.params.num_nodes=${NUM_NODES} \
        agent=${agent} \
        experiment_name=${experiment_name} \
        train_test_split=navtrain \
        dataloader.params.batch_size=${bs} \
        dataloader.params.num_workers=${workers} \
        +dataloader.params.persistent_workers=false \
        dataloader.params.pin_memory=false \
        +dataloader.params.prefetch_factor=${prefetch} \
        trainer.params.limit_train_batches=1.0 \
        trainer.params.max_epochs=${max_epochs} \
        trainer.params.precision=16-mixed \
        agent.checkpoint_path="${BEV_CKPT_PATH}" \
        agent.config.vov_ckpt="${VOV_CKPT_PATH}" \
        agent.pdm_gt_path="${GT_DIR}/navtrain_16384.pkl" \
        agent.config.vocab_path="${NAVSIM_DEVKIT_ROOT}/traj_final/16384.npy" \
        +agent.config.freeze_perception=true \
        agent.lr=${lr} \
        cache_path="${CACHE_DIR}" \
        force_cache_computation=false \
        use_cache_without_dataset=true