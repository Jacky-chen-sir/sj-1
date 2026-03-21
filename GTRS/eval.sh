#!/bin/bash
# Dense evaluation script for NAVSIM (PDM score)
#
# Example (navtest):
# CKPT_PATH="/home/ws/navsim_workspace/GTRS/path/gtrs_dense_vov.ckpt" ./eval.sh

# Prevent noisy xtrace output (e.g. lines starting with '+++')
set +x
set -e

# ========== Environment ==========
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$HOME/navsim_workspace/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$HOME/navsim_workspace/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$HOME/navsim_workspace/GTRS}"

# IMPORTANT:
# OPENSCENE_DATA_ROOT is used by agent configs to locate:
# - traj_pdm_v2/ori/navtrain_16384.pkl (pdm_gt_path)
# - models/dd3d_det_final.pth (vov_ckpt default)
# Default to this workspace's dataset folder.
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$HOME/navsim_workspace/dataset}"

# Default navsim logs location under this dataset folder.
# (Contains navsim_logs/{trainval,test,...})
export NAVSIM_LOGS_ROOT="${NAVSIM_LOGS_ROOT:-$OPENSCENE_DATA_ROOT/navsim_logs}"

# ========== Fixed settings for Dense eval ==========
SPLIT="navtest"
AGENT="gtrs_dense_vov"
TWO_STAGE_SPLIT="navtest_two_stage"
LOG_SPLIT="test"

# Your checkpoint path (.ckpt). MUST be provided.
CKPT_PATH="${CKPT_PATH:-}"

# Optional override for perception backbone
VOV_CKPT_PATH="${VOV_CKPT_PATH:-$OPENSCENE_DATA_ROOT/models/dd3d_det_final.pth}"

# Metric cache location
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/${SPLIT}_two_stage_metric_cache}"

# Runtime
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-29501}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ -z "$CKPT_PATH" ]; then
  echo "ERROR: CKPT_PATH is empty. Example:"
  echo "  CKPT_PATH=\"$NAVSIM_DEVKIT_ROOT/path/gtrs_dense_vov.ckpt\" ./eval.sh"
  exit 1
fi

cd "$NAVSIM_DEVKIT_ROOT"

# ========== Sanity checks ==========
if [ ! -d "$NAVSIM_LOGS_ROOT" ]; then
  echo "ERROR: NAVSIM_LOGS_ROOT not found: $NAVSIM_LOGS_ROOT"
  exit 1
fi
if [ ! -d "$NAVSIM_LOGS_ROOT/$LOG_SPLIT" ]; then
  echo "ERROR: Split folder not found: $NAVSIM_LOGS_ROOT/$LOG_SPLIT"
  echo "Available folders in NAVSIM_LOGS_ROOT:" 
  ls -1 "$NAVSIM_LOGS_ROOT" || true
  exit 1
fi

# Sensor blobs are required for image loading during scoring.
SENSOR_BLOBS_ROOT="$OPENSCENE_DATA_ROOT/sensor_blobs/$LOG_SPLIT"
if [ ! -d "$SENSOR_BLOBS_ROOT" ]; then
  echo "ERROR: sensor_blobs split folder not found: $SENSOR_BLOBS_ROOT"
  echo "Fix: set OPENSCENE_DATA_ROOT to the dataset root that contains sensor_blobs/$LOG_SPLIT,"
  echo "or download the $LOG_SPLIT sensor blobs following docs/install.md."
  exit 1
fi

# Quick check: expect at least some jpg files under CAM_F0.
JPG_COUNT=$(find "$SENSOR_BLOBS_ROOT" -maxdepth 4 -type f -name "*.jpg" 2>/dev/null | head -n 1 | wc -l)
if [ "$JPG_COUNT" -eq 0 ]; then
  echo "ERROR: No image .jpg found under: $SENSOR_BLOBS_ROOT"
  echo "This usually means navtest (test) sensor_blobs are missing or the symlink points to an incomplete dataset."
  echo "Your current test sensor_blobs path is:" 
  ls -lah "$OPENSCENE_DATA_ROOT/sensor_blobs/test" 2>/dev/null || true
  echo "Fix options:"
  echo "  1) Download the test sensor_blobs split per docs/install.md"
  echo "  2) Point OPENSCENE_DATA_ROOT to where the complete sensor_blobs/test lives"
  exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: CKPT_PATH not found: $CKPT_PATH"
  exit 1
fi
if [ ! -f "$VOV_CKPT_PATH" ]; then
  echo "ERROR: VOV_CKPT_PATH not found: $VOV_CKPT_PATH"
  echo "Set VOV_CKPT_PATH to your dd3d_det_final.pth"
  exit 1
fi

# ========== Step 1: Metric caching (skip if already exists) ==========
# A valid metric cache must contain metadata/*.csv, otherwise MetricCacheLoader will crash.
METADATA_DIR="$METRIC_CACHE_PATH/metadata"
NEED_CACHE_BUILD=0
if [ ! -d "$METRIC_CACHE_PATH" ]; then
  NEED_CACHE_BUILD=1
elif [ ! -d "$METADATA_DIR" ]; then
  NEED_CACHE_BUILD=1
else
  CSV_COUNT=$(ls -1 "$METADATA_DIR"/*.csv 2>/dev/null | wc -l)
  if [ "$CSV_COUNT" -eq 0 ]; then
    NEED_CACHE_BUILD=1
  fi
fi

if [ "$NEED_CACHE_BUILD" -eq 0 ]; then
  echo "[eval] Metric cache exists at: $METRIC_CACHE_PATH (skip caching)"
else
  echo "[eval] Metric cache invalid or missing, rebuilding: $METRIC_CACHE_PATH"
  rm -rf "$METRIC_CACHE_PATH"
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  MASTER_PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$NUM_NODES NODE_RANK=$NODE_RANK \
    python navsim/planning/script/run_metric_caching.py \
      train_test_split=${TWO_STAGE_SPLIT} \
      metric_cache_path="$METRIC_CACHE_PATH"
fi

# ========== Step 2: Score ==========
EXP_NAME="eval_${AGENT}_${SPLIT}_$(date +%Y.%m.%d.%H.%M.%S)"

# run_pdm_score_gpu_v2.py requires SUBSCORE_PATH
export PROGRESS_MODE="${PROGRESS_MODE:-eval}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$NAVSIM_EXP_ROOT/$EXP_NAME}"
mkdir -p "$EVAL_OUT_DIR"
export SUBSCORE_PATH="${SUBSCORE_PATH:-$EVAL_OUT_DIR/subscores.pkl}"

echo "[eval] Scoring ckpt: $CKPT_PATH"
echo "[eval] Metric cache: $METRIC_CACHE_PATH"
echo "[eval] VOV_CKPT_PATH: $VOV_CKPT_PATH"
echo "[eval] SUBSCORE_PATH: $SUBSCORE_PATH"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
MASTER_PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$NUM_NODES NODE_RANK=$NODE_RANK \
  python navsim/planning/script/run_pdm_score_gpu_v2.py \
    agent=$AGENT \
    train_test_split=${TWO_STAGE_SPLIT} \
    experiment_name=$EXP_NAME \
    agent.checkpoint_path="$CKPT_PATH" \
    metric_cache_path="$METRIC_CACHE_PATH" \
    agent.config.vov_ckpt="$VOV_CKPT_PATH"
