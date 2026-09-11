#!/usr/bin/env bash
# Evaluate FM joint DP checkpoint on navtest (PDM score).
# Usage:
#   bash scripts/evaluation/eval_dp_fm_joint_navtest.sh
#   CKPT=/path/to.ckpt GPU=0 BS=8 WORKERS=8 bash scripts/evaluation/eval_dp_fm_joint_navtest.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -n "${NAVSIM_DEVKIT_ROOT:-}" ] && [ "${NAVSIM_DEVKIT_ROOT}" != "${ROOT_DIR}" ]; then
  echo "[WARN] Ignoring NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
  echo "[WARN] Using script repo instead: ${ROOT_DIR}"
fi
export NAVSIM_DEVKIT_ROOT="${ROOT_DIR}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/home/ws/navsim_workspace/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_TRAJPDM_ROOT="${NAVSIM_TRAJPDM_ROOT:-${OPENSCENE_DATA_ROOT}/traj_pdm_v2}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PROGRESS_MODE="${PROGRESS_MODE:-eval}"
export REUSE_SUBSCORE_IF_EXISTS="${REUSE_SUBSCORE_IF_EXISTS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"

split="${SPLIT:-navtest}"
gpu="${GPU:-0}"
workers="${WORKERS:-8}"
batch_size="${BS:-${BATCH_SIZE:-8}}"
fm_steps="${FM_NUM_INFERENCE_STEPS:-20}"

DEFAULT_CKPT="$(ls -1t "${NAVSIM_EXP_ROOT}/train_dp_official_fm_joint"/epoch=*.ckpt 2>/dev/null | head -n 1 || true)"
ckpt="${CKPT:-${DEFAULT_CKPT}}"

if [ -z "${ckpt}" ] || [ ! -f "${ckpt}" ]; then
  echo "[ERROR] Checkpoint not found: ${ckpt:-<empty>}" >&2
  exit 1
fi

metric_cache_path="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/${split}_two_stage_metric_cache}"
sensor_path="${OPENSCENE_DATA_ROOT}/sensor_blobs/test/test"
ckpt_stem="$(basename "${ckpt}" .ckpt)"
# Hydra treats '=' as override syntax; sanitize for experiment_name only.
ckpt_tag="${ckpt_stem//=/-}"
experiment_name="${EXP_NAME:-train_dp_official_fm_joint/navtest-${ckpt_tag}-fm${fm_steps}-b${batch_size}-w${workers}}"

mkdir -p "${NAVSIM_EXP_ROOT}/train_dp_official_fm_joint" "${NAVSIM_EXP_ROOT}/logs"
EMPTY_DP_PREDS="${NAVSIM_EXP_ROOT}/train_dp_official_fm_joint/empty_dp_preds.pkl"
python - <<PY
import pickle
from pathlib import Path
p = Path("${EMPTY_DP_PREDS}")
if not p.exists():
    pickle.dump({}, open(p, "wb"))
PY
export DP_PREDS="${EMPTY_DP_PREDS}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}/train_dp_official_fm_joint/${ckpt_stem}_${split}_fm.pkl}"

required_paths=(
  "${ckpt}"
  "${metric_cache_path}"
  "${OPENSCENE_DATA_ROOT}/navsim_logs/test"
  "${sensor_path}"
)
for path in "${required_paths[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${SUBSCORE_PATH}")"

if [ "${FORCE_RERUN:-0}" != "1" ]; then
  completed_csv=$(find "${NAVSIM_EXP_ROOT}/${experiment_name}" -type f -name "*.csv" 2>/dev/null | head -n 1 || true)
  if [ -n "${completed_csv}" ]; then
    echo "[SKIP] completed CSV exists: ${completed_csv}"
    exit 0
  fi
fi

cd "${NAVSIM_DEVKIT_ROOT}"
source /home/ws/anaconda3/etc/profile.d/conda.sh
conda activate conda_gtrs

echo "[RUN] FM joint DP navtest evaluation"
echo "      ckpt=${ckpt}"
echo "      split=${split} gpu=${gpu} bs=${batch_size} workers=${workers}"
echo "      fm_steps=${fm_steps}"
echo "      subscore=${SUBSCORE_PATH}"
echo "      experiment=${experiment_name}"

CUDA_VISIBLE_DEVICES="${gpu}" python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py" \
  agent=gtrs_diffusion_policy \
  dataloader.params.batch_size="${batch_size}" \
  dataloader.params.num_workers=0 \
  dataloader.params.pin_memory=false \
  '~dataloader.params.prefetch_factor' \
  "agent.checkpoint_path=\"${ckpt}\"" \
  ++trainer.params.accelerator=gpu \
  ++trainer.params.devices=1 \
  trainer.params.strategy=auto \
  trainer.params.precision=32 \
  worker.threads_per_node="${workers}" \
  worker.log_to_driver=false \
  ++agent.config.use_flow_matching=true \
  ++agent.config.fm_num_inference_steps="${fm_steps}" \
  experiment_name="${experiment_name}" \
  +cache_path=null \
  metric_cache_path="${metric_cache_path}" \
  original_sensor_path="${sensor_path}" \
  train_test_split="${split}"
