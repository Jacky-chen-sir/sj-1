#!/usr/bin/env bash
# Official GTRS-Aug scorer + FM DP proposals on navtest (combined inference).
# Default: FM joint ep18 DP proposals (best standalone PDM so far).
#
# Usage:
#   bash scripts/evaluation/eval_aug_with_fm_dp_navtest.sh
#   GPU=0 BS=24 WORKERS=8 bash scripts/evaluation/eval_aug_with_fm_dp_navtest.sh
#   DP_PREDS=/path/to/other_navtest_fm.pkl bash ...

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export TMPDIR="${TMPDIR:-/mnt/bigdisk/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/mnt/bigdisk/tmp/ray}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

split="${SPLIT:-navtest}"
gpu="${GPU:-0}"
workers="${WORKERS:-8}"
batch_size="${BS:-${BATCH_SIZE:-24}}"

DEFAULT_DP_PREDS="${NAVSIM_EXP_ROOT}/train_dp_official_fm_joint/epoch=18-step=81776_navtest_fm.pkl"
DP_PREDS_PATH="${DP_PREDS:-${DEFAULT_DP_PREDS}}"
AUG_CKPT="${AUG_CKPT:-${NAVSIM_DEVKIT_ROOT}/data/models/gtrs_aug_model.ckpt}"
VOV_CKPT="${VOV_CKPT:-${OPENSCENE_DATA_ROOT}/models/dd3d_det_final.pth}"
VOCAB_8192="${VOCAB_8192:-${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy}"
metric_cache_path="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/${split}_two_stage_metric_cache}"
sensor_path="${OPENSCENE_DATA_ROOT}/sensor_blobs/test/test"

dp_stem="$(basename "${DP_PREDS_PATH}" .pkl)"
# Hydra-safe experiment name (no '=' from epoch=xx filenames)
dp_tag="${dp_stem//=/-}"
exp_root="${EXP_ROOT_NAME:-train_gtrs_aug_official_fm_dp}"
experiment_name="${EXP_NAME:-${exp_root}/aug-${dp_tag}-b${batch_size}-w${workers}}"
SUBSCORE_PATH="${SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}/${exp_root}/${dp_stem}_${split}_aug.pkl}"

required_paths=(
  "${DP_PREDS_PATH}"
  "${AUG_CKPT}"
  "${VOV_CKPT}"
  "${VOCAB_8192}"
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

mkdir -p "$(dirname "${SUBSCORE_PATH}")" "${NAVSIM_EXP_ROOT}/logs"

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

export DP_PREDS="${DP_PREDS_PATH}"
export SUBSCORE_PATH
export SKIP_INFER="${SKIP_INFER:-0}"

echo "[RUN] Official AUG + FM DP proposals"
echo "      dp_preds=${DP_PREDS}"
echo "      aug_ckpt=${AUG_CKPT}"
echo "      split=${split} gpu=${gpu} bs=${batch_size} workers=${workers}"
echo "      subscore=${SUBSCORE_PATH}"
echo "      experiment=${experiment_name}"

CUDA_VISIBLE_DEVICES="${gpu}" python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2_aug.py" \
  agent=gtrs_aug_vov \
  +combined_inference=true \
  dataloader.params.batch_size="${batch_size}" \
  dataloader.params.num_workers="${workers}" \
  dataloader.params.pin_memory=true \
  dataloader.params.prefetch_factor=2 \
  agent.checkpoint_path="${AUG_CKPT}" \
  agent.config.vocab_path="${VOCAB_8192}" \
  agent.config.vov_ckpt="${VOV_CKPT}" \
  agent.config.training=false \
  agent.config.only_ori_input=true \
  agent.config.inference.model=teacher \
  agent.config.lab.use_first_stage_traj_in_infer=true \
  ++trainer.params.accelerator=gpu \
  ++trainer.params.devices=1 \
  trainer.params.strategy=auto \
  trainer.params.precision=32 \
  worker.threads_per_node="${workers}" \
  worker.log_to_driver=false \
  experiment_name="${experiment_name}" \
  +cache_path=null \
  metric_cache_path="${metric_cache_path}" \
  original_sensor_path="${sensor_path}" \
  train_test_split="${split}"
