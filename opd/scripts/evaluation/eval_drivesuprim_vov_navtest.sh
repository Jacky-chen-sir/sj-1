#!/usr/bin/env bash
# DriveSuprim (VoV) scorer + optional FM DP proposals on navtest.
#
# Usage:
#   # DriveSuprim alone (official ckpt)
#   bash scripts/evaluation/eval_drivesuprim_vov_navtest.sh
#
#   # DriveSuprim + FM DP ep18 proposals (combined)
#   WITH_FM_DP=1 bash scripts/evaluation/eval_drivesuprim_vov_navtest.sh
#
#   CKPT=/path/to.ckpt GPU=0 BS=8 WORKERS=8 bash ...

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
batch_size="${BS:-${BATCH_SIZE:-8}}"
fm_steps="${FM_NUM_INFERENCE_STEPS:-20}"
num_refinement_stage="${NUM_REFINEMENT_STAGE:-1}"
stage_layers="${STAGE_LAYERS:-3}"
topks="${TOPKS:-256}"
inference_model="${INFERENCE_MODEL:-teacher}"

DEFAULT_CKPT="${NAVSIM_EXP_ROOT}/model_ckpt/drivesuprim_vov.ckpt"
ckpt="${CKPT:-${DEFAULT_CKPT}}"
metric_cache_path="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/navtest_two_stage_metric_cache}"
sensor_path="${OPENSCENE_DATA_ROOT}/sensor_blobs/test/test"

WITH_FM_DP="${WITH_FM_DP:-0}"
DEFAULT_DP_PREDS="${NAVSIM_EXP_ROOT}/train_dp_official_fm_joint/epoch=18-step=81776_navtest_fm.pkl"
if [ "${WITH_FM_DP}" = "1" ]; then
  export DP_PREDS="${DP_PREDS:-${DEFAULT_DP_PREDS}}"
  dp_stem="$(basename "${DP_PREDS}" .pkl)"
  dp_tag="${dp_stem//=/-}"
  exp_tag="drivesuprim-vov+fm-${dp_tag}"
else
  unset DP_PREDS || true
  exp_tag="drivesuprim-vov-standalone"
fi

experiment_name="${EXP_NAME:-train_drivesuprim_vov/${exp_tag}-b${batch_size}-w${workers}}"
SUBSCORE_PATH="${SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}/train_drivesuprim_vov/${exp_tag}_${split}.pkl}"

required_paths=(
  "${ckpt}"
  "${metric_cache_path}"
  "${OPENSCENE_DATA_ROOT}/navsim_logs/test"
  "${sensor_path}"
  "${NAVSIM_DEVKIT_ROOT}/traj_final/test_8192_kmeans.npy"
  "${OPENSCENE_DATA_ROOT}/models/dd3d_det_final.pth"
)
if [ "${WITH_FM_DP}" = "1" ]; then
  required_paths+=("${DP_PREDS}")
fi
for path in "${required_paths[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 1
  fi
done
if [ ! -s "${ckpt}" ]; then
  echo "[ERROR] Checkpoint is empty/invalid: ${ckpt}" >&2
  echo "        Download from: https://huggingface.co/alkaid-2000/DriveSuprim/tree/main/model_ckpt" >&2
  echo "        Place as: ${DEFAULT_CKPT}" >&2
  exit 1
fi

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

export SUBSCORE_PATH
export SKIP_INFER="${SKIP_INFER:-0}"

echo "[RUN] DriveSuprim VoV navtest evaluation"
echo "      ckpt=${ckpt}"
echo "      with_fm_dp=${WITH_FM_DP} dp_preds=${DP_PREDS:-<none>}"
echo "      split=${split} gpu=${gpu} bs=${batch_size} workers=${workers}"
echo "      refinement=${num_refinement_stage}/${stage_layers}/${topks} model=${inference_model}"
echo "      subscore=${SUBSCORE_PATH}"
echo "      experiment=${experiment_name}"

CUDA_VISIBLE_DEVICES="${gpu}" python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_gpu_ssl.py" \
  agent=drivesuprim_agent_vov \
  train_test_split="${split}" \
  dataloader.params.batch_size="${batch_size}" \
  dataloader.params.num_workers=0 \
  dataloader.params.pin_memory=false \
  '~dataloader.params.prefetch_factor' \
  "agent.checkpoint_path=\"${ckpt}\"" \
  agent.config.training=false \
  agent.config.only_ori_input=true \
  agent.config.inference.model="${inference_model}" \
  agent.config.inference.save_pickle=false \
  agent.config.refinement.use_multi_stage=true \
  agent.config.refinement.num_refinement_stage="${num_refinement_stage}" \
  agent.config.refinement.stage_layers="${stage_layers}" \
  agent.config.refinement.topks="${topks}" \
  ++trainer.params.accelerator=gpu \
  ++trainer.params.devices=1 \
  trainer.params.strategy=auto \
  trainer.params.precision=32 \
  worker.threads_per_node="${workers}" \
  worker.log_to_driver=false \
  experiment_name="${experiment_name}" \
  +cache_path=null \
  metric_cache_path="${metric_cache_path}" \
  original_sensor_path="${sensor_path}"
