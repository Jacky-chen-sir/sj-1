#!/usr/bin/env bash
# Official GTRS-Dense + GTRS-Aug navtest eval (combined with official DP proposals).
# Assumes official DP proposals pickle already exists (from official DP eval).
#
# Usage:
#   bash eval_official_dense_aug_navtest.sh
#   GPU=2 WORKERS=8 bash eval_official_dense_aug_navtest.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NAVSIM_DEVKIT_ROOT="${SCRIPT_DIR}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT_OVERRIDE:-/home/ws/navsim_workspace/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT_OVERRIDE:-/home/ws/navsim_workspace/exp/official_paper_models_navtest}"
export NAVSIM_TRAJPDM_ROOT="${NAVSIM_TRAJPDM_ROOT_OVERRIDE:-${OPENSCENE_DATA_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PROGRESS_MODE="${PROGRESS_MODE:-eval}"
export HYDRA_FULL_ERROR=1
export TMPDIR="${TMPDIR:-/mnt/bigdisk/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/mnt/bigdisk/tmp/ray}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

GPU_ID="${GPU:-0}"
WORKERS="${WORKERS:-16}"
BATCH_SIZE="${BATCH_SIZE:-24}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/home/ws/navsim_workspace/exp/navtest_two_stage_metric_cache}"
ORIGINAL_SENSOR_PATH="${ORIGINAL_SENSOR_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test/test}"

DENSE_CKPT="${DENSE_CKPT:-${NAVSIM_DEVKIT_ROOT}/data/models/gtrs_dense_model.ckpt}"
AUG_CKPT="${AUG_CKPT:-${NAVSIM_DEVKIT_ROOT}/data/models/gtrs_aug_model.ckpt}"
VOV_CKPT="${VOV_CKPT:-${OPENSCENE_DATA_ROOT}/models/dd3d_det_final.pth}"
VOCAB_8192="${VOCAB_8192:-${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy}"

DP_PREDS_PATH="${DP_PREDS_PATH:-${NAVSIM_EXP_ROOT}/train_dp/official_dp_navtest.pkl}"
DENSE_SUBSCORE_PATH="${DENSE_SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}/train_gtrs_dense/official_dense_navtest.pkl}"
AUG_SUBSCORE_PATH="${AUG_SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}/train_gtrs_aug/official_aug_navtest.pkl}"

LOG_DIR="${NAVSIM_EXP_ROOT}/logs"
mkdir -p "${LOG_DIR}" "$(dirname "${DENSE_SUBSCORE_PATH}")" "$(dirname "${AUG_SUBSCORE_PATH}")"

require_path() {
  if [ ! -e "$1" ]; then
    echo "[ERROR] Missing $2: $1" >&2
    exit 1
  fi
}

require_path "${DP_PREDS_PATH}" "official DP proposals pickle"
require_path "${DENSE_CKPT}" "official Dense ckpt"
require_path "${AUG_CKPT}" "official Aug ckpt"
require_path "${VOV_CKPT}" "VoV backbone"
require_path "${VOCAB_8192}" "8192 vocab"
require_path "${METRIC_CACHE_PATH}" "navtest metric cache"
require_path "${ORIGINAL_SENSOR_PATH}" "navtest sensors"

cd "${NAVSIM_DEVKIT_ROOT}"
source /home/ws/anaconda3/etc/profile.d/conda.sh
conda activate conda_gtrs

export DP_PREDS="${DP_PREDS_PATH}"

echo "[RUN] Official Dense + DP_PREDS on GPU ${GPU_ID} (batch=${BATCH_SIZE}, workers=${WORKERS})"
export SUBSCORE_PATH="${DENSE_SUBSCORE_PATH}"
if [ -s "${DENSE_SUBSCORE_PATH}" ]; then
  export SKIP_INFER=1
  echo "[SKIP] Dense inference pickle exists: ${DENSE_SUBSCORE_PATH}"
else
  export SKIP_INFER=0
fi
python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py" \
  agent=gtrs_dense_vov \
  +combined_inference=true \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${WORKERS}" \
  dataloader.params.pin_memory=true \
  dataloader.params.prefetch_factor=2 \
  agent.checkpoint_path="${DENSE_CKPT}" \
  agent.config.vocab_path="${VOCAB_8192}" \
  agent.config.vov_ckpt="${VOV_CKPT}" \
  ++trainer.params.accelerator=gpu \
  ++trainer.params.devices=1 \
  trainer.params.strategy=auto \
  trainer.params.precision=32 \
  worker.threads_per_node="${WORKERS}" \
  worker.log_to_driver=false \
  experiment_name=official_paper_models_navtest/dense-navtest-b${BATCH_SIZE}-w${WORKERS} \
  +cache_path=null \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  original_sensor_path="${ORIGINAL_SENSOR_PATH}" \
  train_test_split=navtest \
  2>&1 | tee "${LOG_DIR}/official_dense_navtest.log"

echo "[RUN] Official Aug + DP_PREDS on GPU ${GPU_ID} (batch=${BATCH_SIZE}, workers=${WORKERS})"
export SUBSCORE_PATH="${AUG_SUBSCORE_PATH}"
if [ -s "${AUG_SUBSCORE_PATH}" ]; then
  export SKIP_INFER=1
  echo "[SKIP] Aug inference pickle exists: ${AUG_SUBSCORE_PATH}"
else
  export SKIP_INFER=0
fi
python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2_aug.py" \
  agent=gtrs_aug_vov \
  +combined_inference=true \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${WORKERS}" \
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
  worker.threads_per_node="${WORKERS}" \
  worker.log_to_driver=false \
  experiment_name=official_paper_models_navtest/aug-navtest-b${BATCH_SIZE}-w${WORKERS} \
  +cache_path=null \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  original_sensor_path="${ORIGINAL_SENSOR_PATH}" \
  train_test_split=navtest \
  2>&1 | tee "${LOG_DIR}/official_aug_navtest.log"

echo "[DONE] Dense log: ${LOG_DIR}/official_dense_navtest.log"
echo "[DONE] Aug log:   ${LOG_DIR}/official_aug_navtest.log"
echo "[DONE] Results under: ${NAVSIM_EXP_ROOT}/official_paper_models_navtest"
