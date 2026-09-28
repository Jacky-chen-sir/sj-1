#!/usr/bin/env bash
# Vocab-only navtest eval for official DriveSuprim R34 at 5000 / 9000 / 13000.
set -euo pipefail

ABLATION=/home/ws/navsim_workspace/exp/ablation_same_iter_r34
CKPT_DIR="${ABLATION}/official_drivesuprim_r34"
LOG="${ABLATION}/eval_official_drivesuprim_r34.log"
EVAL_GPU="${EVAL_GPU:-2}"
# Official DriveSuprim eval_file.sh uses batch_size=8. bs=2 + num_workers=0
# leaves the 3090 idle (~0.1 it/s). Override with EVAL_BS / EVAL_WORKERS.
EVAL_BS="${EVAL_BS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
# Ray PDMS threads. 8x3 parallel OOM/SIGBUS'd after predict; keep low.
WORKER_THREADS="${WORKER_THREADS:-2}"
if [ -n "${STEPS:-}" ]; then
  # e.g. STEPS="5000" or STEPS="5000 9000 13000"
  # shellcheck disable=SC2206
  TARGETS=(${STEPS})
else
  TARGETS=(5000 9000 13000)
fi

source /home/ws/anaconda3/etc/profile.d/conda.sh
conda activate conda_gtrs

export NAVSIM_DEVKIT_ROOT=/home/ws/navsim_workspace/DriveSuprim-main
export OPENSCENE_DATA_ROOT=/home/ws/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/ws/navsim_workspace/exp
export NAVSIM_TRAJPDM_ROOT=/home/ws/navsim_workspace/dataset/traj_pdm_v2
export NUPLAN_MAPS_ROOT=/home/ws/navsim_workspace/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH=/home/ws/navsim_workspace/DriveSuprim-main
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/mnt/bigdisk/tmp RAY_TMPDIR=/mnt/bigdisk/tmp/ray
export PROGRESS_MODE=eval
unset DP_PREDS || true

mkdir -p "${ABLATION}/evals/official_drivesuprim_r34" "${TMPDIR}" "${RAY_TMPDIR}"

log() { echo "[$(date '+%F %T')] $*"; }

wait_stable() {
  local f="$1" prev="" now=""
  while true; do
    if [ -f "$f" ]; then
      now=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
      if [ "$now" -gt 0 ] && [ "$now" = "$prev" ]; then
        sleep 10
        now=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
        if [ "$now" = "$prev" ]; then return 0; fi
      fi
      prev=$now
    fi
    sleep 60
  done
}

run_eval() {
  local step="$1"
  local padded
  padded=$(printf '%06d' "$step")
  local ckpt="${CKPT_DIR}/step-step=${padded}.ckpt"
  local dest="${ABLATION}/evals/official_drivesuprim_r34/step${step}.csv"
  if [ -f "$dest" ]; then
    log "skip step=${step}, ${dest} exists"
    return 0
  fi
  if [ ! -f "$ckpt" ]; then
    log "waiting for ${ckpt}"
    wait_stable "$ckpt"
  fi
  local exp_name="eval_official_drivesuprim_r34_step${step}_navtest"
  local elog="${ABLATION}/evals/official_drivesuprim_r34/step${step}.log"
  log "EVAL start step=${step} gpu=${EVAL_GPU} bs=${EVAL_BS} workers=${EVAL_WORKERS} ray_threads=${WORKER_THREADS} ckpt=${ckpt}"
  cd "${NAVSIM_DEVKIT_ROOT}"
  # save_pickle=true: if Ray scoring dies after predict, can SKIP_INFER later
  env CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
    torchrun --nproc_per_node=1 --master_port=$((29600 + step / 1000)) \
    navsim/planning/script/run_pdm_score_one_stage_gpu_ssl.py \
    +debug=false \
    +use_pdm_closed=false \
    agent=drivesuprim_agent_r34 \
    train_test_split=navtest \
    dataloader.params.batch_size="${EVAL_BS}" \
    dataloader.params.num_workers="${EVAL_WORKERS}" \
    dataloader.params.pin_memory=true \
    dataloader.params.prefetch_factor=2 \
    "agent.checkpoint_path='${ckpt}'" \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=teacher \
    agent.config.inference.save_pickle=true \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    worker.threads_per_node="${WORKER_THREADS}" \
    worker.log_to_driver=false \
    experiment_name="${exp_name}" \
    +cache_path=null \
    metric_cache_path=/home/ws/navsim_workspace/exp/navtest_two_stage_metric_cache \
    original_sensor_path=/home/ws/navsim_workspace/dataset/sensor_blobs/test/test \
    > "${elog}" 2>&1
  local csv
  csv=$(find "${NAVSIM_EXP_ROOT}/${exp_name}" -name '*.csv' -type f 2>/dev/null | sort | tail -n 1 || true)
  if [ -z "$csv" ]; then
    log "EVAL FAIL step=${step} no csv, see ${elog}"
    return 1
  fi
  cp -a "$csv" "$dest"
  python "${ABLATION}/update_scores.py" official_drivesuprim_r34 "$step" "$dest"
  log "EVAL done step=${step} csv=${dest}"
}

{
  log "queue ${TARGETS[*]} gpu=${EVAL_GPU} bs=${EVAL_BS} workers=${EVAL_WORKERS} ray_threads=${WORKER_THREADS}"
  for step in "${TARGETS[@]}"; do
    run_eval "$step" || log "continue after fail step=${step}"
  done
  log "queue finished"
} >>"${LOG}" 2>&1
