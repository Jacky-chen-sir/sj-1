#!/usr/bin/env bash
# Vocab-only navtest eval for official GTRS R34 at 5000 / 9000 / 13000 only.
# GPU0 leftover, BS=1, nice 19. Wait for missing ckpts; do not eval other steps.
set -euo pipefail

ABLATION=/home/ws/navsim_workspace/exp/ablation_same_iter_r34
CKPT_DIR="${ABLATION}/official_gtrs_r34"
LOG="${ABLATION}/eval_official_gtrs_r34.log"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_BS="${EVAL_BS:-1}"
TARGETS=(5000 9000 13000)

source /home/ws/anaconda3/etc/profile.d/conda.sh
conda activate conda_gtrs

# Inference uses GTRSori (gtrs_aug_r34 ckpts). For SKIP_INFER rescoring of
# existing pickles, prefer score_official_gtrs_r34_skipinfer.sh (GTRS_official).
export NAVSIM_DEVKIT_ROOT=/home/ws/navsim_workspace/GTRSori
export OPENSCENE_DATA_ROOT=/home/ws/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/ws/navsim_workspace/exp
export NAVSIM_TRAJPDM_ROOT=/home/ws/navsim_workspace/dataset/traj_pdm_v2
export NUPLAN_MAPS_ROOT=/home/ws/navsim_workspace/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH=/home/ws/navsim_workspace/GTRSori
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/mnt/bigdisk/tmp
# Keep RAY_TMPDIR short (AF_UNIX path <= 107 bytes).
export RAY_TMPDIR=/mnt/bigdisk/tmp/rg
export PROGRESS_MODE=eval
unset DP_PREDS || true

mkdir -p "${ABLATION}/evals/official_gtrs_r34" "${TMPDIR}" "${RAY_TMPDIR}"

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
  local dest="${ABLATION}/evals/official_gtrs_r34/step${step}.csv"
  if [ -f "$dest" ]; then
    log "skip step=${step}, ${dest} exists"
    return 0
  fi
  if [ ! -f "$ckpt" ]; then
    log "waiting for ${ckpt}"
    wait_stable "$ckpt"
  fi
  local exp_name="eval_official_gtrs_r34_step${step}_navtest"
  local elog="${ABLATION}/evals/official_gtrs_r34/step${step}.log"
  local sub="${ABLATION}/evals/official_gtrs_r34/step${step}_subscores.pkl"
  log "EVAL start step=${step} gpu=${EVAL_GPU} bs=${EVAL_BS} ckpt=${ckpt}"
  cd "${NAVSIM_DEVKIT_ROOT}"
  nice -n 19 env   CUDA_VISIBLE_DEVICES="${EVAL_GPU}" SUBSCORE_PATH="${sub}" \
    python navsim/planning/script/run_pdm_score_gpu_v2_aug.py \
    agent=gtrs_aug_r34 train_test_split=navtest \
    dataloader.params.batch_size="${EVAL_BS}" \
    dataloader.params.num_workers=0 dataloader.params.pin_memory=false \
    '~dataloader.params.prefetch_factor' \
    "agent.checkpoint_path=\"${ckpt}\"" \
    agent.config.training=false agent.config.only_ori_input=true \
    agent.config.inference.model=teacher \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    ++trainer.params.accelerator=gpu ++trainer.params.devices=1 \
    trainer.params.strategy=auto trainer.params.precision=32 \
    worker.threads_per_node="${WORKER_THREADS:-4}" worker.log_to_driver=false \
    experiment_name="${exp_name}" +cache_path=null \
    metric_cache_path=/home/ws/navsim_workspace/exp/navtest_two_stage_metric_cache \
    original_sensor_path=/home/ws/navsim_workspace/dataset/sensor_blobs/test/test \
    > "${elog}" 2>&1
  local csv
  csv=$(find "${NAVSIM_EXP_ROOT}/${exp_name}" -name '*.csv' -type f | sort | tail -n 1 || true)
  if [ -z "$csv" ]; then
    log "EVAL FAIL step=${step} no csv, see ${elog}"
    return 1
  fi
  cp -a "$csv" "$dest"
  python "${ABLATION}/update_scores.py" official_gtrs_r34 "$step" "$dest"
  log "EVAL done step=${step} csv=${dest}"
}

{
  log "queue ${TARGETS[*]} gpu=${EVAL_GPU} bs=${EVAL_BS}"
  for step in "${TARGETS[@]}"; do
    run_eval "$step" || log "continue after fail step=${step}"
  done
  log "queue finished"
} >>"${LOG}" 2>&1
