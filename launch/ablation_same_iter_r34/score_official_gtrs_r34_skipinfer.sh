#!/usr/bin/env bash
# Memory-safe SKIP_INFER PDMS for official_gtrs_r34 using GTRS_official
# (correct Ray model_trajectory merge + one-stage fallback like OPD).
# Does NOT touch DriveSuprim training. CPU/Ray only; low worker parallelism.
set -euo pipefail

ABLATION=/home/ws/navsim_workspace/exp/ablation_same_iter_r34
EVAL_DIR="${ABLATION}/evals/official_gtrs_r34"
CKPT_DIR="${ABLATION}/official_gtrs_r34"
LOG="${ABLATION}/score_official_gtrs_r34_skipinfer.log"
METRIC_CACHE=/home/ws/navsim_workspace/exp/navtest_two_stage_metric_cache
SENSOR=/home/ws/navsim_workspace/dataset/sensor_blobs/test/test
# Low parallelism under DriveSuprim RAM pressure (override: WORKER_THREADS=4)
WORKER_THREADS="${WORKER_THREADS:-2}"
TARGETS=(${TARGETS:-5000 9000 13000})

source /home/ws/anaconda3/etc/profile.d/conda.sh
conda activate conda_gtrs

export NAVSIM_DEVKIT_ROOT=/home/ws/navsim_workspace/GTRS_official
export OPENSCENE_DATA_ROOT=/home/ws/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/ws/navsim_workspace/exp
export NAVSIM_TRAJPDM_ROOT=/home/ws/navsim_workspace/dataset/traj_pdm_v2
export NUPLAN_MAPS_ROOT=/home/ws/navsim_workspace/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH=/home/ws/navsim_workspace/GTRS_official
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export TMPDIR=/mnt/bigdisk/tmp
export PROGRESS_MODE=eval
unset DP_PREDS || true
# Avoid touching GPUs used by DriveSuprim torchrun :29513
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${EVAL_DIR}" "${TMPDIR}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

finalize_one() {
  local step="$1"
  local dest="${EVAL_DIR}/step${step}.csv"
  local exp_name="eval_official_gtrs_r34_step${step}_navtest_score"
  local csv
  csv=$(find "${NAVSIM_EXP_ROOT}/${exp_name}" -name '*.csv' -type f 2>/dev/null | sort | tail -n 1 || true)
  if [ -z "$csv" ]; then
    log "FAIL step=${step}: no csv under ${exp_name}"
    return 1
  fi
  cp -a "$csv" "$dest"
  python "${ABLATION}/update_scores.py" official_gtrs_r34 "$step" "$dest"
  python3 - <<PY
import pandas as pd
df=pd.read_csv("${dest}")
n=int((df["valid"]==True).sum()) if "valid" in df.columns else len(df)
print(f"VERIFY step=${step} n_valid={n} rows={len(df)}")
if n < 11000:
    raise SystemExit(f"n_valid={n} too low (expect ~11992)")
PY
  log "DONE step=${step} csv=${dest}"
}

score_one() {
  local step="$1"
  local sub="${EVAL_DIR}/step${step}_subscores.pkl"
  local dest="${EVAL_DIR}/step${step}.csv"
  local elog="${EVAL_DIR}/step${step}_skipinfer_official.log"
  local exp_name="eval_official_gtrs_r34_step${step}_navtest_score"
  if [ -f "$dest" ]; then
    # Re-verify existing
    local nv
    nv=$(python3 -c "import pandas as pd; df=pd.read_csv('${dest}'); print(int((df.valid==True).sum()) if 'valid' in df.columns else 0)")
    if [ "$nv" -ge 11000 ]; then
      log "skip step=${step}, valid csv exists n_valid=${nv}"
      return 0
    fi
    log "existing csv n_valid=${nv} too low; re-scoring"
    mv -v "$dest" "${dest}.bad_nvalid${nv}_$(date +%Y%m%d_%H%M%S)" | tee -a "${LOG}"
  fi
  if [ ! -f "$sub" ]; then
    log "FAIL step=${step}: missing ${sub}"
    return 1
  fi
  # AF_UNIX path limit is 107 bytes; keep RAY_TMPDIR very short.
  export RAY_TMPDIR="/mnt/bigdisk/tmp/rg${step}"
  mkdir -p "${RAY_TMPDIR}"
  export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
  # Under DriveSuprim RAM pressure Ray's default 0.95 threshold kills healthy
  # workers (~1GB each). Disable the monitor; rely on OS/swap instead.
  export RAY_memory_monitor_refresh_ms=0
  export RAY_memory_usage_threshold=0.99

  log "SKIP_INFER start step=${step} worker.threads_per_node=${WORKER_THREADS}"
  cd "${NAVSIM_DEVKIT_ROOT}"
  # Agent config only needed for Hydra; SKIP_INFER skips model load.
  set +e
  nice -n 19 env SKIP_INFER=1 SUBSCORE_PATH="${sub}" \
    python navsim/planning/script/run_pdm_score_gpu_v2_aug.py \
    agent=gtrs_aug_opd_r34 train_test_split=navtest \
    dataloader.params.batch_size=1 \
    dataloader.params.num_workers=0 dataloader.params.pin_memory=false \
    '~dataloader.params.prefetch_factor' \
    "agent.checkpoint_path=\"${CKPT_DIR}/step-step=$(printf '%06d' "$step").ckpt\"" \
    agent.config.training=false agent.config.only_ori_input=true \
    agent.config.inference.model=student \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    ++trainer.params.accelerator=cpu ++trainer.params.devices=1 \
    trainer.params.strategy=auto trainer.params.precision=32 \
    "worker.threads_per_node=${WORKER_THREADS}" worker.log_to_driver=false \
    experiment_name="${exp_name}" +cache_path=null \
    metric_cache_path="${METRIC_CACHE}" \
    original_sensor_path="${SENSOR}" \
    > "${elog}" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    log "FAIL step=${step}: scoring exited rc=${rc}, see ${elog}"
    return "$rc"
  fi
  finalize_one "$step"
}

{
  log "queue TARGETS=${TARGETS[*]} WORKER_THREADS=${WORKER_THREADS}"
  for step in "${TARGETS[@]}"; do
    score_one "$step" || log "continue after fail step=${step}"
    # Drop Ray caches between steps
    rm -rf "/mnt/bigdisk/tmp/rg${step}" 2>/dev/null || true
  done
  log "queue finished"
  echo "==== scores.csv (official_gtrs_r34) ===="
  python3 - <<'PY'
import pandas as pd
df=pd.read_csv("/home/ws/navsim_workspace/exp/ablation_same_iter_r34/scores.csv")
print(df[df.method.isin(["opd","official_gtrs_r34"])][
  ["method","step","n_valid","epdms","nc","dac","ep","ttc"]
].to_string(index=False))
PY
} 2>&1 | tee -a "${LOG}"
