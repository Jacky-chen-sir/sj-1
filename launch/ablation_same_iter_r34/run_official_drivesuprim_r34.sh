#!/usr/bin/env bash
# Official DriveSuprim R34 (DriveSuprim-main), hyperparameters locked to the OPD 3x3090 run.
# Fair pair: OPD R34 vs this job. Same backbone / LR / BS / ACCUM / 13000 steps.
# Eval later: vocab-only (unset DP_PREDS). Combined uses OUR FM-DP pickle for both sides.
# Usage: bash /home/ws/navsim_workspace/exp/ablation_same_iter_r34/run_official_drivesuprim_r34.sh
set -euo pipefail

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
export DDP_FIND_UNUSED=1

cd "$NAVSIM_DEVKIT_ROOT"

ABLATION=/home/ws/navsim_workspace/exp/ablation_same_iter_r34
LOG="${ABLATION}/train_official_drivesuprim_r34.log"
CKPT_DIR="${ABLATION}/official_drivesuprim_r34"
OFFLINE_JSON="${NAVSIM_TRAJPDM_ROOT}/random_aug/rot_30-trans_0-va_0-p_0.5-ensemble.json"

mkdir -p /mnt/bigdisk/tmp /mnt/bigdisk/tmp/ray "$CKPT_DIR" "$(dirname "$LOG")"

# Do not change BS/ACCUM/NPROC/LR. If OOM, STOP — do not silently drop BS.
# Resume if a prior step ckpt exists (prefer highest step < 13000).
RESUME_ARG=()
if [ -n "${RESUME_CKPT:-}" ] && [ -f "${RESUME_CKPT}" ]; then
  RESUME_ARG=(+resume_ckpt_path="'${RESUME_CKPT}'")
elif ls "${CKPT_DIR}"/step-step=*.ckpt >/dev/null 2>&1; then
  LATEST=$(ls -1 "${CKPT_DIR}"/step-step=*.ckpt | sort | tail -n 1)
  if [[ "${LATEST}" != *step=013000.ckpt ]]; then
    RESUME_ARG=(+resume_ckpt_path="'${LATEST}'")
    echo "Resuming from ${LATEST}"
  else
    echo "Already have step-step=013000.ckpt — not restarting. Exiting."
    exit 0
  fi
fi

{
  echo "[$(date '+%F %T')] start official_drivesuprim_r34 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}"
  echo "[$(date '+%F %T')] resume_args=${RESUME_ARG[*]:-none}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
  torchrun --nproc_per_node=3 --master_port="${MASTER_PORT:-29513}" \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ssl.py" \
    +debug=false \
    agent=drivesuprim_agent_r34 \
    experiment_name=ablation_same_iter_r34/official_drivesuprim_r34 \
    split=trainval \
    train_test_split=navtrain \
    dataloader.params.batch_size=3 \
    dataloader.params.num_workers=4 \
    '~trainer.params.strategy' \
    trainer.params.precision=32 \
    trainer.params.max_epochs=10 \
    trainer.params.limit_val_batches=0.1 \
    '++trainer.params.accumulate_grad_batches=4' \
    '++trainer.params.sync_batchnorm=true' \
    '++trainer.params.max_steps=13000' \
    agent.config.ckpt_path=ablation_same_iter_r34/official_drivesuprim_r34 \
    agent.lr=4.219e-05 \
    agent.config.ego_perturb.n_student_rotation_ensemble=3 \
    agent.config.ego_perturb.offline_aug_angle_boundary=30 \
    "agent.config.ego_perturb.offline_aug_file=${OFFLINE_JSON}" \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    agent.config.ori_vocab_pdm_score_full_path="${NAVSIM_TRAJPDM_ROOT}/ori/vocab_score_8192_navtrain_final/navtrain.pkl" \
    agent.config.aug_vocab_pdm_score_dir="${NAVSIM_TRAJPDM_ROOT}/random_aug/rot_30-p_0.5-ensemble/vocab_score_8192_navtrain_final/split_pickles" \
    cache_path=null \
    "${RESUME_ARG[@]}"
  echo "[$(date '+%F %T')] finished official_drivesuprim_r34"
} 2>&1 | tee -a "$LOG"

echo "log: $LOG"
echo "ckpt/TB: $CKPT_DIR  (tb symlink: ${ABLATION}/tb/official_drivesuprim)"
