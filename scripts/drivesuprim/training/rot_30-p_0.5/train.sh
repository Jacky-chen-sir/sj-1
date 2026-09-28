#!/bin/bash
# Train GTRS-Aug with DriveSuprim paper recipe (soft labels + multi-stage + ego-rot SSL).
# Usage:
#   bash scripts/drivesuprim/training/rot_30-p_0.5/train.sh gtrs_aug_drivesuprim_vov 1 3 256
#   bash scripts/drivesuprim/training/rot_30-p_0.5/train.sh gtrs_aug_drivesuprim_r34 1 3 256
#   bash scripts/drivesuprim/training/rot_30-p_0.5/train.sh gtrs_aug_drivesuprim_vit 1 3 256
#
# 环境变量：
#   BS（默认 8）  NPROC（默认 8）  ACCUM（默认 0=不用累积）
#   LR（默认 7.5e-5 × 有效批量/64）  PRECISION  RESUME_CKPT  EXP_TAG  EXP_DIR  MAX_STEPS
# 同 iter 消融：BS=3 NPROC=3 ACCUM=4 EXP_DIR=ablation_same_iter_r34/official_recipe_r34

agent=$1
num_refinement_stage=$2
stage_layers=$3
topks=$4

allowed_agents=(
  "gtrs_aug_drivesuprim_r34"
  "gtrs_aug_drivesuprim_r50"
  "gtrs_aug_drivesuprim_vit"
  "gtrs_aug_drivesuprim_vov"
  # legacy DriveSuprim agent names still accepted → remapped below
  "drivesuprim_agent_r34"
  "drivesuprim_agent_r50"
  "drivesuprim_agent_vit"
  "drivesuprim_agent_vov"
)
valid_agent=false
for valid in "${allowed_agents[@]}"; do
  if [ "$agent" == "$valid" ]; then
    valid_agent=true
    break
  fi
done
if [ "$valid_agent" == false ]; then
  echo "Error: agent must be one of: ${allowed_agents[*]}"
  exit 1
fi

# Remap legacy DriveSuprim agent names onto gtrs_aug recipes
case "$agent" in
  drivesuprim_agent_r34) agent=gtrs_aug_drivesuprim_r34 ;;
  drivesuprim_agent_r50) agent=gtrs_aug_drivesuprim_r50 ;;
  drivesuprim_agent_vit) agent=gtrs_aug_drivesuprim_vit ;;
  drivesuprim_agent_vov) agent=gtrs_aug_drivesuprim_vov ;;
esac

if [ "$agent" == "gtrs_aug_drivesuprim_vit" ]; then
  epoch=6
else
  epoch=10
fi

echo "Using agent: $agent, setting epoch: $epoch"

bs=${BS:-8}
rot=30
probability=0.5
nproc=${NPROC:-8}
accum=${ACCUM:-0}
master_port=${MASTER_PORT:-29500}
exp_tag=${EXP_TAG:-}

# 有效批量 = BS × NPROC × max(ACCUM,1)。基线 LR 7.5e-5 对应 8×8=64。
if [ "$accum" -gt 0 ]; then
  eff_batch=$(( bs * nproc * accum ))
else
  eff_batch=$(( bs * nproc ))
fi
if [ -z "${LR:-}" ]; then
  lr=$(awk -v e="$eff_batch" 'BEGIN{printf "%.3e", 7.5e-5 * e / 64}')
else
  lr=$LR
fi

if [ -n "${EXP_DIR:-}" ]; then
  dir="$EXP_DIR"
else
  dir="training/$agent/rot_$rot-p_$probability/stage_layers_$stage_layers-topks_$topks${exp_tag:+-${exp_tag}}"
fi

# Prefer DriveSuprim-named offline json if present; else use existing equivalent.
offline_json="${NAVSIM_EXP_ROOT}/offline_files/training_ego_aug_files/rot_${rot}-p_${probability}-ensemble.json"
if [ ! -f "${offline_json}" ]; then
  offline_json="${NAVSIM_TRAJPDM_ROOT}/random_aug/rot_${rot}-trans_0-va_0-p_${probability}-ensemble.json"
fi

command_string="$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_aug.py \
    +debug=false \
    agent=$agent \
    experiment_name=$dir \
    split=trainval \
    train_test_split=navtrain \
    dataloader.params.batch_size=$bs \
    ~trainer.params.strategy \
    trainer.params.precision=${PRECISION:-32} \
    trainer.params.max_epochs=$epoch \
    trainer.params.limit_val_batches=0.1 \
    agent.config.ckpt_path=$dir \
    agent.lr=$lr \
    agent.config.student_rotation_ensemble=3 \
    agent.config.ego_perturb.n_student_rotation_ensemble=3 \
    agent.config.ego_perturb.offline_aug_angle_boundary=$rot \
    agent.config.ego_perturb.rotation.offline_aug_angle_boundary=$rot \
    agent.config.ego_perturb.offline_aug_file=$offline_json \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.refinement_approach=transformer_decoder \
    agent.config.refinement.num_refinement_stage=$num_refinement_stage \
    agent.config.refinement.stage_layers=$stage_layers \
    agent.config.refinement.topks=$topks \
    agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_navtrain_final/navtrain.pkl \
    agent.config.aug_vocab_pdm_score_dir=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_$rot-p_$probability-ensemble/vocab_score_8192_navtrain_final/split_pickles \
    cache_path=null
"

if [ "$accum" -gt 0 ]; then
  command_string="$command_string ++trainer.params.accumulate_grad_batches=$accum ++trainer.params.sync_batchnorm=true"
fi
if [ -n "${MAX_STEPS:-}" ]; then
  command_string="$command_string ++trainer.params.max_steps=${MAX_STEPS}"
fi

# Optional resume: RESUME_CKPT=/abs/path.ckpt bash ... (restores weights+optimizer+epoch)
if [ -n "${RESUME_CKPT:-}" ]; then
  command_string="$command_string +resume_ckpt_path='${RESUME_CKPT}'"
  echo "[RESUME] from ${RESUME_CKPT}"
fi

echo "=== DriveSuprim-recipe train ==="
echo "  agent    : $agent"
echo "  dir      : $dir"
echo "  effective: BS=$bs x NPROC=$nproc x ACCUM=${accum:-0} = $eff_batch  → LR=$lr"
echo "--- COMMAND ---"
echo $command_string
echo

torchrun --nproc_per_node=$nproc --master_port=$master_port $command_string
