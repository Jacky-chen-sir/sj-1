#!/bin/bash
# Resume DriveSuprim-recipe GTRS-Aug training.
# Usage:
#   bash scripts/drivesuprim/training/rot_30-p_0.5/train-resume.sh gtrs_aug_drivesuprim_vov 1 3 256 2

agent=$1
num_refinement_stage=$2
stage_layers=$3
topks=$4
ckpt_epoch=$5

allowed_agents=(
  "gtrs_aug_drivesuprim_r34"
  "gtrs_aug_drivesuprim_r50"
  "gtrs_aug_drivesuprim_vit"
  "gtrs_aug_drivesuprim_vov"
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

case "$agent" in
  drivesuprim_agent_r34) agent=gtrs_aug_drivesuprim_r34 ;;
  drivesuprim_agent_r50) agent=gtrs_aug_drivesuprim_r50 ;;
  drivesuprim_agent_vit) agent=gtrs_aug_drivesuprim_vit ;;
  drivesuprim_agent_vov) agent=gtrs_aug_drivesuprim_vov ;;
esac

if [ "$agent" == "gtrs_aug_drivesuprim_vit" ]; then
  epoch=8
else
  epoch=10
fi

bs=${BS:-8}
lr=${LR:-0.000075}
rot=30
probability=0.5
nproc=${NPROC:-8}
master_port=${MASTER_PORT:-29500}

dir=training/$agent/rot_$rot-p_$probability/stage_layers_$stage_layers-topks_$topks
padded_ckpt_epoch=$(printf "%02d" $ckpt_epoch)
step=$((($ckpt_epoch + 1) * 1330))
resume="epoch=${padded_ckpt_epoch}-step=${step}.ckpt"

if [ -z "$ckpt_epoch" ]; then
  echo "Wrong! Provide resume epoch as 5th arg."
  exit 1
fi

echo "Using agent: $agent, setting epoch: $epoch, start from epoch: $ckpt_epoch"

offline_json="${NAVSIM_EXP_ROOT}/offline_files/training_ego_aug_files/rot_${rot}-p_${probability}-ensemble.json"
if [ ! -f "${offline_json}" ]; then
  offline_json="${NAVSIM_TRAJPDM_ROOT}/random_aug/rot_${rot}-trans_0-va_0-p_${probability}-ensemble.json"
fi

command_string="$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_aug.py \
    +debug=false \
    agent=$agent \
    experiment_name=$dir \
    +resume_ckpt_path='$NAVSIM_EXP_ROOT/$dir/$resume' \
    split=trainval \
    train_test_split=navtrain \
    dataloader.params.batch_size=$bs \
    ~trainer.params.strategy \
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

echo "--- COMMAND ---"
echo $command_string
echo

torchrun --nproc_per_node=$nproc --master_port=$master_port $command_string
