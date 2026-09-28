#!/bin/bash
# Eval DriveSuprim-recipe GTRS-Aug checkpoint on navtest.
# Usage:
#   bash scripts/drivesuprim/evaluation/eval_epoch.sh \
#     5 training/gtrs_aug_drivesuprim_vov/rot_30-p_0.5/stage_layers_3-topks_256 \
#     1 3 256 gtrs_aug_drivesuprim_vov teacher

epoch=$1
dir=$2
num_refinement_stage=$3
stage_layers=$4
topks=$5
agent=${6:-"gtrs_aug_drivesuprim_vov"}
inference_model=${7:-"teacher"}

case "$agent" in
  drivesuprim_agent_r34) agent=gtrs_aug_drivesuprim_r34 ;;
  drivesuprim_agent_r50) agent=gtrs_aug_drivesuprim_r50 ;;
  drivesuprim_agent_vit) agent=gtrs_aug_drivesuprim_vit ;;
  drivesuprim_agent_vov) agent=gtrs_aug_drivesuprim_vov ;;
esac

padded_epoch=$(printf "%02d" $epoch)
step=$((($epoch + 1) * 1330))
metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/test/ori"
nproc=${NPROC:-8}
master_port=${MASTER_PORT:-29500}
bs=${BS:-8}

if [ "$inference_model" = "teacher" ]; then
  experiment_name="${dir}/test-${padded_epoch}ep"
else
  experiment_name="${dir}/test-${padded_epoch}ep-${inference_model}"
fi

command_string="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2_aug.py \
    +debug=false \
    agent=$agent \
    train_test_split=navtest \
    dataloader.params.batch_size=$bs \
    worker.threads_per_node=128 \
    agent.checkpoint_path='${NAVSIM_EXP_ROOT}/${dir}/epoch=${padded_epoch}-step=${step}.ckpt' \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=${inference_model} \
    agent.config.inference.save_pickle=false \
    agent.config.lab.save_pickle=false \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.refinement_approach=transformer_decoder \
    agent.config.refinement.num_refinement_stage=$num_refinement_stage \
    agent.config.refinement.stage_layers=$stage_layers \
    agent.config.refinement.topks=$topks \
    experiment_name=${experiment_name} \
    +cache_path=null \
    metric_cache_path=${metric_cache_path}
"

echo "--- COMMAND ---"
echo $command_string
echo

torchrun --nproc_per_node=$nproc --master_port=$master_port $command_string
