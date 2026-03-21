#!/bin/bash

# 1. 设置环境变量
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/GTRS"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/bigdisk/GTRS/download}"
export NAVSIM_TRAJPDM_ROOT="$HOME/navsim_workspace/dataset/traj_pdm_v2"

# 2. 选择评测模型和权重
agent=gtrs_dense_vov  # 可选 gtrs_dense_vov, gtrs_aug_vov, gtrs_diffusion_policy 等
split=navhard
experiment_dir=train_gtrs_dense  # 你的实验目录
epoch=20  # 你要评测的epoch
ckpt=${NAVSIM_EXP_ROOT}/${experiment_dir}/epoch${epoch}.ckpt

# 3. 设置评测缓存路径
metric_cache_path="${NAVSIM_EXP_ROOT}/${split}_two_stage_metric_cache"

# 4. 运行官方评测脚本
cd ${NAVSIM_DEVKIT_ROOT}

python navsim/planning/script/run_pdm_score_gpu_v2.py \
    agent=$agent \
    dataloader.params.batch_size=32 \
    agent.checkpoint_path=${ckpt} \
    trainer.params.precision=32 \
    experiment_name=${experiment_dir}/test-ep${epoch}-${split}-eval \
    +cache_path=null \
    metric_cache_path=${metric_cache_path} \
    train_test_split=${split}_two_stage

# 5. 评测结果会输出在指定 experiment_name 目录下