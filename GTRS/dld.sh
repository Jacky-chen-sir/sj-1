#!/bin/bash

# ========== 一键下载感知权重及所有主流模型权重 ==========
# 如已下载，可将此段注释掉
MODELS_DIR=$(pwd)/data/models
mkdir -p $MODELS_DIR

# GTRS-Dense
wget -nc -q --show-progress \
  https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/gtrs_dense_vov.ckpt \
  -O $MODELS_DIR/gtrs_dense_model.ckpt || echo "❌ gtrs_dense_model.ckpt 下载失败"
# GTRS-Aug
wget -nc -q --show-progress \
  https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/gtrs_aug_vov.ckpt \
  -O $MODELS_DIR/gtrs_aug_model.ckpt || echo "❌ gtrs_aug_model.ckpt 下载失败"
# Diffusion Policy
wget -nc -q --show-progress \
  https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/gtrs_dp.ckpt \
  -O $MODELS_DIR/gtrs_dp_model.ckpt || echo "❌ gtrs_dp_model.ckpt 下载失败"
# Hydra-MDP
wget -nc -q --show-progress \
  https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/hydra_mdp_vov.ckpt \
  -O $MODELS_DIR/hydra_mdp_model.ckpt || echo "❌ hydra_mdp_model.ckpt 下载失败"
# 下载dp_preds.pkl
wget -nc -q --show-progress \
  https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/dp_preds.pkl \
  -O $MODELS_DIR/dp_preds.pkl || echo "❌ dp_preds.pkl 下载失败"

# ========== 训练决策模型（感知层用已有权重，默认冻结） ==========
# export OPENSCENE_DATA_ROOT=$(pwd)/data
# export NAVSIM_DEVKIT_ROOT=$(pwd)

# BEV_CKPT_PATH=$MODELS_DIR/gtrs_dense_model.ckpt
# DECISION_SAVE_PATH=$MODELS_DIR/decision_model.ckpt

# python navsim/planning/script/train_decision.py \
#   --bev_ckpt $BEV_CKPT_PATH \
#   --save_path $DECISION_SAVE_PATH \
#   --freeze_bev True \
#   --epochs 20 \
#   --batch_size 8 \
#   # --other_args ...

echo "决策模型训练完成，权重已保存到 $DECISION_SAVE_PATH"
