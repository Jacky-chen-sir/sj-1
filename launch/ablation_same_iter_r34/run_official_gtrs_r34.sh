#!/usr/bin/env bash
# Official GTRS-Aug (GTRSori) with R34 backbone, same init as OPD:
#   timm.create_model('resnet34', pretrained=False, features_only=True)
# Hyperparameters locked to the OPD 3x3090 run. No VoV dd3d load.
# Usage: bash /home/ws/navsim_workspace/exp/ablation_same_iter_r34/run_official_gtrs_r34.sh
set -euo pipefail

source /home/ws/anaconda3/etc/profile.d/conda.sh
conda activate conda_gtrs

export NAVSIM_DEVKIT_ROOT=/home/ws/navsim_workspace/GTRSori
export OPENSCENE_DATA_ROOT=/home/ws/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/ws/navsim_workspace/exp
export NAVSIM_TRAJPDM_ROOT=/home/ws/navsim_workspace/dataset/traj_pdm_v2
export NUPLAN_MAPS_ROOT=/home/ws/navsim_workspace/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH=/home/ws/navsim_workspace/GTRSori
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/mnt/bigdisk/tmp RAY_TMPDIR=/mnt/bigdisk/tmp/ray
export DDP_FIND_UNUSED=1

cd "$NAVSIM_DEVKIT_ROOT"
mkdir -p /mnt/bigdisk/tmp /mnt/bigdisk/tmp/ray

# Do not change BS/ACCUM/NPROC/LR. If OOM, STOP — do not silently drop BS.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
torchrun --nproc_per_node=3 --master_port=29512 \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_aug.py" \
  --config-name competition_training \
  +debug=false \
  agent=gtrs_aug_r34 \
  experiment_name=ablation_same_iter_r34/official_gtrs_r34 \
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
  agent.config.ckpt_path=ablation_same_iter_r34/official_gtrs_r34 \
  agent.lr=4.219e-05 \
  cache_path=null
