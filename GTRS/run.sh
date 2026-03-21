export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/GTRS"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/bigdisk/GTRS/download}"
export NAVSIM_TRAJPDM_ROOT="$HOME/navsim_workspace/dataset/traj_pdm_v2"

NUM_NODES=1
MASTER_ADDR=127.0.0.1
NODE_RANK=0
config="competition_training"
experiment_name=train_dp
agent=gtrs_diffusion_policy
lr=0.0002
bs=6
max_epochs=17

CACHE_DIR="${CACHE_DIR:-/mnt/bigdisk/cache_GTRS}"
CACHE_PATH_DEFAULT="/mnt/bigdisk/training_cache_trainval_backview"
CACHE_PATH="${CACHE_PATH:-$CACHE_PATH_DEFAULT}"
# Dense (agent) checkpoint
BEV_CKPT_PATH="$NAVSIM_DEVKIT_ROOT/path/gtrs_dense_vov.ckpt"
# Perception (VOV/DD3D) backbone checkpoint
VOV_CKPT_PATH="$NAVSIM_DEVKIT_ROOT/path/dd3d_det_final.pth"

CUDA_VISIBLE_DEVICES=0,1 \
MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_dense.py \
        --config-name ${config} \
        trainer.params.num_nodes=${NUM_NODES} \
        agent=${agent} \
        experiment_name=${experiment_name} \
        train_test_split=navtrain \
        dataloader.params.batch_size=${bs} \
        ~trainer.params.strategy \
        trainer.params.max_epochs=${max_epochs} \
        trainer.params.precision=32 \
        agent.config.ckpt_path="${BEV_CKPT_PATH}" \
        agent.config.vov_ckpt="${VOV_CKPT_PATH}" \
        +agent.config.bev_loss_weight=0.0 \
        agent.lr=${lr} \
        cache_path="${CACHE_PATH}" \
        force_cache_computation=false \
        use_cache_without_dataset=true
