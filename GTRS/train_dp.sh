export HYDRA_FULL_ERROR=1

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/GTRS"

# Dataset root used by configs (default_dataset_paths.yaml uses OPENSCENE_DATA_ROOT).
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/bigdisk/GTRS/download}"
export NAVSIM_TRAJPDM_ROOT="$HOME/navsim_workspace/dataset/traj_pdm_v2"

NUM_NODES=3
MASTER_ADDR=MASTER_NODE_IP # your master node ip, which can be set to 127.0.0.1 for single-node training 
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes
config="competition_training" # this config uses the entire navtrain dataset for training
experiment_name=train_gtrs_dense # this could also be train_hydra_mdp
agent=gtrs_dense_vov # the agent could also be hydra_mdp_vov

# training hyper-parameters
lr=0.0002
bs=22
max_epochs=20

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
            agent.config.ckpt_path=${experiment_name} \
            agent.lr=${lr} \
            cache_path=null