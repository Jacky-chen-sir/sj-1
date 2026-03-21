NUM_NODES=1
MASTER_ADDR=MASTER_NODE_IP # your master node ip
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes
config="competition_training" # this config uses the entire navtrain dataset for training
experiment_name=train_dp
agent=gtrs_diffusion_policy

# training hyper-parameters
lr=0.0002
bs=6
max_epochs=17

WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
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
            cache_path=/mnt/bigdisk/cache_GTRS