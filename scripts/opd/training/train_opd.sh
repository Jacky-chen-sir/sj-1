#!/bin/bash
# OPD 蒸馏训练（论文 §4）：R34 学生 + 离线冻结 ViT-L 教师。
#
# 位置参数（与 drivesuprim 系列一致）：
#   bash scripts/opd/training/train_opd.sh <agent> <num_refinement_stage> <stage_layers> <topks>
#   bash scripts/opd/training/train_opd.sh gtrs_aug_opd_r34 1 3 256
#
# 环境变量（OPD_* 前缀统一映射到 ++agent.config.opd.*；留空则用 yaml 默认）：
#   OPD_TEACHER_SCORE_DIR   教师缓存目录（必填，否则 yaml 默认路径）
#   OPD_TAU_IMI / OPD_TAU_HEAD / OPD_TAU_LIST   三个温度（默认 2.0 每个）
#   OPD_LAMBDA_IMI / _HEAD / _REFINE / _RECALL   四项权重（默认 1.0/1.0/1.0/0.5）
#   OPD_TOPK_REFINE / OPD_TOPK_RECALL            两处 Top-K（默认 256 / 32）
#   OPD_ON_POLICY_ROUNDS / OPD_ON_POLICY_WEIGHT  on-policy（默认 0）
#   OPD_STORE_HEADS                               教师缓存是否含 8 头 logits（须与 cache_teacher.sh 一致）
#   OPD_ON_POLICY_VIEW_IDX                        on-policy 缓存配对的学生视图下标（默认 1）
#   OPD_TEACHER_MODE                              offline | ema | none（默认 offline）
#   OPD_EMA_EVAL / OPD_EMA_SOFT_LABEL             保留 EMA 副本并评测它 / 叠加 EMA 软标签（默认 true/true）
#   OPD_EMA_HARDCOPY_EPOCHS                       EMA 前几个 epoch m=0 硬拷贝（默认 1；原配方 3）
#   OPD_LAMBDA_DECAY / OPD_LAMBDA_FINAL_RATIO     蒸馏总权重调度 none|cosine、终值比例（默认 cosine / 0.3）
#
# 通用旋钮：
#   BS（每卡 batch，默认 3）  NPROC（默认 3）  ACCUM（梯度累积，默认 4）
#   LR（默认自动 = 7.5e-5 × 有效批量/64；显式传则用之）  PRECISION（默认 32）
#   MAX_EPOCHS（默认 10）  RESUME_CKPT（断点续训）  EXP_TAG（追加到目录名）
set -euo pipefail

agent=$1
num_refinement_stage=$2
stage_layers=$3
topks=$4

allowed_agents=( "gtrs_aug_opd_r34" )
valid_agent=false
for valid in "${allowed_agents[@]}"; do
  if [ "$agent" == "$valid" ]; then valid_agent=true; break; fi
done
if [ "$valid_agent" == false ]; then
  echo "Error: agent must be one of: ${allowed_agents[*]}"
  exit 1
fi

bs=${BS:-3}
nproc=${NPROC:-3}
accum=${ACCUM:-4}
max_epochs=${MAX_EPOCHS:-10}
precision=${PRECISION:-32}
master_port=${MASTER_PORT:-29510}

# 有效批量 = BS × NPROC × ACCUM；基线 LR 7.5e-5 对应 8×8=64。线性缩放。
eff_batch=$(( bs * nproc * accum ))
if [ -z "${LR:-}" ]; then
  lr=$(awk -v e="$eff_batch" 'BEGIN{printf "%.3e", 7.5e-5 * e / 64}')
else
  lr=$LR
fi

rot=30
probability=0.5
exp_tag=${EXP_TAG:-}
dir="training/opd/${agent}/rot_${rot}-p_${probability}/stage_layers_${stage_layers}-topks_${topks}${exp_tag:+-${exp_tag}}"

offline_json="${NAVSIM_EXP_ROOT}/offline_files/training_ego_aug_files/rot_${rot}-p_${probability}-ensemble.json"
if [ ! -f "${offline_json}" ]; then
  offline_json="${NAVSIM_TRAJPDM_ROOT}/random_aug/rot_${rot}-trans_0-va_0-p_${probability}-ensemble.json"
fi

# OPD_* 环境变量 → ++agent.config.opd.*
opd_over=()
if [ -n "${OPD_TEACHER_SCORE_DIR:-}" ];        then opd_over+=( "++agent.config.opd.teacher_score_dir=${OPD_TEACHER_SCORE_DIR}" ); fi
if [ -n "${OPD_TEACHER_ONPOLICY_SCORE_DIR:-}" ];then opd_over+=( "++agent.config.opd.teacher_onpolicy_score_dir=${OPD_TEACHER_ONPOLICY_SCORE_DIR}" ); fi
if [ -n "${OPD_TAU_IMI:-}" ];                  then opd_over+=( "++agent.config.opd.tau_imi=${OPD_TAU_IMI}" ); fi
if [ -n "${OPD_TAU_HEAD:-}" ];                 then opd_over+=( "++agent.config.opd.tau_head=${OPD_TAU_HEAD}" ); fi
if [ -n "${OPD_TAU_LIST:-}" ];                 then opd_over+=( "++agent.config.opd.tau_list=${OPD_TAU_LIST}" ); fi
if [ -n "${OPD_LAMBDA_IMI:-}" ];               then opd_over+=( "++agent.config.opd.lambda_imi=${OPD_LAMBDA_IMI}" ); fi
if [ -n "${OPD_LAMBDA_HEAD:-}" ];              then opd_over+=( "++agent.config.opd.lambda_head=${OPD_LAMBDA_HEAD}" ); fi
if [ -n "${OPD_LAMBDA_REFINE:-}" ];            then opd_over+=( "++agent.config.opd.lambda_refine=${OPD_LAMBDA_REFINE}" ); fi
if [ -n "${OPD_LAMBDA_RECALL:-}" ];            then opd_over+=( "++agent.config.opd.lambda_recall=${OPD_LAMBDA_RECALL}" ); fi
# 创新点 2 乘积一致性项（式 4-23）与创新点 1 双流评分头的开关；默认值在 agent yaml 里，
# 这里只在显式设了环境变量时覆盖。
if [ -n "${OPD_LAMBDA_PROD:-}" ];              then opd_over+=( "++agent.config.opd.lambda_prod=${OPD_LAMBDA_PROD}" ); fi
if [ -n "${OPD_HEAD_SCOPE:-}" ];               then opd_over+=( "++agent.config.opd.head_scope=${OPD_HEAD_SCOPE}" ); fi
if [ -n "${OPD_DUAL_STREAM_SCORE:-}" ];        then opd_over+=( "++agent.config.opd.dual_stream_score=${OPD_DUAL_STREAM_SCORE}" ); fi
if [ -n "${OPD_BETA_IMI:-}" ];                 then opd_over+=( "++agent.config.opd.beta_imi=${OPD_BETA_IMI}" ); fi
if [ -n "${OPD_TOPK_REFINE:-}" ];              then opd_over+=( "++agent.config.opd.topk_refine=${OPD_TOPK_REFINE}" ); fi
if [ -n "${OPD_TOPK_RECALL:-}" ];              then opd_over+=( "++agent.config.opd.topk_recall=${OPD_TOPK_RECALL}" ); fi
if [ -n "${OPD_ON_POLICY_ROUNDS:-}" ];         then opd_over+=( "++agent.config.opd.on_policy_rounds=${OPD_ON_POLICY_ROUNDS}" ); fi
if [ -n "${OPD_ON_POLICY_WEIGHT:-}" ];         then opd_over+=( "++agent.config.opd.on_policy_weight=${OPD_ON_POLICY_WEIGHT}" ); fi
if [ -n "${OPD_STORE_HEADS:-}" ];              then opd_over+=( "++agent.config.opd.store_heads=${OPD_STORE_HEADS}" ); fi
if [ -n "${OPD_ON_POLICY_VIEW_IDX:-}" ];       then opd_over+=( "++agent.config.opd.on_policy_view_idx=${OPD_ON_POLICY_VIEW_IDX}" ); fi
if [ -n "${OPD_TEACHER_MODE:-}" ];             then opd_over+=( "++agent.config.opd.teacher_mode=${OPD_TEACHER_MODE}" ); fi
if [ -n "${OPD_EMA_EVAL:-}" ];                 then opd_over+=( "++agent.config.opd.ema_eval=${OPD_EMA_EVAL}" ); fi
if [ -n "${OPD_EMA_SOFT_LABEL:-}" ];           then opd_over+=( "++agent.config.opd.ema_soft_label=${OPD_EMA_SOFT_LABEL}" ); fi
if [ -n "${OPD_EMA_HARDCOPY_EPOCHS:-}" ];      then opd_over+=( "++agent.config.opd.ema_hardcopy_epochs=${OPD_EMA_HARDCOPY_EPOCHS}" ); fi
if [ -n "${OPD_LAMBDA_DECAY:-}" ];             then opd_over+=( "++agent.config.opd.lambda_decay=${OPD_LAMBDA_DECAY}" ); fi
if [ -n "${OPD_LAMBDA_FINAL_RATIO:-}" ];       then opd_over+=( "++agent.config.opd.lambda_final_ratio=${OPD_LAMBDA_FINAL_RATIO}" ); fi

command_string="$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_aug.py \
    +debug=false \
    agent=$agent \
    experiment_name=$dir \
    split=trainval \
    train_test_split=navtrain \
    dataloader.params.batch_size=$bs \
    ~trainer.params.strategy \
    trainer.params.precision=$precision \
    trainer.params.max_epochs=$max_epochs \
    trainer.params.limit_val_batches=0.1 \
    ++trainer.params.accumulate_grad_batches=$accum \
    ++trainer.params.sync_batchnorm=true \
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

# resume
if [ -n "${RESUME_CKPT:-}" ]; then
  command_string="$command_string +resume_ckpt_path='${RESUME_CKPT}'"
fi
# opd overrides（用 + 前提：yaml 含 opd: 块；此处都含）
if [ "${#opd_over[@]}" -gt 0 ]; then
  command_string="$command_string ${opd_over[*]}"
fi

echo "=== OPD train ==="
echo "  agent    : $agent"
echo "  dir      : $dir"
echo "  effective: BS=$bs x NPROC=$nproc x ACCUM=$accum = $eff_batch  → LR=$lr"
echo "  opd      : ${opd_over[*]:-<yaml defaults, no overrides>}"
echo "--- COMMAND ---"
echo "$command_string"
echo

torchrun --nproc_per_node=$nproc --master_port=$master_port $command_string
