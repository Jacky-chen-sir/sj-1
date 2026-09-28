#!/bin/bash
# OPD 教师离线打分缓存（论文 §4 三路蒸馏的训练信号源）。
# 教师 = 冻结 ViT-L；本脚本把每 token 的 logits/top-K 原子写到 $OPD_TEACHER_SCORE_DIR。
#
# 用法（远程 3×3090，精度无关——纯前向）：
#   OPD_TEACHER_SCORE_DIR=$NAVSIM_TRAJPDM_ROOT/opd/teacher/vit_l/8192_cache_ori \
#   TEACHER_CKPT=$NAVSIM_EXP_ROOT/models/gtrs_aug_drivesuprim_vit/epoch=05-step=1330.ckpt \
#   NPROC=3 bash scripts/opd/training/cache_teacher.sh
#
# 旋钮：
#   OPD_STORE_HEADS=1   （默认）落盘 8 头 logits（fp16）。lambda_head 默认 1.0 就需要它；
#                       设 0 会让训练端把 head 那一路整体置零（不报错，但少一路监督）。
#   OPD_ONPOLICY=1      写 on-policy 目录（用 OPD_TEACHER_ONPOLICY_SCORE_DIR）
#   OPD_VIEW_IDX=n      缓存落盘的视图编号；训练端会断言它与配对的学生 predictions[n] 一致。
#                       原视图缓存用 0（默认）；on-policy 缓存用 1（= opd.on_policy_view_idx）。
#   OPD_DUAL_STREAM_SCORE=true（默认）缓存用式 (4-6)~(4-8) 的双流融合式算 coarse/topk_idx。
#                       **必须与训练端 agent.config.opd.dual_stream_score 一致**，
#                       否则训练端一致性断言会硬报错。旧的 legacy 缓存可用
#                       `scripts/opd/tools/migrate_teacher_cache.py --dual_stream` 原地刷成双流。
#
# 容量（navtrain ~103k token，vocab 8192）：
#   imi fp32 32KB + coarse fp32 32KB + topk 2KB ≈ 66KB/token → ~6.8 GB
#   + 8 头 fp16 128KB/token                                  → 共 ~20 GB
set -euo pipefail

teacher_agent=${TEACHER_AGENT:-gtrs_aug_drivesuprim_vit}
teacher_ckpt=${TEACHER_CKPT:?must set TEACHER_CKPT to the frozen ViT-L checkpoint}
nproc=${NPROC:-3}
master_port=${MASTER_PORT:-29520}
bs=${BS:-8}

if [ "${OPD_ONPOLICY:-0}" = "1" ]; then
  score_dir=${OPD_TEACHER_ONPOLICY_SCORE_DIR:?must set OPD_TEACHER_ONPOLICY_SCORE_DIR when OPD_ONPOLICY=1}
  # on-policy 观测 = 旋转了 -dθ 的视图，对应学生 predictions[1]
  view_idx=${OPD_VIEW_IDX:-1}
else
  score_dir=${OPD_TEACHER_SCORE_DIR:?must set OPD_TEACHER_SCORE_DIR}
  view_idx=${OPD_VIEW_IDX:-0}
fi

mkdir -p "$score_dir"

# default_evaluation.yaml 里 `experiment_name: ???` 且 `hydra.run.dir: ${output_dir}`
# 在 job 启动时就要解析 —— 不传这个键，Hydra 在跑任何代码前就 MissingMandatoryValue，
# 一个 token 都不会落盘。
experiment_name=${EXP_NAME:-opd_teacher_cache}

command_string="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_teacher_opd_cache.py \
    +debug=false \
    agent=${teacher_agent} \
    experiment_name=${experiment_name} \
    train_test_split=navtrain \
    dataloader.params.batch_size=${bs} \
    ~trainer.params.strategy \
    trainer.params.precision=32 \
    agent.checkpoint_path='${teacher_ckpt}' \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=teacher \
    ++agent.config.opd.safe_fused_score=true \
    ++agent.config.opd.dual_stream_score=${OPD_DUAL_STREAM_SCORE:-true} \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.refinement_approach=transformer_decoder \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    +cache_path=null
"

echo "=== OPD teacher cache ==="
echo "  out_dir : $score_dir"
echo "  agent   : $teacher_agent"
echo "  ckpt    : $teacher_ckpt"
echo "  nproc   : $nproc   bs $bs   heads ${OPD_STORE_HEADS:-1}   onpolicy ${OPD_ONPOLICY:-0}   view_idx $view_idx"
echo "--- COMMAND ---"
echo "$command_string"
echo

OPD_TEACHER_SCORE_DIR="$score_dir" \
OPD_TEACHER_ONPOLICY_SCORE_DIR="${OPD_TEACHER_ONPOLICY_SCORE_DIR:-$score_dir}" \
OPD_STORE_HEADS="${OPD_STORE_HEADS:-1}" \
OPD_VIEW_IDX="$view_idx" \
torchrun --nproc_per_node="$nproc" --master_port="$master_port" $command_string
