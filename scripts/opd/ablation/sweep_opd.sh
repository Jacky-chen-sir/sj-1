#!/bin/bash
# OPD 消融 sweep 总控：按组定义 env 组合，循环调 train_opd.sh 打分训练。
# 覆盖论文 §5.4 的格子（λ 独立性 / 三个温度 / λ 组合 / on-policy 轮数 / EMA 自蒸馏对照）。
#
# 用法：
#   bash scripts/opd/ablation/sweep_opd.sh            # 全量
#   SWEEP_GROUPS="default tau_imi_1" bash scripts/opd/ablation/sweep_opd.sh   # 只跑指定组
#   DRY_RUN=1 bash scripts/opd/ablation/sweep_opd.sh   # 只打命令不跑
#
# 词表规模消融（4096/16384）不在此处：换词表要重算 PDM 分数缓存 + 教师缓存 + 词表 npy，
# 不在 shell sweep 里做。EMA 自蒸馏对照在同构 R34 下进行（异构 ViT-L→R34 的 EMA 不成立）。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_OP="$HERE/../training/train_opd.sh"
MAIN_AGENT=${MAIN_AGENT:-gtrs_aug_opd_r34}
NUM_REFINE=${NUM_REFINE_STAGE:-1}
STAGE_LAYERS=${STAGE_LAYERS:-3}
TOPKS=${TOPKS:-256}
DRY_RUN=${DRY_RUN:-0}

# 教师缓存目录（离线 ViT-L）
TEACHER_SCORE_DIR=${OPD_TEACHER_SCORE_DIR:-${NAVSIM_TRAJPDM_ROOT}/opd/teacher/vit_l/8192_cache_ori}

run_group() {
  # 用法：run_group <tag> [KEY=VAL ...]   —— KEY=VAL 逐个以 env 前缀传入子进程
  local tag="$1"; shift
  local tag_full="${EXP_TAG:-}${tag}"
  echo ">>> [${tag}]  EXP_TAG=${tag_full}  $*"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN: env $* OPD_TEACHER_SCORE_DIR=$TEACHER_SCORE_DIR EXP_TAG=$tag_full \\"
    echo "          bash $TRAIN_OP $MAIN_AGENT $NUM_REFINE $STAGE_LAYERS $TOPKS"
    echo
    return 0
  fi
  env "$@" OPD_TEACHER_SCORE_DIR="$TEACHER_SCORE_DIR" EXP_TAG="$tag_full" \
      bash "$TRAIN_OP" "$MAIN_AGENT" "$NUM_REFINE" "$STAGE_LAYERS" "$TOPKS"
  echo
}

want() { case " ${SWEEP_GROUPS} " in *" $1 "*) return 0;; *) return 1;; esac; }

# 先跑 `none lam_02 default lam_20` 这 4 组：它们标定蒸馏总权重的量级。
# 四路 λ 默认合计 3.5，而 KL 项内部还乘了 τ²=4，加在 O(1) 的学生基础损失上——
# 蒸馏很可能盖过主损失、把学生拉向教师的错误。没有这条标定曲线，其余格子的差异读不出来。
# 注意：不能叫 GROUPS——那是 bash 内置只读数组（当前用户 gid 列表），赋值无效/报错，
# 会导致一组都匹配不上。
SWEEP_GROUPS=${SWEEP_GROUPS:-"pure_offline no_decay ema_eval_only hardcopy_3 \
none lam_02 default lam_20 \
im_only head_only refine_only recall_only \
tau_imi_1 tau_imi_4 tau_head_1 tau_head_4 tau_list_1 tau_list_4 \
no_head no_recall emis_recall_1 topk_refine_32 topk_refine_1024 \
rounds_0 rounds_1 rounds_2 ema"}

echo "=== OPD sweep ==="
echo "  teacher_score_dir: $TEACHER_SCORE_DIR"
echo "  groups           : $SWEEP_GROUPS"
echo "  agent            : $MAIN_AGENT  stages=$NUM_REFINE/$STAGE_LAYERS/$TOPKS"
echo

# —— 混合配方拆解（先跑这 4 组 + default + none）——
# default      = 混合配方：ViT-L 蒸馏 + EMA 软标签 + EMA 评测 + 硬拷贝 1 epoch + λ cosine→0.3
# pure_offline = 改动前的配方（纯 ViT-L 蒸馏、无 EMA、无软标签、λ 常数）——对照上界提升来自哪
# none         = default 去掉 ViT-L 蒸馏（≈官方 base + 硬拷贝 1 epoch）——隔离蒸馏本身的贡献
want pure_offline   && run_group "pure_offline"  OPD_EMA_EVAL=false OPD_EMA_SOFT_LABEL=false OPD_LAMBDA_DECAY=none
want no_decay       && run_group "no_decay"      OPD_LAMBDA_DECAY=none
want ema_eval_only  && run_group "ema_eval_only" OPD_EMA_SOFT_LABEL=false
want hardcopy_3     && run_group "hardcopy_3"    OPD_EMA_HARDCOPY_EPOCHS=3

# —— 蒸馏总权重标定 ——
# none = 走完全相同的代码路径但不读教师缓存（EMA/软标签配置与 default 相同），是"同路径基线"
want none           && run_group "none"          OPD_TEACHER_MODE=none
want lam_02         && run_group "lam_02"        OPD_LAMBDA_IMI=0.2 OPD_LAMBDA_HEAD=0.2 OPD_LAMBDA_REFINE=0.2 OPD_LAMBDA_RECALL=0.1
want default        && run_group "default"
want lam_20         && run_group "lam_20"        OPD_LAMBDA_IMI=2.0 OPD_LAMBDA_HEAD=2.0 OPD_LAMBDA_REFINE=2.0 OPD_LAMBDA_RECALL=1.0

want im_only        && run_group "im_only"       OPD_LAMBDA_HEAD=0 OPD_LAMBDA_REFINE=0 OPD_LAMBDA_RECALL=0
want head_only      && run_group "head_only"     OPD_LAMBDA_IMI=0  OPD_LAMBDA_REFINE=0 OPD_LAMBDA_RECALL=0
want refine_only    && run_group "refine_only"   OPD_LAMBDA_IMI=0  OPD_LAMBDA_HEAD=0   OPD_LAMBDA_RECALL=0
want recall_only    && run_group "recall_only"   OPD_LAMBDA_IMI=0  OPD_LAMBDA_HEAD=0   OPD_LAMBDA_REFINE=0
want tau_imi_1      && run_group "tau_imi_1"     OPD_TAU_IMI=1.0
want tau_imi_4      && run_group "tau_imi_4"     OPD_TAU_IMI=4.0
want tau_head_1     && run_group "tau_head_1"    OPD_TAU_HEAD=1.0
want tau_head_4     && run_group "tau_head_4"    OPD_TAU_HEAD=4.0
want tau_list_1     && run_group "tau_list_1"    OPD_TAU_LIST=1.0
want tau_list_4     && run_group "tau_list_4"    OPD_TAU_LIST=4.0
want no_head        && run_group "no_head"       OPD_LAMBDA_HEAD=0
want no_recall      && run_group "no_recall"     OPD_LAMBDA_RECALL=0
want emis_recall_1  && run_group "emis_recall_1" OPD_LAMBDA_RECALL=1.0
# listwise 的教师 Top-K 宽度（缓存落盘 K=256，这里只能往小截）
want topk_refine_32   && run_group "topk_refine_32"   OPD_TOPK_REFINE=32
want topk_refine_1024 && run_group "topk_refine_1024" OPD_TOPK_REFINE=1024   # 会被截到落盘 K
want rounds_0       && run_group "rounds_0"      OPD_ON_POLICY_ROUNDS=0
want rounds_1       && run_group "rounds_1"      OPD_ON_POLICY_ROUNDS=1
want rounds_2       && run_group "rounds_2"      OPD_ON_POLICY_ROUNDS=2
want ema            && run_group "ema"           OPD_TEACHER_MODE=ema

echo "=== sweep done ==="
