#!/bin/bash
# 论文第 4 章「零训练成本」分析链的总驱动：一次跑完 表 6（效率）与 表 7（分桶 + 显著性）。
#
# 为什么需要它：这几步在 README 里是分开写的，但彼此有顺序依赖，且有两处极易出错——
#   1. 两个方法的 POLICY 必须取同一个值（否则比的是打分口径不是模型，见 rescore_pickle.sh 顶部）；
#   2. bootstrap 必须用**同一份** per_token.csv 交集（bootstrap_ci.py 自己求交并报丢弃数）。
# 串起来跑就不会漏。
#
# 用法：
#   BASE_PKL=<自训基线预测 pickle> OURS_PKL=<本文方法预测 pickle> \
#     bash scripts/opd/analysis/run_zero_cost_analysis.sh
#
# 常用可选项：
#   POLICY=non_reactive|reactive   第一阶段交通流（两个方法必须一致；默认 non_reactive）
#   TAG_BASE=base_13k TAG_OURS=opd_13k
#   N_BOOT=10000                    bootstrap 次数
#   TEACHER_CKPT=<ViT-L ckpt> STUDENT_CKPT=<R34 ckpt>
#                                  给了就跑效率表（表 6）；不给则跳过
#   ABL=<消融根目录>                默认 $NAVSIM_EXP_ROOT/ablation
#   SKIP_RESCORE=1                  per_token.csv 已存在时跳过重打分
#   DRY_RUN=1                       只打印命令
#
# 产物（全部落在 $ABL/analysis/）：
#   efficiency.csv          表 6 的原始行（多次调用追加）
#   scene_attributes.csv    分桶维度
#   challenge_subsets.csv   表 7 的「场景数 / 自训基线 / 本文方法」列
#   bootstrap_ci.csv        表 7 的「Δ / 95% CI / p 值」列
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?请先 source 环境变量（见 scripts/opd/analysis/README.md §0）}"
: "${BASE_PKL:?BASE_PKL=<自训基线预测 pickle> required}"
: "${OURS_PKL:?OURS_PKL=<本文方法预测 pickle> required}"

POLICY=${POLICY:-non_reactive}
TAG_BASE=${TAG_BASE:-base_13k}
TAG_OURS=${TAG_OURS:-opd_13k}
N_BOOT=${N_BOOT:-10000}
ABL=${ABL:-${NAVSIM_EXP_ROOT}/ablation}
OUT=${OUT:-$ABL/analysis}
SKIP_RESCORE=${SKIP_RESCORE:-0}
DRY_RUN=${DRY_RUN:-0}

PT_BASE="$NAVSIM_EXP_ROOT/rescore/${TAG_BASE}-${POLICY}/per_token.csv"
PT_OURS="$NAVSIM_EXP_ROOT/rescore/${TAG_OURS}-${POLICY}/per_token.csv"
ATTRS="$OUT/scene_attributes.csv"

run() {
  echo "+ $*"
  [ "$DRY_RUN" = "1" ] || "$@"
}

cd "$NAVSIM_DEVKIT_ROOT"
[ "$DRY_RUN" = "1" ] || mkdir -p "$OUT"

echo "=== 零成本分析链 ==="
echo "  baseline : $TAG_BASE   ($BASE_PKL)"
echo "  ours     : $TAG_OURS   ($OURS_PKL)"
echo "  policy   : $POLICY   （两个方法必须一致）"
echo "  out      : $OUT"
echo

# —— 步骤 1：逐 token 打分表（#9 / #10 的输入，纯 CPU）——
if [ "$SKIP_RESCORE" = "1" ]; then
  echo "--- 1. 跳过重打分（SKIP_RESCORE=1）---"
else
  echo "--- 1. 逐 token 打分表 ---"
  run env PKL="$BASE_PKL" TAG="$TAG_BASE" POLICY="$POLICY" \
      bash scripts/opd/evaluation/rescore_pickle.sh
  run env PKL="$OURS_PKL" TAG="$TAG_OURS" POLICY="$POLICY" \
      bash scripts/opd/evaluation/rescore_pickle.sh
fi

# —— 步骤 2：场景属性表（分桶维度，只用 metric cache，跑一次可复用）——
echo "--- 2. 场景属性表 ---"
if [ "$DRY_RUN" != "1" ] && [ -f "$ATTRS" ]; then
  echo "  已存在，跳过：$ATTRS（要重算先删掉它）"
else
  run python navsim/planning/script/run_scene_attributes.py \
      train_test_split=navtest "++attrs.out=$ATTRS" ++attrs.workers=16
fi

# —— 步骤 3：效率表（表 6）——
echo "--- 3. 效率表（表 6）---"
if [ -z "${TEACHER_CKPT:-}" ] || [ -z "${STUDENT_CKPT:-}" ]; then
  echo "  跳过：未提供 TEACHER_CKPT / STUDENT_CKPT"
else
  run python navsim/planning/script/run_efficiency_profile.py \
      agent=gtrs_aug_opd_r34 "agent.checkpoint_path=$TEACHER_CKPT" \
      agent.config.inference.model=teacher \
      "++profile.tag=teacher_vitl" "++profile.out=$OUT/efficiency.csv"
  run python navsim/planning/script/run_efficiency_profile.py \
      agent=gtrs_aug_opd_r34 "agent.checkpoint_path=$STUDENT_CKPT" \
      agent.config.inference.model=student \
      "++profile.tag=student_r34" "++profile.out=$OUT/efficiency.csv"
fi

# —— 步骤 4：分桶 EPDMS（表 7 的左半边）——
echo "--- 4. 分桶 EPDMS ---"
run python scripts/opd/analysis/challenge_subsets.py \
    --attrs "$ATTRS" \
    "--run" "$TAG_BASE=$PT_BASE" \
    "--run" "$TAG_OURS=$PT_OURS" \
    --out "$OUT/challenge_subsets.csv"

# —— 步骤 5：bootstrap 置信区间与 p 值（表 7 的右半边）——
# 论文里 cluster（按 log）与 none（按场景独立）两个口径都要报，故各跑一次。
echo "--- 5. bootstrap（cluster = 按 log，论文主口径）---"
run python scripts/opd/analysis/bootstrap_ci.py \
    --baseline "$PT_BASE" --ours "$PT_OURS" --attrs "$ATTRS" \
    --cluster log_name --n-boot "$N_BOOT" \
    --out "$OUT/bootstrap_ci.csv"

echo "--- 5b. bootstrap（按场景独立，对照口径）---"
run python scripts/opd/analysis/bootstrap_ci.py \
    --baseline "$PT_BASE" --ours "$PT_OURS" --attrs "$ATTRS" \
    --cluster none --n-boot "$N_BOOT" \
    --out "$OUT/bootstrap_ci_scene.csv"

echo
echo "=== 完成。产物在 $OUT ==="
echo "  表 6 ← efficiency.csv"
echo "  表 7 ← challenge_subsets.csv（场景数 / 两列 EPDMS）+ bootstrap_ci.csv（Δ / CI / p）"
echo "  ⚠ 表 7 的两种 bootstrap 口径都要写进论文，并说明以 cluster 为准。"
