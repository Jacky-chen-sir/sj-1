#!/bin/bash
# 第三章 Phase 0：推断方式对照（不训练）。
#
# 三档：
#   p0_first_1..N  proposals[0]（官方行为），重复 N 次估计方差
#   p0_medoid      到其余提案距离和最小的样本（medoid）
#   p0_zero        零噪声起点（确定性路径；自动 num_proposals=1）
#
# ⚠ 与历史 0.7380 对齐：默认 FM_SELF_CONDITIONING=false（ep18 的 input_emb 是 32 列）。
#   三档必须使用同一自条件设置，否则比较混入第二个变量。
#
# 用法：
#   CKPT=$NAVSIM_EXP_ROOT/train_dp_official_fm_joint/epoch=18-step=81776.ckpt \
#     bash scripts/dp/runs/ch3_df_model/run_phase0.sh
#   DRY_RUN=1 ...        只打印命令
#   N_FIRST=3 ...        first 重复次数
#   FM_SELF_CONDITIONING=true ...  若 ckpt 是自条件训练的
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EVAL="$ROOT_DIR/scripts/evaluation/eval_dp_fm_joint_navtest.sh"
SUMMARIZE="$ROOT_DIR/scripts/dp/runs/ch3_df_model/summarize_phase0.py"

: "${CKPT:?请指定 CKPT=<FM-DP checkpoint>}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
REL="${REL:-ch3_df}"
N_FIRST="${N_FIRST:-3}"
FM_SELF_CONDITIONING="${FM_SELF_CONDITIONING:-false}"
FM_STEPS="${FM_STEPS:-20}"
GPU="${GPU:-0}"
BS="${BS:-8}"
WORKERS="${WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
# 官方 GTRS-DP（DDPM）对照：确认 0.7305 的自复现口径。设 RUN_OFFICIAL=0 跳过。
OFFICIAL_CKPT="${OFFICIAL_CKPT:-/home/ws/navsim_workspace/GTRS_official/data/models/gtrs_dp_model.ckpt}"
RUN_OFFICIAL="${RUN_OFFICIAL:-1}"

run_one() {
  local tag="$1"; local agg="$2"; local zero="$3"; local nprop="${4:-}"; local seed="${5:-0}"
  local use_fm="${USE_FM:-true}"
  local subscore="$NAVSIM_EXP_ROOT/$REL/$tag/proposals.pkl"
  echo
  echo "=============================================================="
  echo "[PHASE0] $tag  agg=$agg zero=$zero self_cond=$FM_SELF_CONDITIONING nprop=${nprop:-<default>} seed=$seed use_fm=$use_fm"
  echo "=============================================================="
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: CKPT=$CKPT EXP_NAME=$REL/$tag FORCE_RERUN=1 \\"
    echo "         DP_PROPOSAL_AGG=$agg DP_EVAL_ZERO_NOISE=$zero SEED=$seed USE_FM=$use_fm \\"
    echo "         FM_SELF_CONDITIONING=$FM_SELF_CONDITIONING FM_CLIP_SAMPLE=true \\"
    echo "         FM_NUM_INFERENCE_STEPS=$FM_STEPS GPU=$GPU BS=$BS WORKERS=$WORKERS \\"
    echo "         SUBSCORE_PATH=$subscore ${nprop:+NUM_PROPOSALS=$nprop }\\"
    echo "         bash $EVAL"
    return 0
  fi
  mkdir -p "$(dirname "$subscore")"
  CKPT="$CKPT" EXP_NAME="$REL/$tag" FORCE_RERUN=1 \
  DP_PROPOSAL_AGG="$agg" DP_EVAL_ZERO_NOISE="$zero" \
  SEED="$seed" USE_FM="$use_fm" \
  FM_SELF_CONDITIONING="$FM_SELF_CONDITIONING" FM_CLIP_SAMPLE=true \
  FM_NUM_INFERENCE_STEPS="$FM_STEPS" GPU="$GPU" BS="$BS" WORKERS="$WORKERS" \
  SUBSCORE_PATH="$subscore" \
  ${nprop:+NUM_PROPOSALS="$nprop"} \
    bash "$EVAL"
}

# first ×N（估方差）。nprop=1 是"单样本推断"这一臂的定义，省去另外 99 条的计算；
# 注意它**不等价**于"100 条取第 0 条"：噪声按 (B*P, H, A) 行优先生成，P 不同则
# 每个场景拿到的随机数不同。所以与历史 0.7380 的比较是分布层面的，不是逐场景配对。
# 若 medoid/zero 的增益接近判据（>0.01），再用 NUM_PROPOSALS=100 补一组配对确认。
# seed 必须显式区分：Lightning 缺省固定 seed=0，否则 N 次会抽出同一条轨迹。
for i in $(seq 1 "$N_FIRST"); do
  run_one "p0_first_$i" first false 1 "$((1000 + i))"
done

# medoid：需要多样本
run_one "p0_medoid" medoid false 100 2000

# 零噪声（评测脚本也会自动 num_proposals=1；此处显式写出）
run_one "p0_zero" first true 1 3000

# 官方 DDPM 对照（历史 0.7305）。用 100 提案取 [0]，与出 0.7305 时口径一致；
# 与 p0_first_1 同 seed，便于和 FM 单提案臂对照（注意 nprop 不同，非逐场景配对）。
if [ "$RUN_OFFICIAL" = "1" ]; then
  if [ -f "$OFFICIAL_CKPT" ] || [ "$DRY_RUN" = "1" ]; then
    CKPT="$OFFICIAL_CKPT" USE_FM=false \
      run_one "p0_official_first" first false 100 1001
  else
    echo "[WARN] 官方 ckpt 不存在，跳过官方对照: $OFFICIAL_CKPT"
    echo "       可设 OFFICIAL_CKPT=... 或 RUN_OFFICIAL=0"
  fi
else
  echo "[INFO] RUN_OFFICIAL=0，跳过官方 DDPM 对照"
fi

echo
echo "=============================================================="
echo "[SUMMARY]"
echo "=============================================================="
if [ "$DRY_RUN" = "1" ] || [ "$SKIP_EVAL" = "1" ]; then
  echo "跳过汇总（DRY_RUN/SKIP_EVAL）"
else
  python3 "$SUMMARIZE" --root "$NAVSIM_EXP_ROOT/$REL" \
    --out "$NAVSIM_EXP_ROOT/$REL/summary_phase0.csv"
fi
