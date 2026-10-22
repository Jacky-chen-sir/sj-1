#!/bin/bash
# 第三章「从零（官方初始化）」公平对照。
#
# 「从零」的定义：
#   agent.checkpoint_path = 官方 gtrs_dp_model.ckpt（保留感知骨干）
#   REINIT_TRAJ_HEAD      = true（轨迹头随机重初始化）
#   不设 RESUME_CKPT      = 不恢复优化器 / 不继承上一段训练
#
# 两条臂**除 FM 开关外完全相同**：同初始化、同 epoch 预算、同 BS、同评测口径。
#
# 用法：
#   ARM=both EPOCHS=10 bash scripts/dp/runs/ch3_guidance_supplement/run_from0.sh
#   ARM=fm_only EPOCHS=18 ...
#   DRY_RUN=1 ...          只打印命令
#   SKIP_EVAL=1 ...        只训练
#
# 评测口径：
#   fm_only 训练时自条件关闭 → 评测必须 FM_SELF_CONDITIONING=false
#   full    训练时自条件开启 → 评测必须 FM_SELF_CONDITIONING=true
#   两边 K=20、同一 metric cache。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TRAIN="$ROOT_DIR/scripts/training/train_dp_official_fm.sh"
EVAL="$ROOT_DIR/scripts/evaluation/eval_dp_fm_joint_navtest.sh"

NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
REL="${REL:-ch3_from0}"
OUT="${OUT:-$NAVSIM_EXP_ROOT/$REL}"

ARM="${ARM:-both}"                 # fm_only | full | both
EPOCHS="${EPOCHS:-10}"
BS="${BS:-8}"
WORKERS="${WORKERS:-8}"
GPU="${GPU:-0,1,2}"
FM_STEPS="${FM_STEPS:-20}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "$OUT"
SUMMARY="$OUT/summary.csv"
[ -f "$SUMMARY" ] || echo "arm,epochs,score,ckpt" > "$SUMMARY"

say() { echo; echo "=============================================================="; echo "$*"; echo "=============================================================="; }

# ---- 公共：官方初始化，不 resume ----
# 注意：OFFICIAL_DP_CKPT 在训练脚本里是硬编码路径，这里只做存在性检查。
train_arm() {
  local arm="$1"; shift
  local exp="$REL/$arm"
  say "[TRAIN] $arm  (${EPOCHS} epochs, from official init)"
  echo "[ENV] $*"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: env $* MAX_EPOCHS=$EPOCHS EXP_NAME=$exp GPU=$GPU BS=$BS NUM_WORKERS=$WORKERS \\"
    echo "         REINIT_TRAJ_HEAD=true bash $TRAIN"
    return 0
  fi
  env "$@" \
    MAX_EPOCHS="$EPOCHS" \
    EXP_NAME="$exp" \
    GPU="$GPU" BS="$BS" NUM_WORKERS="$WORKERS" \
    REINIT_TRAJ_HEAD=true \
    bash "$TRAIN"
}

eval_arm() {
  local arm="$1"; local self_cond="$2"
  local exp="$OUT/$arm"
  local ckpt
  ckpt="$(ls -1t "$exp"/epoch=*.ckpt 2>/dev/null | head -n 1 || true)"
  [ -n "$ckpt" ] || { echo "[WARN] $arm: 找不到 epoch checkpoint"; return 0; }
  say "[EVAL] $arm  ($ckpt, self_cond=$self_cond)"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: CKPT=$ckpt EXP_NAME=$REL/navtest-$arm GPU=0 BS=8 WORKERS=8 \\"
    echo "         FM_NUM_INFERENCE_STEPS=$FM_STEPS FM_SELF_CONDITIONING=$self_cond FM_CLIP_SAMPLE=true bash $EVAL"
    return 0
  fi
  CKPT="$ckpt" EXP_NAME="$REL/navtest-$arm" GPU=0 BS=8 WORKERS=8 \
  FM_NUM_INFERENCE_STEPS="$FM_STEPS" \
  FM_SELF_CONDITIONING="$self_cond" FM_CLIP_SAMPLE=true \
    bash "$EVAL"

  local csv
  csv="$(find "$NAVSIM_EXP_ROOT/$REL/navtest-$arm" -name '*.csv' 2>/dev/null | head -n 1 || true)"
  if [ -n "$csv" ]; then
    python3 - "$csv" "$arm" "$EPOCHS" "$ckpt" "$SUMMARY" <<'PY'
import csv, sys
path, arm, ep, ckpt, out = sys.argv[1:6]
score = ''
with open(path, newline='', encoding='utf-8', errors='replace') as f:
    for row in csv.DictReader(f):
        tok = (row.get('token') or '').strip()
        if 'score_stage_one' in tok:
            score = (row.get('score') or '').strip(); break
with open(out, 'a', newline='', encoding='utf-8') as f:
    csv.writer(f).writerow([arm, ep, score, ckpt])
print(f'[SUMMARY] {arm}\tepochs={ep}\tscore={score}')
PY
  fi
}

# ---------------------------------------------------------------- fm_only
# 基线臂：与 ep18 相同的配方——只开流匹配，其余全关。
if [ "$ARM" = "fm_only" ] || [ "$ARM" = "both" ]; then
  train_arm fm_only \
    SCORER_GUIDANCE=false \
    FM_TRAJ_AUX_WEIGHT=0 FM_KIN_WEIGHT=0 \
    FM_LATTICE_T_PROB=0 FM_SELF_CONSISTENCY_P=0 FM_SELF_CONDITIONING=false \
    LAMBDA_SCORE=0 LAMBDA_SAFE_FEAS=0 LAMBDA_COMFORT_DIV=0
  [ "$SKIP_EVAL" = "1" ] || eval_arm fm_only false
fi

# ---------------------------------------------------------------- full
# 方法臂：当前默认（结构化改进全开），但引导关闭。
if [ "$ARM" = "full" ] || [ "$ARM" = "both" ]; then
  train_arm full \
    SCORER_GUIDANCE=false \
    LAMBDA_SCORE=0 LAMBDA_SAFE_FEAS=0 LAMBDA_COMFORT_DIV=0
  [ "$SKIP_EVAL" = "1" ] || eval_arm full true
fi

say "[DONE] ARM=$ARM EPOCHS=$EPOCHS"
echo "汇总：$SUMMARY"
[ -s "$SUMMARY" ] && column -s, -t "$SUMMARY" 2>/dev/null || true
echo
echo "判读：两臂必须取**同一 epoch** 的 ckpt 比较，不要拿 final 对 final 之外的点。"
echo "      full > fm_only 且差值稳定 → 结构化改进成立；否则回到单项消融。"
