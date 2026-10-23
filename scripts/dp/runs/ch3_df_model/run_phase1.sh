#!/bin/bash
# 第三章 Phase 1：中心一致性训练（两臂对照）。
#
#   P1_base   ：官方初始化 + 重 init 轨迹头，fm_center_weight=0
#   P1_center ：同上，fm_center_weight=0.5（零噪声 Euler 路径监督）
#
# 两臂其余开关完全一致；训完后都用 DP_EVAL_ZERO_NOISE=true 评测（与 Phase0 主推一致）。
# 判据：P1_center − P1_base > +0.005（见 PLAN.md）。
#
# 用法：
#   bash scripts/dp/runs/ch3_df_model/run_phase1.sh
#   EPOCHS=5 ARM=base ...          # 先看趋势
#   ARM=center SKIP_TRAIN=1 ...    # 只评已有 ckpt
#   DRY_RUN=1 ...
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TRAIN="$ROOT_DIR/scripts/training/train_dp_official_fm.sh"
EVAL="$ROOT_DIR/scripts/evaluation/eval_dp_fm_joint_navtest.sh"

NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
REL="${REL:-ch3_df}"
ARM="${ARM:-both}"                 # base | center | both
# 默认 5：两臂同一基准（官方 ckpt + reinit 头），只跑短趋势；要拉满再 EPOCHS=18
EPOCHS="${EPOCHS:-5}"
BS="${BS:-8}"
WORKERS="${WORKERS:-4}"
GPU="${GPU:-0,1,2}"
FM_STEPS="${FM_STEPS:-20}"
FM_CENTER_WEIGHT="${FM_CENTER_WEIGHT:-0.5}"
FM_CENTER_STEPS="${FM_CENTER_STEPS:-4}"
FM_CENTER_WARMUP_STEPS="${FM_CENTER_WARMUP_STEPS:-2000}"
# 与 ep18 / Phase 0 对齐：自条件关闭。两臂必须一致；想同时试自条件就都设 true。
FM_SELF_CONDITIONING="${FM_SELF_CONDITIONING:-false}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

SUMMARY="$NAVSIM_EXP_ROOT/$REL/summary_phase1.csv"
if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$NAVSIM_EXP_ROOT/$REL" "$NAVSIM_EXP_ROOT/logs"
  [ -f "$SUMMARY" ] || echo "arm,epochs,score,ckpt,eval_mode" > "$SUMMARY"
fi

say() { echo; echo "=============================================================="; echo "$*"; echo "=============================================================="; }

train_arm() {
  local arm="$1"; local center_w="$2"
  local exp="$REL/p1_$arm"
  say "[TRAIN] p1_$arm  epochs=$EPOCHS center_w=$center_w"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: MAX_EPOCHS=$EPOCHS EXP_NAME=$exp GPU=$GPU BS=$BS NUM_WORKERS=$WORKERS \\"
    echo "         REINIT_TRAJ_HEAD=true FM_CENTER_WEIGHT=$center_w FM_CENTER_STEPS=$FM_CENTER_STEPS \\"
    echo "         FM_CENTER_WARMUP_STEPS=$FM_CENTER_WARMUP_STEPS \\"
    echo "         FM_SELF_CONDITIONING=$FM_SELF_CONDITIONING SCORER_GUIDANCE=false bash $TRAIN"
    return 0
  fi
  if [ "$SKIP_TRAIN" = "1" ]; then
    echo "[SKIP_TRAIN] $exp"
    return 0
  fi
  MAX_EPOCHS="$EPOCHS" EXP_NAME="$exp" GPU="$GPU" BS="$BS" NUM_WORKERS="$WORKERS" \
  REINIT_TRAJ_HEAD=true \
  FM_CENTER_WEIGHT="$center_w" FM_CENTER_STEPS="$FM_CENTER_STEPS" FM_CENTER_P=1.0 \
  FM_CENTER_WARMUP_STEPS="$FM_CENTER_WARMUP_STEPS" \
  FM_SELF_CONDITIONING="$FM_SELF_CONDITIONING" \
  SCORER_GUIDANCE=false LAMBDA_SCORE=0 LAMBDA_SAFE_FEAS=0 LAMBDA_COMFORT_DIV=0 \
    bash "$TRAIN"
}

eval_arm() {
  local arm="$1"
  local exp_dir="$NAVSIM_EXP_ROOT/$REL/p1_$arm"
  local ckpt
  if [ "$DRY_RUN" = "1" ]; then
    ckpt="<${exp_dir}/epoch=XX.ckpt>"
  else
    ckpt="$(ls -1t "$exp_dir"/epoch=*.ckpt 2>/dev/null | head -n 1 || true)"
    [ -n "$ckpt" ] || { echo "[WARN] p1_$arm: 无 epoch ckpt"; return 0; }
  fi
  local tag="navtest-p1_${arm}-zero"
  say "[EVAL] p1_$arm  zero-noise  ($ckpt)"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: CKPT=$ckpt EXP_NAME=$REL/$tag DP_EVAL_ZERO_NOISE=true NUM_PROPOSALS=1 \\"
    echo "         FM_SELF_CONDITIONING=$FM_SELF_CONDITIONING SEED=3000 bash $EVAL"
    return 0
  fi
  if [ "$SKIP_EVAL" = "1" ]; then
    echo "[SKIP_EVAL] $arm"
    return 0
  fi
  mkdir -p "$NAVSIM_EXP_ROOT/$REL/$tag"
  CKPT="$ckpt" EXP_NAME="$REL/$tag" GPU=0 BS=8 WORKERS=8 FORCE_RERUN=1 \
  DP_EVAL_ZERO_NOISE=true NUM_PROPOSALS=1 SEED=3000 \
  FM_NUM_INFERENCE_STEPS="$FM_STEPS" \
  FM_SELF_CONDITIONING="$FM_SELF_CONDITIONING" FM_CLIP_SAMPLE=true \
  SUBSCORE_PATH="$NAVSIM_EXP_ROOT/$REL/$tag/proposals.pkl" \
    bash "$EVAL"

  local csv
  csv="$(find "$NAVSIM_EXP_ROOT/$REL/$tag" -name '*.csv' 2>/dev/null | head -n 1 || true)"
  if [ -n "$csv" ]; then
    python3 - "$csv" "$arm" "$EPOCHS" "$ckpt" "$SUMMARY" <<'PY'
import csv, sys
path, arm, ep, ckpt, out = sys.argv[1:6]
score = ''
with open(path, newline='', encoding='utf-8', errors='replace') as f:
    for row in csv.DictReader(f):
        tok = (row.get('token') or '').strip()
        if tok in ('extended_pdm_score_stage_one', 'extended_pdm_score_combined'):
            score = (row.get('score') or '').strip(); break
with open(out, 'a', newline='', encoding='utf-8') as f:
    csv.writer(f).writerow([arm, ep, score, ckpt, 'zero_noise'])
print(f'[SUMMARY] p1_{arm}\tepochs={ep}\tscore={score}\t(zero)')
PY
  fi
}

case "$ARM" in
  base)    train_arm base 0.0; eval_arm base ;;
  center)  train_arm center "$FM_CENTER_WEIGHT"; eval_arm center ;;
  both)
    train_arm base 0.0
    train_arm center "$FM_CENTER_WEIGHT"
    eval_arm base
    eval_arm center
    ;;
  *) echo "ARM must be base|center|both"; exit 1 ;;
esac

say "[DONE] Phase 1  summary → $SUMMARY"
if [ -f "$SUMMARY" ] && [ "$DRY_RUN" != "1" ]; then
  column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
fi
