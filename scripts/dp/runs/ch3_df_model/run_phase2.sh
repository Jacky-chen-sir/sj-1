#!/bin/bash
# 第三章 Phase 2：长预算复现（按需，Phase 1 过线后跑）。
#
#   P2_center18 ：Phase 1 有效配方（fm_center_weight=0.5）拉满 18 epoch
#   P2_fm18     ：已有 ep18 ckpt，默认只做对照评测（SKIP 训练）
#
# 评测统一 DP_EVAL_ZERO_NOISE=true。对照：P0 zero 0.7657 / P1_center 0.7125。
#
# 用法：
#   bash scripts/dp/runs/ch3_df_model/run_phase2.sh
#   # 与 Phase1 实际配方对齐（p1 启动时环境带了 self_cond=true）：
#   FM_SELF_CONDITIONING=true bash scripts/dp/runs/ch3_df_model/run_phase2.sh
#   ARM=center SKIP_EVAL=1 ...     # 只训
#   ARM=center SKIP_TRAIN=1 ...    # 只评已有 ckpt
#   DRY_RUN=1 ...
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TRAIN="$ROOT_DIR/scripts/training/train_dp_official_fm.sh"
EVAL="$ROOT_DIR/scripts/evaluation/eval_dp_fm_joint_navtest.sh"

NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
REL="${REL:-ch3_df}"
ARM="${ARM:-center}"               # center | fm18 | both
EPOCHS="${EPOCHS:-18}"
BS="${BS:-8}"
WORKERS="${WORKERS:-4}"
GPU="${GPU:-0,1,2}"
FM_STEPS="${FM_STEPS:-20}"
FM_CENTER_WEIGHT="${FM_CENTER_WEIGHT:-0.5}"
FM_CENTER_STEPS="${FM_CENTER_STEPS:-4}"
FM_CENTER_WARMUP_STEPS="${FM_CENTER_WARMUP_STEPS:-2000}"
# 默认 true：与已跑通的 p1_center 同口径（若要对齐 Phase0 再显式 false）
FM_SELF_CONDITIONING="${FM_SELF_CONDITIONING:-true}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
# 已有 FM-only 18ep ckpt（P2_fm18 对照）
FM18_CKPT="${FM18_CKPT:-$NAVSIM_EXP_ROOT/train_dp_official_fm_joint/epoch=18-step=81776.ckpt}"

SUMMARY="$NAVSIM_EXP_ROOT/$REL/summary_phase2.csv"
if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$NAVSIM_EXP_ROOT/$REL" "$NAVSIM_EXP_ROOT/logs"
  [ -f "$SUMMARY" ] || echo "arm,epochs,score,ckpt,eval_mode" > "$SUMMARY"
fi

say() { echo; echo "=============================================================="; echo "$*"; echo "=============================================================="; }

append_summary() {
  local csv="$1" arm="$2" ep="$3" ckpt="$4"
  [ -n "$csv" ] || return 0
  python3 - "$csv" "$arm" "$ep" "$ckpt" "$SUMMARY" <<'PY'
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
print(f'[SUMMARY] p2_{arm}\tepochs={ep}\tscore={score}\t(zero)')
PY
}

train_center() {
  local exp="$REL/p2_center18"
  say "[TRAIN] p2_center18  epochs=$EPOCHS center_w=$FM_CENTER_WEIGHT self_cond=$FM_SELF_CONDITIONING"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: MAX_EPOCHS=$EPOCHS EXP_NAME=$exp GPU=$GPU BS=$BS NUM_WORKERS=$WORKERS \\"
    echo "         REINIT_TRAJ_HEAD=true FM_CENTER_WEIGHT=$FM_CENTER_WEIGHT \\"
    echo "         FM_SELF_CONDITIONING=$FM_SELF_CONDITIONING bash $TRAIN"
    return 0
  fi
  if [ "$SKIP_TRAIN" = "1" ]; then
    echo "[SKIP_TRAIN] $exp"
    return 0
  fi
  MAX_EPOCHS="$EPOCHS" EXP_NAME="$exp" GPU="$GPU" BS="$BS" NUM_WORKERS="$WORKERS" \
  REINIT_TRAJ_HEAD=true \
  FM_CENTER_WEIGHT="$FM_CENTER_WEIGHT" FM_CENTER_STEPS="$FM_CENTER_STEPS" FM_CENTER_P=1.0 \
  FM_CENTER_WARMUP_STEPS="$FM_CENTER_WARMUP_STEPS" \
  FM_SELF_CONDITIONING="$FM_SELF_CONDITIONING" \
  SCORER_GUIDANCE=false LAMBDA_SCORE=0 LAMBDA_SAFE_FEAS=0 LAMBDA_COMFORT_DIV=0 \
    bash "$TRAIN"
}

eval_tag() {
  local arm="$1" ckpt="$2" tag="$3" ep="$4"
  say "[EVAL] p2_$arm  zero-noise  ($ckpt)"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: CKPT=$ckpt EXP_NAME=$REL/$tag DP_EVAL_ZERO_NOISE=true bash $EVAL"
    return 0
  fi
  if [ "$SKIP_EVAL" = "1" ]; then
    echo "[SKIP_EVAL] $arm"
    return 0
  fi
  [ -f "$ckpt" ] || { echo "[WARN] missing ckpt: $ckpt"; return 0; }
  mkdir -p "$NAVSIM_EXP_ROOT/$REL/$tag"
  CKPT="$ckpt" EXP_NAME="$REL/$tag" GPU=0 BS=8 WORKERS=8 FORCE_RERUN=1 \
  DP_EVAL_ZERO_NOISE=true NUM_PROPOSALS=1 SEED=3000 \
  FM_NUM_INFERENCE_STEPS="$FM_STEPS" \
  FM_SELF_CONDITIONING="$FM_SELF_CONDITIONING" FM_CLIP_SAMPLE=true \
  SUBSCORE_PATH="$NAVSIM_EXP_ROOT/$REL/$tag/proposals.pkl" \
    bash "$EVAL"
  local csv
  csv="$(find "$NAVSIM_EXP_ROOT/$REL/$tag" -name '*.csv' 2>/dev/null | head -n 1 || true)"
  append_summary "$csv" "$arm" "$ep" "$ckpt"
}

eval_center() {
  local exp_dir="$NAVSIM_EXP_ROOT/$REL/p2_center18"
  local ckpt
  if [ "$DRY_RUN" = "1" ]; then
    ckpt="<${exp_dir}/epoch=XX.ckpt>"
  else
    ckpt="$(ls -1t "$exp_dir"/epoch=*.ckpt 2>/dev/null | head -n 1 || true)"
    [ -n "$ckpt" ] || { echo "[WARN] p2_center18: 无 epoch ckpt"; return 0; }
  fi
  eval_tag center18 "$ckpt" "navtest-p2_center18-zero" "$EPOCHS"
}

eval_fm18() {
  eval_tag fm18 "$FM18_CKPT" "navtest-p2_fm18-zero" 18
}

case "$ARM" in
  center)
    train_center
    eval_center
    ;;
  fm18)
    eval_fm18
    ;;
  both)
    train_center
    eval_center
    eval_fm18
    ;;
  *) echo "ARM must be center|fm18|both"; exit 1 ;;
esac

say "[DONE] Phase 2  summary → $SUMMARY"
if [ -f "$SUMMARY" ] && [ "$DRY_RUN" != "1" ]; then
  column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
fi
