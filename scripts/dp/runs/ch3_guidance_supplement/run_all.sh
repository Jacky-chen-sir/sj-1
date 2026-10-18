#!/bin/bash
# 第三章「评价器引导」补充实验总控。设计说明见同目录 RUNBOOK.md。
#
# 用法：
#   export START_CKPT=.../epoch=18-step=81776.ckpt
#   export SCORER_CKPT=.../step-step=013000.ckpt
#   STAGE=0 bash scripts/dp/runs/ch3_guidance_supplement/run_all.sh
#   STAGE=1 bash scripts/dp/runs/ch3_guidance_supplement/run_all.sh
#   STAGE=2 bash scripts/dp/runs/ch3_guidance_supplement/run_all.sh
#
#   DRY_RUN=1 ...       只打印命令
#   SKIP_EVAL=1 ...     只训练不评测
#   EVAL_ONLY=1 ...     只对已有 checkpoint 评测（训练已跑完时用）
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TRAIN="$ROOT_DIR/scripts/training/train_dp_official_fm.sh"
EVAL="$ROOT_DIR/scripts/evaluation/eval_dp_fm_joint_navtest.sh"

: "${START_CKPT:?请指定 START_CKPT（FM-DP epoch=18）}"
: "${SCORER_CKPT:?请指定 SCORER_CKPT（第 4 章 OPD R34）}"
SCORER_VOCAB="${SCORER_VOCAB:-${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR}/traj_final/test_8192_kmeans.npy}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
REL="${REL:-ch3_supplement}"
OUT="${OUT:-$NAVSIM_EXP_ROOT/$REL}"

STAGE="${STAGE:-0}"
GPU="${GPU:-0,1,2}"
EPOCHS="${EPOCHS:-23}"
BS="${BS:-4}"
WORKERS="${WORKERS:-4}"
FM_STEPS="${FM_STEPS:-20}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

mkdir -p "$OUT"
SUMMARY="$OUT/summary.csv"
[ -f "$SUMMARY" ] || echo "tag,stage,score,ckpt" > "$SUMMARY"

say() { echo; echo "=============================================================="; echo "$*"; echo "=============================================================="; }

train_one() {
  local tag="$1"; shift
  local exp="$REL/$tag"
  say "[TRAIN] $tag"
  echo "[ENV] $*"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: env $* RESUME_CKPT=$START_CKPT MAX_EPOCHS=$EPOCHS EXP_NAME=$exp \\"
    echo "         GPU=$GPU BS=$BS NUM_WORKERS=$WORKERS SCORER_CKPT=$SCORER_CKPT \\"
    echo "         SCORER_VOCAB_PATH=$SCORER_VOCAB bash $TRAIN"
    return 0
  fi
  env "$@" \
    RESUME_CKPT="$START_CKPT" \
    MAX_EPOCHS="$EPOCHS" \
    EXP_NAME="$exp" \
    GPU="$GPU" BS="$BS" NUM_WORKERS="$WORKERS" \
    SCORER_CKPT="$SCORER_CKPT" \
    SCORER_VOCAB_PATH="$SCORER_VOCAB" \
    bash "$TRAIN"
}

latest_ckpt() {
  ls -1t "$1"/epoch=*.ckpt 2>/dev/null | head -n 1 || true
}

eval_one() {
  local tag="$1"; local ckpt="$2"
  say "[EVAL] $tag  ($ckpt)"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: CKPT=$ckpt EXP_NAME=$REL/navtest-$tag GPU=0 BS=8 WORKERS=8 FM_NUM_INFERENCE_STEPS=$FM_STEPS bash $EVAL"
    return 0
  fi
  CKPT="$ckpt" EXP_NAME="$REL/navtest-$tag" GPU=0 BS=8 WORKERS=8 \
  FM_NUM_INFERENCE_STEPS="$FM_STEPS" FM_SELF_CONDITIONING=true FM_CLIP_SAMPLE=true \
    bash "$EVAL"

  # 从评测目录里取总分行
  local csv
  csv="$(find "$NAVSIM_EXP_ROOT/$REL/navtest-$tag" -name '*.csv' 2>/dev/null | head -n 1 || true)"
  if [ -n "$csv" ]; then
    python3 - "$csv" "$tag" "$STAGE" "$ckpt" "$SUMMARY" <<'PY'
import csv, sys
path, tag, stage, ckpt, out = sys.argv[1:6]
score = ''
with open(path, newline='', encoding='utf-8', errors='replace') as f:
    for row in csv.DictReader(f):
        tok = (row.get('token') or '').strip()
        if tok in ('extended_pdm_score_stage_one', 'pdm_score_stage_one', 'score') or 'score_stage_one' in tok:
            score = (row.get('score') or '').strip(); break
with open(out, 'a', newline='', encoding='utf-8') as f:
    csv.writer(f).writerow([tag, stage, score, ckpt])
print(f'[SUMMARY] {tag}\tstage={stage}\tscore={score}')
PY
  else
    echo "[WARN] 没找到评测 CSV：$NAVSIM_EXP_ROOT/$REL/navtest-$tag"
  fi
}

run_case() {
  local tag="$1"; shift
  local exp="$OUT/$tag"          # 绝对路径，供 latest_ckpt/find 使用
  if [ "$EVAL_ONLY" = "1" ]; then
    local ck; ck="$(latest_ckpt "$exp")"
    [ -n "$ck" ] || { echo "[SKIP] $tag: 没有 checkpoint"; return 0; }
    eval_one "$tag" "$ck"
    return 0
  fi
  train_one "$tag" "$@"
  if [ "$DRY_RUN" = "1" ] || [ "$SKIP_EVAL" = "1" ]; then return 0; fi
  # 训练脚本停下来后，取最新 epoch checkpoint
  local ck; ck="$(latest_ckpt "$exp")"
  [ -n "$ck" ] || { echo "[WARN] $tag: 找不到 epoch checkpoint"; return 0; }
  eval_one "$tag" "$ck"
}

# ------------------------------------------------------------------ Stage 0
if [ "$STAGE" = "0" ]; then
  run_case control \
    SCORER_GUIDANCE=false \
    LAMBDA_SCORE=0 LAMBDA_SAFE_FEAS=0 LAMBDA_COMFORT_DIV=0
fi

# ------------------------------------------------------------------ Stage 1
if [ "$STAGE" = "1" ]; then
  run_case g_xs \
    SCORER_GUIDANCE=true \
    LAMBDA_SCORE=0.02 LAMBDA_SAFE_FEAS=0.02 LAMBDA_COMFORT_DIV=0.01
  run_case g_s \
    SCORER_GUIDANCE=true \
    LAMBDA_SCORE=0.05 LAMBDA_SAFE_FEAS=0.05 LAMBDA_COMFORT_DIV=0.025
fi

# ------------------------------------------------------------------ Stage 2
if [ "$STAGE" = "2" ]; then
  run_case warm0 \
    SCORER_GUIDANCE=true \
    LAMBDA_SCORE=0.05 LAMBDA_SAFE_FEAS=0.05 LAMBDA_COMFORT_DIV=0.025 \
    GUIDANCE_WARMUP_STEPS=0
  run_case band03 \
    SCORER_GUIDANCE=true \
    LAMBDA_SCORE=0.05 LAMBDA_SAFE_FEAS=0.05 LAMBDA_COMFORT_DIV=0.025 \
    GUIDANCE_T_MIN=0.3
  run_case n4 \
    SCORER_GUIDANCE=true \
    LAMBDA_SCORE=0.05 LAMBDA_SAFE_FEAS=0.05 LAMBDA_COMFORT_DIV=0.025 \
    GUIDANCE_SAMPLES=4
  run_case margin03 \
    SCORER_GUIDANCE=true \
    LAMBDA_SCORE=0.05 LAMBDA_SAFE_FEAS=0.05 LAMBDA_COMFORT_DIV=0.025 \
    SAFE_FEAS_MARGIN=0.3
fi

say "[DONE] Stage $STAGE"
echo "汇总：$SUMMARY"
[ -s "$SUMMARY" ] && column -s, -t "$SUMMARY" 2>/dev/null || true
