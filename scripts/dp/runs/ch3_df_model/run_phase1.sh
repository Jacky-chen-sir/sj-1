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
#   ARM=both SKIP_TRAIN=1 ...      # 只评已有 ckpt（训练已在跑时用这个补评测）
#   EVAL_LAST_N=3 EVAL_STEPS="20 4" ...   # 默认：最后 3 个 epoch × K∈{20,4}
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

# 新表头：arm,epoch,K,score,ckpt（旧版 summary_phase1.csv 表头不同，换文件名避免混写）
SUMMARY="$NAVSIM_EXP_ROOT/$REL/summary_phase1_v2.csv"
if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$NAVSIM_EXP_ROOT/$REL" "$NAVSIM_EXP_ROOT/logs"
  [ -f "$SUMMARY" ] || echo "arm,epoch,K,score,ckpt" > "$SUMMARY"
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

# 评测最后 EVAL_LAST_N 个 epoch ckpt（相邻 ckpt 分数波动可达 0.01 以上，大于 +0.005 判据），
# 每个 ckpt 在 EVAL_STEPS 列出的每个 K 上做零噪声评测：
#   K=20 是论文口径；K=FM_CENTER_STEPS(4) 与训练时中心路径同一网格，用来判断增益能否迁移到 K=20。
EVAL_LAST_N="${EVAL_LAST_N:-3}"
EVAL_STEPS="${EVAL_STEPS:-$FM_STEPS $FM_CENTER_STEPS}"

eval_one() {
  local arm="$1" ckpt="$2" k="$3"
  local ep_tag
  ep_tag="$(basename "$ckpt" .ckpt)"; ep_tag="${ep_tag%%-step*}"; ep_tag="${ep_tag//=/}"
  local tag="navtest-p1_${arm}-zero-k${k}-${ep_tag}"
  say "[EVAL] p1_$arm  zero-noise  K=$k  ($ckpt)"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: CKPT=$ckpt EXP_NAME=$REL/$tag DP_EVAL_ZERO_NOISE=true NUM_PROPOSALS=1 \\"
    echo "         FM_NUM_INFERENCE_STEPS=$k FM_SELF_CONDITIONING=$FM_SELF_CONDITIONING SEED=3000 bash $EVAL"
    return 0
  fi
  mkdir -p "$NAVSIM_EXP_ROOT/$REL/$tag"
  CKPT="$ckpt" EXP_NAME="$REL/$tag" GPU=0 BS=8 WORKERS=8 FORCE_RERUN=1 \
  DP_EVAL_ZERO_NOISE=true NUM_PROPOSALS=1 SEED=3000 \
  FM_NUM_INFERENCE_STEPS="$k" \
  FM_SELF_CONDITIONING="$FM_SELF_CONDITIONING" FM_CLIP_SAMPLE=true \
  SUBSCORE_PATH="$NAVSIM_EXP_ROOT/$REL/$tag/proposals.pkl" \
    bash "$EVAL"

  # 取最新的 csv（重跑会产生多个时间戳子目录，find 的顺序不保证是最新）
  local csv
  csv="$(ls -1t $(find "$NAVSIM_EXP_ROOT/$REL/$tag" -name '*.csv' 2>/dev/null) 2>/dev/null | head -n 1 || true)"
  [ -n "$csv" ] || { echo "[WARN] $tag: 没有评测 csv"; return 0; }
  python3 - "$csv" "$arm" "$ep_tag" "$k" "$ckpt" "$SUMMARY" <<'PY'
import csv, sys
path, arm, ep, k, ckpt, out = sys.argv[1:7]
score = ''
with open(path, newline='', encoding='utf-8', errors='replace') as f:
    for row in csv.DictReader(f):
        tok = (row.get('token') or '').strip()
        if tok in ('extended_pdm_score_stage_one', 'extended_pdm_score_combined'):
            score = (row.get('score') or '').strip(); break
# 同一 (arm, ckpt, K) 只保留最新一行
with open(out, newline='', encoding='utf-8') as f:
    rows = [r for r in csv.reader(f)]
header, rows = rows[0], rows[1:]
rows = [r for r in rows if not (len(r) >= 4 and r[0] == arm and r[1] == ep and r[2] == k)]
rows.append([arm, ep, k, score, ckpt])
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(header); w.writerows(rows)
print(f'[SUMMARY] p1_{arm}\t{ep}\tK={k}\tscore={score}')
PY
}

eval_arm() {
  local arm="$1"
  local exp_dir="$NAVSIM_EXP_ROOT/$REL/p1_$arm"
  if [ "$SKIP_EVAL" = "1" ]; then
    echo "[SKIP_EVAL] $arm"
    return 0
  fi
  local ckpts
  if [ "$DRY_RUN" = "1" ]; then
    ckpts="<${exp_dir}/epoch=XX.ckpt>"
  else
    # 按 epoch 号排序取最后 N 个（ls -t 会被 last.ckpt/重写打乱）
    ckpts="$(ls -1 "$exp_dir"/epoch=*.ckpt 2>/dev/null \
             | sed -E 's/.*epoch=([0-9]+).*/\1 &/' | sort -n | tail -n "$EVAL_LAST_N" | cut -d' ' -f2- || true)"
    [ -n "$ckpts" ] || { echo "[WARN] p1_$arm: 无 epoch ckpt"; return 0; }
  fi
  local ckpt k
  for ckpt in $ckpts; do
    for k in $EVAL_STEPS; do
      eval_one "$arm" "$ckpt" "$k"
    done
  done
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
  # 每臂每个 K 取最后 N 个 ckpt 的均值，再给出 center − base
  python3 - "$SUMMARY" <<'PY'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
agg = {}
for r in rows:
    try:
        agg.setdefault((r['arm'], r['K']), []).append(float(r['score']))
    except (KeyError, ValueError):
        pass
print()
for (arm, k), v in sorted(agg.items()):
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f'p1_{arm:6s} K={k:>3s}  n={len(v)}  mean={st.mean(v):.4f}  std={sd:.4f}')
for k in sorted({k for _, k in agg}):
    if ('base', k) in agg and ('center', k) in agg:
        d = st.mean(agg[('center', k)]) - st.mean(agg[('base', k)])
        print(f'center - base @K={k}: {d:+.4f}  (判据 > +0.005)')
PY
fi
