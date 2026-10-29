#!/bin/bash
# 第三章：对已训练好的 ckpt 补评测（不训练）。Phase 1 补充对照、Phase 2 选点都用它。
#
# 每个 (臂, ckpt, 推断方式, K) 组合各评一次，汇总为 mean±std 与臂间差值。
#
#   推断方式 INFER：
#     zero   零噪声路径（论文主口径；num_proposals=1）
#     first  单样本随机取第 1 条（num_proposals=1，seed 固定）
#   为什么要评 first：中心一致性损失直接优化的就是 zero 路径，只报 zero 会被质疑
#   “只是让模型对齐了评测规则”。first 也涨，才能说明整个分布变好了。
#
# 用法（服务器上；不要和训练抢同一张卡）：
#   # Phase 1 补充：两臂最后 3 个 epoch × {zero, first} × K=20
#   ARMS="p1_base p1_center" bash scripts/dp/runs/ch3_df_model/eval_ckpts.sh
#
#   # Phase 2 选点：中间 ckpt（训练中也可评已落盘的 epoch，建议用空闲卡）
#   ARMS="p2_center18" EPOCHS_LIST="8 12 15 17" INFER="zero" GPU=0 \
#     bash scripts/dp/runs/ch3_df_model/eval_ckpts.sh
#
#   DRY_RUN=1 ...   只打印命令
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EVAL="$ROOT_DIR/scripts/evaluation/eval_dp_fm_joint_navtest.sh"

NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
REL="${REL:-ch3_df}"
ARMS="${ARMS:-p1_base p1_center}"
LAST_N="${LAST_N:-3}"                  # 未给 EPOCHS_LIST 时，取每臂最后 N 个 epoch
EPOCHS_LIST="${EPOCHS_LIST:-}"         # 显式指定 epoch 号，如 "8 12 15 17"
INFER="${INFER:-zero first}"
K_LIST="${K_LIST:-20}"
FM_SELF_CONDITIONING="${FM_SELF_CONDITIONING:-true}"   # 与 P1/P2 训练口径一致
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY="${SUMMARY:-$NAVSIM_EXP_ROOT/$REL/summary_eval_ckpts.csv}"

say() { echo; echo "=============================================================="; echo "$*"; echo "=============================================================="; }

if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$(dirname "$SUMMARY")"
  [ -f "$SUMMARY" ] || echo "arm,epoch,infer,K,score,ckpt" > "$SUMMARY"
fi

pick_ckpts() {
  local dir="$1"
  if [ "$DRY_RUN" = "1" ] && [ ! -d "$dir" ]; then
    if [ -n "$EPOCHS_LIST" ]; then
      for e in $EPOCHS_LIST; do printf '%s/epoch=%02d-step=XX.ckpt\n' "$dir" "$e"; done
    else
      echo "$dir/epoch=XX-step=XX.ckpt"
    fi
    return
  fi
  if [ -n "$EPOCHS_LIST" ]; then
    for e in $EPOCHS_LIST; do
      ls -1 "$dir"/epoch=$(printf '%02d' "$e")-*.ckpt 2>/dev/null | head -n 1 || true
    done
  else
    ls -1 "$dir"/epoch=*.ckpt 2>/dev/null \
      | sed -E 's/.*epoch=([0-9]+).*/\1 &/' | sort -n | tail -n "$LAST_N" | cut -d' ' -f2-
  fi
}

eval_one() {
  local arm="$1" ckpt="$2" infer="$3" k="$4"
  local ep; ep="$(basename "$ckpt" .ckpt)"; ep="${ep%%-step*}"; ep="${ep//=/}"
  local zero=false seed=1001
  [ "$infer" = "zero" ] && { zero=true; seed=3000; }
  local tag="eval-${arm}-${ep}-${infer}-k${k}"
  say "[EVAL] $arm $ep infer=$infer K=$k"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: CKPT=$ckpt EXP_NAME=$REL/$tag DP_EVAL_ZERO_NOISE=$zero DP_PROPOSAL_AGG=first \\"
    echo "         NUM_PROPOSALS=1 SEED=$seed FM_NUM_INFERENCE_STEPS=$k FM_SELF_CONDITIONING=$FM_SELF_CONDITIONING GPU=$GPU"
    return 0
  fi
  [ -f "$ckpt" ] || { echo "[WARN] missing $ckpt"; return 0; }
  mkdir -p "$NAVSIM_EXP_ROOT/$REL/$tag"
  CKPT="$ckpt" EXP_NAME="$REL/$tag" GPU="$GPU" BS=8 WORKERS=8 FORCE_RERUN=1 \
  DP_EVAL_ZERO_NOISE="$zero" DP_PROPOSAL_AGG=first NUM_PROPOSALS=1 SEED="$seed" \
  FM_NUM_INFERENCE_STEPS="$k" FM_SELF_CONDITIONING="$FM_SELF_CONDITIONING" FM_CLIP_SAMPLE=true \
  SUBSCORE_PATH="$NAVSIM_EXP_ROOT/$REL/$tag/proposals.pkl" \
    bash "$EVAL"
  # 按修改时间取最新 csv；同一 (arm, epoch, infer, K) 只保留一行
  local csv
  csv="$(find "$NAVSIM_EXP_ROOT/$REL/$tag" -name '*.csv' -print0 2>/dev/null \
         | xargs -0 ls -1t 2>/dev/null | head -n 1 || true)"
  [ -n "$csv" ] || { echo "[WARN] $tag 无 csv"; return 0; }
  python3 - "$csv" "$arm" "$ep" "$infer" "$k" "$ckpt" "$SUMMARY" <<'PY'
import csv, sys
path, arm, ep, infer, k, ckpt, out = sys.argv[1:8]
score = ''
with open(path, newline='', encoding='utf-8', errors='replace') as f:
    for r in csv.DictReader(f):
        if (r.get('token') or '').strip() in ('extended_pdm_score_stage_one', 'extended_pdm_score_combined'):
            score = (r.get('score') or '').strip(); break
rows = list(csv.reader(open(out, encoding='utf-8')))
head, rows = rows[0], [r for r in rows[1:] if r[:4] != [arm, ep, infer, k]]
rows.append([arm, ep, infer, k, score, ckpt])
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(head); w.writerows(rows)
print(f'[SUMMARY] {arm} {ep} {infer} K={k} score={score}')
PY
}

for arm in $ARMS; do
  for ckpt in $(pick_ckpts "$NAVSIM_EXP_ROOT/$REL/$arm"); do
    for infer in $INFER; do
      for k in $K_LIST; do
        eval_one "$arm" "$ckpt" "$infer" "$k"
      done
    done
  done
done

say "[DONE] summary → $SUMMARY"
if [ "$DRY_RUN" != "1" ] && [ -f "$SUMMARY" ]; then
  python3 - "$SUMMARY" <<'PY'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
agg = {}
for r in rows:
    try:
        agg.setdefault((r['arm'], r['infer'], r['K']), []).append(float(r['score']))
    except ValueError:
        pass
for (arm, inf, k), v in sorted(agg.items()):
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f'{arm:14s} {inf:5s} K={k:>2s}  n={len(v)}  mean={st.mean(v):.4f}  std={sd:.4f}  max={max(v):.4f}')
for inf in sorted({i for _, i, _ in agg}):
    for k in sorted({kk for _, _, kk in agg}):
        b, c = agg.get(('p1_base', inf, k)), agg.get(('p1_center', inf, k))
        if b and c:
            print(f'p1_center - p1_base  [{inf}, K={k}]: {st.mean(c) - st.mean(b):+.4f}')
PY
fi
