#!/usr/bin/env bash
# Evaluate all FM joint DP checkpoints on navtest, newest-first (reverse epoch order).
# Usage:
#   bash scripts/evaluation/eval_dp_fm_joint_navtest_all.sh
#   GPU=0 BS=8 WORKERS=8 bash scripts/evaluation/eval_dp_fm_joint_navtest_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
LOG_DIR="${NAVSIM_EXP_ROOT}/logs"
mkdir -p "${LOG_DIR}"

gpu="${GPU:-0}"
workers="${WORKERS:-8}"
batch_size="${BS:-${BATCH_SIZE:-8}}"

ckpts=($(ls -1t "${NAVSIM_EXP_ROOT}/train_dp_official_fm_joint"/epoch=*.ckpt 2>/dev/null || true))
if [ "${#ckpts[@]}" -eq 0 ]; then
  echo "[ERROR] No checkpoints found in ${NAVSIM_EXP_ROOT}/train_dp_official_fm_joint" >&2
  exit 1
fi

echo "[INFO] Evaluating ${#ckpts[@]} checkpoints (newest first) on GPU ${gpu}"
for ckpt in "${ckpts[@]}"; do
  echo "[QUEUE] ${ckpt}"
  GPU="${gpu}" BS="${batch_size}" WORKERS="${workers}" CKPT="${ckpt}" \
    bash "${SCRIPT_DIR}/eval_dp_fm_joint_navtest.sh"
done

echo "[DONE] All evaluations finished"
