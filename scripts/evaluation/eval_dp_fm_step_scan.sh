#!/usr/bin/env bash
# 第 3 章 5.2.1 节：推理步数扫描（精度 vs 步数）
#
# 用途：对**同一个** checkpoint，用不同推理步数 K 分别评测，得到「精度—步数」曲线。
# 论文要论证的是：FM-DP 在少步（K=20）下即可达到扩散基线在 K=100 时的精度水平，
# 即同等精度下推理开销降至约 1/5。这条曲线是该论证的直接证据。
#
# 注意：扫描的每个 K 都必须在训练时的 `fm_lattice_k_set` 里出现过
# （默认 [2,3,4,5,8,10,20]），否则那个 K 对模型是 OOD，曲线不可信。
# 若要用本脚本扫 50 / 100，需先把训练脚本的 FM_LATTICE_K_SET 加上这两个值并重训。
#
# 用法：
#   CKPT=/exp/train_dp_official_fm_joint/epoch=27.ckpt bash scripts/evaluation/eval_dp_fm_step_scan.sh
#   CKPT=... FM_STEPS_LIST="1 2 5 10 20" GPU=0 BS=8 bash ...
#   TAG=ddpm CKPT=<官方 DP ckpt> USE_FM=false bash ...   # 扩散基线（对照组）
#
# 产出：每个 K 一个实验目录，最终由 collect_dp_step_scan.py 汇总成 CSV。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"

ckpt="${CKPT:?必须指定 CKPT=<checkpoint 路径>}"
# 默认覆盖「训练时网格里出现过」的全部档位 + 两端
steps_list="${FM_STEPS_LIST:-1 2 3 5 8 10 20}"
# 曲线上的方法名，用于汇总时区分 FM-DP 与扩散基线
tag="${TAG:-fm_dp}"
use_fm="${USE_FM:-true}"

if [ ! -f "${ckpt}" ]; then
  echo "[ERROR] checkpoint 不存在: ${ckpt}" >&2
  exit 1
fi

echo "[INFO] 步数扫描 tag=${tag} use_fm=${use_fm}"
echo "[INFO] checkpoint: ${ckpt}"
echo "[INFO] 步数档位 : ${steps_list}"
echo

for k in ${steps_list}; do
  echo "=============================================================="
  echo "[SCAN] K=${k}"
  echo "=============================================================="
  CKPT="${ckpt}" \
  FM_NUM_INFERENCE_STEPS="${k}" \
  EXP_NAME="${tag}/stepscan-${tag}-K${k}" \
  USE_FM="${use_fm}" \
    bash "${SCRIPT_DIR}/eval_dp_fm_joint_navtest.sh"
done

echo
echo "[DONE] 全部档位评测完成。下一步汇总："
echo "  python scripts/evaluation/collect_dp_step_scan.py \\"
echo "      --root ${NAVSIM_EXP_ROOT}/${tag} --tag ${tag} \\"
echo "      --out ${NAVSIM_EXP_ROOT}/analysis/dp_step_scan_${tag}.csv"
