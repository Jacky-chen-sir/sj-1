#!/usr/bin/env bash
# 第 3 章消融 sweep 总控：按组定义 env 组合，循环调 train_dp_official_fm.sh 训练。
# 覆盖论文 5.4.1（FM-DP 组件消融）、5.4.2（创新点 A/B 消融）。
#
# 用法：
#   bash scripts/dp/ablation/sweep_dp.sh                    # 全量
#   SWEEP_GROUPS="comp_kin comp_sc" bash scripts/dp/ablation/sweep_dp.sh
#   DRY_RUN=1 bash scripts/dp/ablation/sweep_dp.sh          # 只打命令不跑
#   bash scripts/dp/ablation/sweep_dp.sh --list             # 列出全部组名
#
# 组名前缀约定：
#   comp_*   组件消融（5.4.1）——逐项开关 FM-DP 的三项改进
#   guide_*  创新点 A（5.4.2）——评分引导
#   struct_* 创新点 B（5.4.2）——指标结构感知（安全可行性 / 舒适多样性）
#   full     完整配置
#   base     全部关闭的基线
#
# ⚠ 顺序有讲究：`base` 与 `full` 必须先跑，它们是其余所有格子的参照系。
#   组件消融的每一项都是「full 去掉该项」，这样单格差异才可解释。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN="${ROOT_DIR}/scripts/training/train_dp_official_fm.sh"

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
DRY_RUN="${DRY_RUN:-0}"
BASE_EXP="${BASE_EXP:-train_dp_official_fm_joint}"

# 所有格子从同一个起点续训，保证只差被消融的那一项
if [ "${1:-}" = "--list" ]; then
  cat <<'EOF'
组件消融（5.4.1，全部相对 full 逐项关闭）
  base            全关（对照基线）
  full            全开（= 论文完整配置，含 A/B 需另设 SCORER_CKPT）
  comp_no_kin     去掉运动学监督（加速度/jerk）
  comp_no_aux     去掉绝对轨迹辅助损失（漂移 + 航向）
  comp_no_lattice 去掉推理网格 t 采样（回到纯连续均匀采样）
  comp_no_sc      去掉自洽监督
  comp_no_selfcond 去掉自条件输入

创新点 A（5.4.2）
  guide_xs        评分引导权重 0.02（修复门控后的保守档）
  guide_lo        评分引导权重 0.05
  guide_mid       评分引导权重 0.2
  guide_hi        评分引导权重 0.5
  guide_tmin03    引导区间上界 0.3（只覆盖低 t 的可靠段）
  guide_tmin07    引导区间上界 0.7（覆盖到较噪声的一段）

创新点 B（5.4.2）
  struct_feas     只开安全可行性
  struct_div      只开舒适多样性
  struct_both     两者都开
  struct_feas_lo  可行性阈值 0.3
  struct_feas_hi  可行性阈值 0.7
EOF
  exit 0
fi

RESUME_CKPT="${RESUME_CKPT:?必须指定 RESUME_CKPT=<起点 checkpoint>}"
EPOCHS="${EPOCHS:-6}"

# 创新点 A/B 需要第 4 章训好的评分器；只跑组件消融时可以不设
SCORER_CKPT="${SCORER_CKPT:-}"

# 允许只跑指定组（空格分隔）。不能叫 GROUPS——那是 bash 内置只读数组。
SWEEP_GROUPS="${SWEEP_GROUPS:-}"

want() {
  [ -z "${SWEEP_GROUPS}" ] && return 0
  for g in ${SWEEP_GROUPS}; do [ "${g}" = "$1" ] && return 0; done
  return 1
}

run_group() {
  local tag="$1"; shift
  if ! want "${tag}"; then return 0; fi
  local exp="${BASE_EXP}/${tag}"
  echo "=============================================================="
  echo "[GROUP] ${tag}"
  echo "[ENV  ] $*"
  echo "[EXP  ] ${exp}"
  echo "=============================================================="
  if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN: RESUME_CKPT=${RESUME_CKPT} MAX_EPOCHS=${EPOCHS} EXP_NAME=${exp} \\"
    echo "         SCORER_CKPT=${SCORER_CKPT} $* bash ${TRAIN}"
    return 0
  fi
  env "$@" \
    RESUME_CKPT="${RESUME_CKPT}" \
    MAX_EPOCHS="${EPOCHS}" \
    EXP_NAME="${exp}" \
    SCORER_CKPT="${SCORER_CKPT}" \
    bash "${TRAIN}"
}


echo "[INFO] 起点 checkpoint: ${RESUME_CKPT}"
echo "[INFO] 每个格子训练 ${EPOCHS} epoch"
echo "[INFO] 评分器: ${SCORER_CKPT:-<未设置，创新点 A/B 的组会退化为随机评分器>}"
echo

# —— 参照系：先跑这两个 ——
want base && run_group "base" \
  FM_TRAJ_AUX_WEIGHT=0.0 FM_KIN_WEIGHT=0.0 \
  FM_LATTICE_T_PROB=0.0 FM_SELF_CONSISTENCY_P=0.0 FM_SELF_CONDITIONING=false \
  LAMBDA_SCORE=0.0 LAMBDA_SAFE_FEAS=0.0 LAMBDA_COMFORT_DIV=0.0

want full && run_group "full" \
  FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.5 \
  FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=true \
  LAMBDA_SCORE=0.2 LAMBDA_SAFE_FEAS=0.1 LAMBDA_COMFORT_DIV=0.05 \
  SCORER_GUIDANCE=true

# —— 组件消融（5.4.1）：每项都是「full 去掉该项」——
want comp_no_kin && run_group "comp_no_kin" \
  FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.0 \
  FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=true \
  LAMBDA_SCORE=0.2 LAMBDA_SAFE_FEAS=0.1 LAMBDA_COMFORT_DIV=0.05 \
  SCORER_GUIDANCE=true

want comp_no_aux && run_group "comp_no_aux" \
  FM_TRAJ_AUX_WEIGHT=0.0 FM_KIN_WEIGHT=0.5 \
  FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=true \
  LAMBDA_SCORE=0.2 LAMBDA_SAFE_FEAS=0.1 LAMBDA_COMFORT_DIV=0.05 \
  SCORER_GUIDANCE=true

want comp_no_lattice && run_group "comp_no_lattice" \
  FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.5 \
  FM_LATTICE_T_PROB=0.0 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=true \
  LAMBDA_SCORE=0.2 LAMBDA_SAFE_FEAS=0.1 LAMBDA_COMFORT_DIV=0.05 \
  SCORER_GUIDANCE=true

want comp_no_sc && run_group "comp_no_sc" \
  FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.5 \
  FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.0 FM_SELF_CONDITIONING=true \
  LAMBDA_SCORE=0.2 LAMBDA_SAFE_FEAS=0.1 LAMBDA_COMFORT_DIV=0.05 \
  SCORER_GUIDANCE=true

want comp_no_selfcond && run_group "comp_no_selfcond" \
  FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.5 \
  FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=false \
  LAMBDA_SCORE=0.2 LAMBDA_SAFE_FEAS=0.1 LAMBDA_COMFORT_DIV=0.05 \
  SCORER_GUIDANCE=true

# —— 创新点 A：评分引导权重扫描（5.4.2）——
# 权重的量级比其余项更敏感：过大会模式塌缩，过小不起作用，所以先扫三档。
for pair in "guide_xs 0.02" "guide_lo 0.05" "guide_mid 0.2" "guide_hi 0.5"; do
  set -- ${pair}
  want "$1" && run_group "$1" \
    FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.5 \
    FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=true \
    LAMBDA_SCORE="$2" LAMBDA_SAFE_FEAS=0.0 LAMBDA_COMFORT_DIV=0.0 \
    SCORER_GUIDANCE=true
done

# —— 创新点 A：引导时间区间上界（t_min；下界统一用 guidance_t_floor）——
# 注意约定：x_t=(1-t)x1+t·noise，t=0 数据、t=1 噪声。t_min 越大，区间越宽、
# 越会覆盖到 x̂1 不可靠的高 t 段；越小则只在可靠的低 t 段引导。
for pair in "guide_tmin03 0.3" "guide_tmin07 0.7"; do
  set -- ${pair}
  want "$1" && run_group "$1" \
    FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.5 \
    FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=true \
    LAMBDA_SCORE=0.2 LAMBDA_SAFE_FEAS=0.0 LAMBDA_COMFORT_DIV=0.0 \
    SCORER_GUIDANCE=true GUIDANCE_T_MIN="$2"
done

# —— 创新点 B：两条约束的分离与组合 ——
want struct_feas && run_group "struct_feas" \
  FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.5 \
  FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=true \
  LAMBDA_SCORE=0.0 LAMBDA_SAFE_FEAS=0.1 LAMBDA_COMFORT_DIV=0.0 \
  SCORER_GUIDANCE=true

want struct_div && run_group "struct_div" \
  FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.5 \
  FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=true \
  LAMBDA_SCORE=0.0 LAMBDA_SAFE_FEAS=0.0 LAMBDA_COMFORT_DIV=0.05 \
  SCORER_GUIDANCE=true

want struct_both && run_group "struct_both" \
  FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.5 \
  FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=true \
  LAMBDA_SCORE=0.0 LAMBDA_SAFE_FEAS=0.1 LAMBDA_COMFORT_DIV=0.05 \
  SCORER_GUIDANCE=true

for pair in "struct_feas_lo 0.3" "struct_feas_hi 0.7"; do
  set -- ${pair}
  want "$1" && run_group "$1" \
    FM_TRAJ_AUX_WEIGHT=0.5 FM_KIN_WEIGHT=0.5 \
    FM_LATTICE_T_PROB=0.5 FM_SELF_CONSISTENCY_P=0.5 FM_SELF_CONDITIONING=true \
    LAMBDA_SCORE=0.0 LAMBDA_SAFE_FEAS=0.1 LAMBDA_COMFORT_DIV=0.0 \
    SCORER_GUIDANCE=true SAFE_FEAS_MARGIN="$2"
done

echo
echo "=== sweep done ==="
echo "评测：CKPT=<exp>/<tag>/epoch=NN.ckpt bash scripts/evaluation/eval_dp_fm_joint_navtest.sh"
echo "步数扫描：TAG=<tag> CKPT=... bash scripts/evaluation/eval_dp_fm_step_scan.sh"
