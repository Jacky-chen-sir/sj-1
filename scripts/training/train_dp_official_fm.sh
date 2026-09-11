#!/usr/bin/env bash
# Fine-tune official GTRS-DP with Flow Matching:
# - load official DP ckpt
# - randomly re-init traj head (FM from scratch)
# - unfreeze perception; backbone uses smaller LR (lr_mult_backbone)
# - small BEV loss weight as regularization
#
# Usage:
#   bash scripts/training/train_dp_official_fm.sh
#   GPU=0,1,2 MAX_EPOCHS=5 BS=24 bash scripts/training/train_dp_official_fm.sh
#   RESUME_CKPT=/path/to/epoch=xx.ckpt MAX_EPOCHS=20 bash ...  # resume: set max_epochs = completed_epoch + 1 + extra_epochs
#   FREEZE_EXCEPT_TRAJ_HEAD=true  # optional: old decoder-only mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Always use this repo. A leftover NAVSIM_DEVKIT_ROOT=.../GTRS imports the wrong DPConfig.
if [ -n "${NAVSIM_DEVKIT_ROOT:-}" ] && [ "${NAVSIM_DEVKIT_ROOT}" != "${ROOT_DIR}" ]; then
  echo "[WARN] Ignoring NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
  echo "[WARN] Using script repo instead: ${ROOT_DIR}"
fi
export NAVSIM_DEVKIT_ROOT="${ROOT_DIR}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/home/ws/navsim_workspace/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
export NAVSIM_TRAJPDM_ROOT="${NAVSIM_TRAJPDM_ROOT:-${OPENSCENE_DATA_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if ! grep -q "freeze_except_traj_head" "${NAVSIM_DEVKIT_ROOT}/navsim/agents/dp/dp_config.py"; then
  echo "[ERROR] ${NAVSIM_DEVKIT_ROOT} is not the sj-dp-fm tree (missing freeze_except_traj_head)." >&2
  exit 1
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if [ -z "${GPU:-}" ]; then
  GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
NUM_GPUS="$(awk -F',' '{print NF}' <<<"${GPU}")"

OFFICIAL_DP_CKPT="/home/ws/navsim_workspace/GTRS_official/data/models/gtrs_dp_model.ckpt"
RESUME_CKPT="${RESUME_CKPT:-}"
VOV_CKPT="${VOV_CKPT:-${OPENSCENE_DATA_ROOT}/models/dd3d_det_final.pth}"

experiment_name="${EXP_NAME:-train_dp_official_fm_joint}"
lr="${LR:-0.0001}"
# Joint (unfrozen) training needs much more activation memory than decoder-only.
# BS=24 OOMs on 24GB GPUs; default to 8 (override with BS=...).
bs="${BS:-8}"
max_epochs="${MAX_EPOCHS:-5}"
num_workers="${NUM_WORKERS:-8}"
persistent_workers="${PERSISTENT_WORKERS:-true}"
# 保持与历史实验一致的 fp32；想提速可显式 PRECISION=bf16-mixed（建议先跑 1-2 epoch 对比 val loss）
precision="${PRECISION:-32}"
# Fresh FM: reinit traj head. Resume Lightning ckpt: keep weights as-is (reinit off).
reinit_traj_head="${REINIT_TRAJ_HEAD:-}"
freeze_except_traj_head="${FREEZE_EXCEPT_TRAJ_HEAD:-false}"
lr_mult_backbone="${LR_MULT_BACKBONE:-0.1}"
bev_loss_weight="${BEV_LOSS_WEIGHT:-1.0}"
# FM 辅助监督 + 训推对齐（全部默认开启；跑 baseline 要显式把权重/概率置 0）
# 辅助损失现在在归一化增量空间、残差除以 t_eff，与 FM 主损失同量纲，所以 1.0 不再是
# 原来那个"实际比主损失大一个数量级"的 1.0；0.5 就是真正的辅助量级。
fm_traj_aux_weight="${FM_TRAJ_AUX_WEIGHT:-0.5}"
fm_traj_aux_pos_weight="${FM_TRAJ_AUX_POS_WEIGHT:-1.0}"
fm_traj_aux_head_weight="${FM_TRAJ_AUX_HEAD_WEIGHT:-1.0}"
# 运动学（加速度/jerk）损失 → comfort 子指标
fm_kin_weight="${FM_KIN_WEIGHT:-0.5}"
fm_lattice_t_prob="${FM_LATTICE_T_PROB:-0.5}"
# 训练采 t 用的 K 集合：评测要扫的每个 K 都必须在这里，否则那个 K 是 OOD
fm_lattice_k_set="${FM_LATTICE_K_SET:-[2,3,4,5,8,10,20]}"
# x̂1 裁剪（DDPM clip_sample 的 FM 对应），少步采样必开
fm_clip_sample="${FM_CLIP_SAMPLE:-true}"
# 自洽监督：模型自己积分到 x̂_t + 动态剩余速度目标
fm_self_consistency_p="${FM_SELF_CONSISTENCY_P:-0.5}"
fm_sc_max_steps="${FM_SC_MAX_STEPS:-4}"
fm_sc_warmup_steps="${FM_SC_WARMUP_STEPS:-2000}"
# 自条件：input_emb 会被零填充扩列，可以直接从旧 ckpt 续训
fm_self_conditioning="${FM_SELF_CONDITIONING:-true}"

if [ -n "${RESUME_CKPT}" ]; then
  if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[ERROR] Resume checkpoint not found: ${RESUME_CKPT}" >&2
    exit 1
  fi
  INIT_CKPT_ARG="agent.checkpoint_path=null"
  RESUME_ARG="++resume_ckpt_path='${RESUME_CKPT}'"
  reinit_traj_head="${reinit_traj_head:-false}"
  echo "[RESUME] from ${RESUME_CKPT} (max_epochs=${max_epochs}, reinit_traj_head=${reinit_traj_head})"
else
  if [ ! -f "${OFFICIAL_DP_CKPT}" ]; then
    echo "[ERROR] Official DP checkpoint not found: ${OFFICIAL_DP_CKPT}" >&2
    exit 1
  fi
  INIT_CKPT_ARG="agent.checkpoint_path=${OFFICIAL_DP_CKPT}"
  RESUME_ARG=""
  reinit_traj_head="${reinit_traj_head:-true}"
  echo "[INIT] official DP ckpt ${OFFICIAL_DP_CKPT} (reinit_traj_head=${reinit_traj_head})"
fi

LOG_DIR="${NAVSIM_EXP_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${NAVSIM_EXP_ROOT}/${experiment_name}"
EMPTY_DP_PREDS="${NAVSIM_EXP_ROOT}/${experiment_name}/empty_dp_preds.pkl"
python - <<PY
import pickle
from pathlib import Path
p = Path("${EMPTY_DP_PREDS}")
if not p.exists():
    pickle.dump({}, open(p, "wb"))
PY
export DP_PREDS="${EMPTY_DP_PREDS}"

cd "${NAVSIM_DEVKIT_ROOT}"
source /home/ws/anaconda3/etc/profile.d/conda.sh
conda activate conda_gtrs

echo "[RUN] official DP -> Flow Matching (joint perception + traj head)"
echo "      GPUs=${GPU} epochs=${max_epochs} bs=${bs} lr=${lr} workers=${num_workers}"
echo "      exp=${experiment_name} reinit_traj_head=${reinit_traj_head}"
echo "      freeze_except_traj_head=${freeze_except_traj_head} lr_mult_backbone=${lr_mult_backbone} bev_loss_weight=${bev_loss_weight}"
echo "      fm_aux=${fm_traj_aux_weight}(pos=${fm_traj_aux_pos_weight},head=${fm_traj_aux_head_weight}) kin=${fm_kin_weight}"
echo "      lattice_t=${fm_lattice_t_prob} k_set=${fm_lattice_k_set} clip=${fm_clip_sample}"
echo "      self_consistency_p=${fm_self_consistency_p}(max_steps=${fm_sc_max_steps},warmup=${fm_sc_warmup_steps}) self_cond=${fm_self_conditioning}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_dense.py" \
  --config-name competition_training \
  trainer.params.num_nodes=1 \
  ++trainer.params.devices="${NUM_GPUS}" \
  ++trainer.params.accelerator=gpu \
  agent=gtrs_diffusion_policy \
  experiment_name="${experiment_name}" \
  train_test_split=navtrain \
  dataloader.params.batch_size="${bs}" \
  ++dataloader.params.num_workers="${num_workers}" \
  ++dataloader.params.pin_memory=true \
  ++dataloader.params.persistent_workers="${persistent_workers}" \
  ~trainer.params.strategy \
  trainer.params.max_epochs="${max_epochs}" \
  trainer.params.precision="${precision}" \
  agent.config.ckpt_path="${experiment_name}" \
  ${INIT_CKPT_ARG} \
  ${RESUME_ARG} \
  agent.lr="${lr}" \
  ++agent.config.use_flow_matching=true \
  ++agent.config.fm_num_inference_steps=20 \
  ++agent.config.freeze_except_traj_head="${freeze_except_traj_head}" \
  ++agent.config.reinit_traj_head="${reinit_traj_head}" \
  ++agent.config.lr_mult_backbone="${lr_mult_backbone}" \
  ++agent.config.bev_loss_weight="${bev_loss_weight}" \
  ++agent.config.fm_traj_aux_weight="${fm_traj_aux_weight}" \
  ++agent.config.fm_traj_aux_pos_weight="${fm_traj_aux_pos_weight}" \
  ++agent.config.fm_traj_aux_head_weight="${fm_traj_aux_head_weight}" \
  ++agent.config.fm_kin_weight="${fm_kin_weight}" \
  ++agent.config.fm_lattice_t_prob="${fm_lattice_t_prob}" \
  ++agent.config.fm_lattice_k_set="${fm_lattice_k_set}" \
  ++agent.config.fm_clip_sample="${fm_clip_sample}" \
  ++agent.config.fm_self_consistency_p="${fm_self_consistency_p}" \
  ++agent.config.fm_sc_max_steps="${fm_sc_max_steps}" \
  ++agent.config.fm_sc_warmup_steps="${fm_sc_warmup_steps}" \
  ++agent.config.fm_self_conditioning="${fm_self_conditioning}" \
  ++agent.config.vov_ckpt="${VOV_CKPT}" \
  cache_path="${CACHE_PATH:-null}" \
  force_cache_computation="${FORCE_CACHE_COMPUTATION:-false}"
