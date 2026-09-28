#!/bin/bash
# OPD 蒸馏学生评估。与 eval_epoch.sh 的三点区别：
#   1. 按文件名 glob ckpt，不算 step=$((epoch*1330)) —— NPROC=3 时 step 数不对，1330 是 8GPU×BS8 的步数。
#   2. inference.model 默认 teacher（= 学生的 EMA 副本，与官方 base 的评测口径一致）。
#      无 EMA 副本的旧 ckpt（ema_eval/ema_soft_label 都关、或本次改动前训的）用 INFER_MODEL=student；
#      即使忘了传，agent.initialize() 发现 ckpt 无 teacher.* 也会大声 warn 并回落 student。
#   3. BS 默认 4（按显存预算），不沿用 8。
#
# 用法：
#   bash scripts/opd/evaluation/eval_opd.sh <epoch> <dir> [num_refinement_stage] [stage_layers] [topks]
#   bash scripts/opd/evaluation/eval_opd.sh 5 training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256 1 3 256
set -euo pipefail

epoch=$1
dir=$2
num_refinement_stage=${3:-1}
stage_layers=${4:-3}
topks=${5:-256}
agent=${6:-gtrs_aug_opd_r34}
# teacher = EMA 副本；student = 原始权重。两个都评一次可以直接看出 EMA 的增益。
inference_model=${INFER_MODEL:-teacher}

padded_epoch=$(printf "%02d" "$epoch")
metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/test/ori"
nproc=${NPROC:-3}
master_port=${MASTER_PORT:-29530}
bs=${BS:-4}

# 按文件名 glob ckpt（不依赖 steps/epoch 公式）
ckpt=""
for f in "${NAVSIM_EXP_ROOT}/${dir}/epoch=${padded_epoch}"-step=*.ckpt; do
  if [ -f "$f" ]; then ckpt="$f"; break; fi
done
if [ -z "$ckpt" ]; then
  # 兜底：目录里按 mtime 最新的任意 epoch-*.ckpt（如 resume 中断后命名变化）
  ckpt=$(ls -t "${NAVSIM_EXP_ROOT}/${dir}"/epoch="${padded_epoch}"*.ckpt 2>/dev/null | head -n1 || true)
fi
if [ -z "$ckpt" ]; then
  echo "Error: no checkpoint matching epoch=${padded_epoch} in ${NAVSIM_EXP_ROOT}/${dir}" >&2
  echo "可用 ckpt：" >&2
  ls -1 "${NAVSIM_EXP_ROOT}/${dir}"/*.ckpt 2>/dev/null | sed 's/^/  /' >&2 || true
  exit 1
fi

experiment_name="${dir}/test-${padded_epoch}ep-${inference_model}"

command_string="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2_aug.py \
    +debug=false \
    agent=$agent \
    train_test_split=navtest \
    dataloader.params.batch_size=$bs \
    worker.threads_per_node=128 \
    agent.checkpoint_path='${ckpt}' \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=${inference_model} \
    agent.config.inference.save_pickle=false \
    agent.config.lab.save_pickle=false \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.refinement_approach=transformer_decoder \
    agent.config.refinement.num_refinement_stage=$num_refinement_stage \
    agent.config.refinement.stage_layers=$stage_layers \
    agent.config.refinement.topks=$topks \
    experiment_name=${experiment_name} \
    +cache_path=null \
    metric_cache_path=${metric_cache_path}
"

echo "=== OPD eval ==="
echo "  ckpt : $ckpt"
echo "  exp  : $experiment_name"
echo "--- COMMAND ---"
echo "$command_string"
echo

torchrun --nproc_per_node=$nproc --master_port=$master_port $command_string
