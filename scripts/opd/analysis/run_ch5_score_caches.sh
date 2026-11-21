#!/usr/bin/env bash
# 第 5 章 5.3.4/5.3.5 所需的小规模打分缓存（不训练）。
# GPU1：navtest 30 场景教师缓存（给 fullchain 配 FM 提案）
# GPU2：navtrain 2k 学生 base + OPD 缓存（给 scoring）
set -euo pipefail
ROOT="${NAVSIM_DEVKIT_ROOT:-/home/ws/navsim_workspace/GTRS_official}"
EXP="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
FIG="$EXP/ch5_figs"
cd "$ROOT"

TEACHER_CKPT="${TEACHER_CKPT:-$EXP/model_ckpt/model_ckpt/drivesuprim_vit.ckpt}"
BASE_CKPT="${BASE_CKPT:-$EXP/model_ckpt/model_ckpt/drivesuprim_r34.ckpt}"
OPD_CKPT="${OPD_CKPT:-$EXP/training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256-9cd57b3/epoch=05-step=13000.ckpt}"

TEACH_NAVTEST="$FIG/cache_teacher_navtest30"
BASE_CACHE="$FIG/cache_base_r34_2k"
OPD_CACHE="$FIG/cache_opd_r34_2k"
mkdir -p "$TEACH_NAVTEST" "$BASE_CACHE" "$OPD_CACHE"

run_cache () {
  local gpu="$1" agent="$2" ckpt="$3" out="$4" split="$5" token_list="$6" exp_name="$7" model="${8:-teacher}"
  echo "=== cache $exp_name on GPU$gpu → $out ==="
  CUDA_VISIBLE_DEVICES="$gpu" \
  OPD_TEACHER_SCORE_DIR="$out" \
  OPD_TOKEN_LIST="$token_list" \
  OPD_SKIP_EXISTING=1 \
  OPD_STORE_HEADS=0 \
  python "$ROOT/navsim/planning/script/run_teacher_opd_cache.py" \
    +debug=false \
    agent="$agent" \
    experiment_name="$exp_name" \
    train_test_split="$split" \
    dataloader.params.batch_size=4 \
    dataloader.params.num_workers=2 \
    ~trainer.params.strategy \
    trainer.params.precision=32 \
    ++trainer.params.devices=1 \
    ++trainer.params.accelerator=gpu \
    "agent.checkpoint_path='${ckpt}'" \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model="$model" \
    ++agent.config.opd.safe_fused_score=true \
    ++agent.config.opd.dual_stream_score=true \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.refinement_approach=transformer_decoder \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    +cache_path=null
}

# 1) teacher on navtest 30 (fullchain)
run_cache 1 gtrs_aug_drivesuprim_vit "$TEACHER_CKPT" "$TEACH_NAVTEST" \
  navtest "$FIG/navtest_tokens_30.txt" ch5_teacher_navtest30 teacher

# 2) students on navtrain 2k (scoring) — sequential on GPU2
run_cache 2 gtrs_aug_drivesuprim_r34 "$BASE_CKPT" "$BASE_CACHE" \
  navtrain "$FIG/navtrain_tokens_2000.txt" ch5_base_r34_2k student

run_cache 2 gtrs_aug_opd_r34 "$OPD_CKPT" "$OPD_CACHE" \
  navtrain "$FIG/navtrain_tokens_2000.txt" ch5_opd_r34_2k student

echo "DONE caches"
ls "$TEACH_NAVTEST" | wc -l
ls "$BASE_CACHE" | wc -l
ls "$OPD_CACHE" | wc -l
