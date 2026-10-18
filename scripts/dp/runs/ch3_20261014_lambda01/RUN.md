# 第三章实际跑过的一次（与手册 `full` 不一致）

仓库：`GTRS_official`，分支 `sj-opd`，基线提交 `800764c`，另加本次提交里的本地修复后才训完。
最终权重（未入库，约 1.3GB）：`exp/train_dp_ch3_scorer_guidance/epoch=22-step=116209.ckpt`
推理 pkl（未入库，约 671MB）：`exp/train_dp_ch3_scorer_guidance/epoch=22-step=116209_navtest_fm.pkl`
逐 token CSV：同目录 `navtest_epoch22_fm20.csv`

## 和手册的差别

手册 `full` / 4.4 示例是 `LAMBDA_SCORE=0.2`，每个格子 `EPOCHS=6`。
这次是 **A+B 同时开，但 `LAMBDA_SCORE=0.1`**，从 FM-DP `epoch=18` 续到 `epoch=22`（`MAX_EPOCHS=23`）。
评测协议与手册 5.1 一致（navtest、K=20、自条件、裁剪、one-stage）。

对照同一评测协议下的 FM-DP 基线 `epoch=18`：score **0.738 → 0.637**。掉分主要在 DAC 与 two-frame comfort。

## 训练命令

```bash
cd /home/ws/navsim_workspace/GTRS_official
GPU=0,1,2 BS=4 NUM_WORKERS=4 MAX_EPOCHS=23 \
  EXP_NAME=train_dp_ch3_scorer_guidance \
  RESUME_CKPT=/home/ws/navsim_workspace/exp/train_dp_official_fm_joint/epoch=18-step=81776.ckpt \
  SCORER_GUIDANCE=true \
  SCORER_CKPT=/home/ws/navsim_workspace/exp/training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256-9cd57b3/step-step=013000.ckpt \
  SCORER_VOCAB_PATH=/home/ws/navsim_workspace/GTRS_official/traj_final/test_8192_kmeans.npy \
  LAMBDA_SCORE=0.1 LAMBDA_SAFE_FEAS=0.1 LAMBDA_COMFORT_DIV=0.05 \
  GUIDANCE_SAMPLES=2 GUIDANCE_T_MIN=0.5 \
  bash scripts/training/train_dp_official_fm.sh
```

磁盘写满后从 `step-step=094000.ckpt` 用同一组环境变量续到结束。

未改、沿用脚本默认：`FM_TRAJ_AUX_WEIGHT=0.5`，`FM_KIN_WEIGHT=0.5`，`FM_LATTICE_T_PROB=0.5`，`FM_LATTICE_K_SET=[2,3,4,5,8,10,20]`，`FM_CLIP_SAMPLE=true`，`FM_SELF_CONSISTENCY_P=0.5`，`FM_SC_MAX_STEPS=4`，`FM_SC_WARMUP_STEPS=2000`，`FM_SELF_CONDITIONING=true`，`SCORE_TEMPERATURE=1.0`，`SAFE_FEAS_MARGIN=0.5`，`COMFORT_DIV_TARGET=1.0`，`GUIDANCE_EVERY_N=1`，`PRECISION=32`，`LR=1e-4`。

## 评测命令

```bash
CKPT=exp/train_dp_ch3_scorer_guidance/epoch=22-step=116209.ckpt \
EXP_NAME=train_dp_ch3_scorer_guidance/navtest-epoch22-fm20 \
GPU=0 BS=8 WORKERS=8 FM_NUM_INFERENCE_STEPS=20 \
FM_SELF_CONDITIONING=true FM_CLIP_SAMPLE=true \
  bash scripts/evaluation/eval_dp_fm_joint_navtest.sh
```

## navtest 分数（one_stage_fallback，11989 场景，0 失败）

| | ep18 基线 | 本次 ep22 |
|---|---:|---:|
| score | 0.7380 | 0.6369 |
| no_at_fault_collisions | 0.9764 | 0.9585 |
| drivable_area_compliance | 0.9215 | 0.8268 |
| driving_direction_compliance | 0.9927 | 0.9815 |
| traffic_light_compliance | 0.9975 | 0.9977 |
| ego_progress | 0.8649 | 0.8644 |
| time_to_collision_within_bound | 0.9719 | 0.9500 |
| lane_keeping | 0.9519 | 0.9399 |
| history_comfort | 0.9827 | 0.9823 |
| two_frame_extended_comfort | 0.8584 | 0.7331 |
