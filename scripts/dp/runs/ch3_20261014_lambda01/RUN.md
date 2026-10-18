# 第三章这次实际训练：配置、代码、navtest 结果

结论：跑的是第三章 FM-DP + 评价器引导（创新点 A 和 B 同时打开）。配置和 `第三章README.md` 里的 `full` 不一致。`LAMBDA_SCORE=0.1`，手册 `full` 是 `0.2`。评测协议和手册 5.1 一致。分数下降是这次权重下的结果。

## 1. 代码

| 项 | 值 |
|---|---|
| 仓库 | `/home/ws/navsim_workspace/GTRS_official` |
| 远端 | `github.com:Jacky-chen-sir/sj-1.git` 分支 `sj-opd` |
| 第三章功能提交 | `800764c` feat(dp): 第 3 章创新点 A/B |
| 手册提交 | `8c24dcb` docs: 第三章运行手册 |
| 本次实际训练用的修复 | `7d0f83b` 及之后补记本文件的提交 |

训练入口：`scripts/training/train_dp_official_fm.sh` → `navsim/planning/script/run_training_dense.py`，agent=`gtrs_diffusion_policy`。

相对 `800764c`，为了能续训加的改动：

- `scorer_guidance.py` / `dp_model.py` / `dp_config.py`：按 OPD R34 加载评分器（`agent.model.student.model.` 前缀、resnet34、8192 词表、40 pose），生成器 8 点轨迹插值后再打分，评分器权重放到 GPU。
- `agent_lightning_module.py`：`fm_self_conditioning` 把 `input_emb` 从 32 列扩到 64 列时，同步补齐 Adam 的 `exp_avg` / `exp_avg_sq`，否则第一步就 `32 vs 64`。
- `train_dp_official_fm.sh`：传入评分器词表；ckpt 路径里的 `=` 加引号。
- `dp_agent.py`：step checkpoint 改为每 2000 step、只留最近 3 个，避免写满根分区。

## 2. 和手册怎么对不上

手册 `scripts/dp/ablation/sweep_dp.sh` 的 `full`：

- `LAMBDA_SCORE=0.2`
- `LAMBDA_SAFE_FEAS=0.1`
- `LAMBDA_COMFORT_DIV=0.05`
- `SCORER_GUIDANCE=true`
- 所有格子从同一个起点再训 `EPOCHS=6`

这次实际：

- A 和 B 同时开，但 `LAMBDA_SCORE=0.1`（介于 `guide_lo=0.05` 和 `guide_mid=0.2` 之间，又叠加了 B 的两条约束）
- 起点是已经训完的 FM-DP `epoch=18-step=81776`，`MAX_EPOCHS=23`，最后停在 `epoch=22-step=116209`
- 中途根分区满盘，从 `step-step=094000.ckpt` 用同一组环境变量续完
- `BS=4`（手册默认 8；评分器进训练后 24GB 卡用 8 容易爆显存）
- `NUM_WORKERS=4`（手册默认 8）

其余 FM 开关沿用脚本默认，和手册 4.3 一致：

`FM_TRAJ_AUX_WEIGHT=0.5`，`FM_TRAJ_AUX_POS_WEIGHT=1.0`，`FM_TRAJ_AUX_HEAD_WEIGHT=1.0`，`FM_KIN_WEIGHT=0.5`，`FM_LATTICE_T_PROB=0.5`，`FM_LATTICE_K_SET=[2,3,4,5,8,10,20]`，`FM_CLIP_SAMPLE=true`，`FM_SELF_CONSISTENCY_P=0.5`，`FM_SC_MAX_STEPS=4`，`FM_SC_WARMUP_STEPS=2000`，`FM_SELF_CONDITIONING=true`，`SCORE_TEMPERATURE=1.0`，`SAFE_FEAS_MARGIN=0.5`，`COMFORT_DIV_TARGET=1.0`，`GUIDANCE_T_MIN=0.5`，`GUIDANCE_SAMPLES=2`，`GUIDANCE_EVERY_N=1`，`PRECISION=32`，`LR=1e-4`，`LR_MULT_BACKBONE=0.1`，`BEV_LOSS_WEIGHT=1.0`，`FREEZE_EXCEPT_TRAJ_HEAD=false`，`REINIT_TRAJ_HEAD=false`。

评分器：第 4 章 OPD R34 `9cd57b3` 的 `step-step=013000.ckpt`，词表 `traj_final/test_8192_kmeans.npy`，`freeze=True`，`dual_stream=True`。引导只进训练损失，推理时不跑评分器。

## 3. 训练命令

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

磁盘满之后的续训只改了 `RESUME_CKPT`：

`RESUME_CKPT=.../exp/train_dp_ch3_scorer_guidance/step-step=094000.ckpt`

日志（本机，未入库，约 60MB）：`exp/logs/train_dp_ch3_scorer_guidance_20261014_200216.log`。训练在 2026-10-16 01:57 正常结束，Epoch 22 跑满 8608 step 并完成验证。

## 4. 评测

和手册 5.1、以及当初 FM-DP `epoch=18` 的评测是同一条脚本、同一套缓存、同一种聚合。

```bash
CKPT=/home/ws/navsim_workspace/exp/train_dp_ch3_scorer_guidance/epoch=22-step=116209.ckpt \
EXP_NAME=train_dp_ch3_scorer_guidance/navtest-epoch22-fm20 \
SUBSCORE_PATH=/home/ws/navsim_workspace/exp/train_dp_ch3_scorer_guidance/epoch=22-step=116209_navtest_fm.pkl \
GPU=0 BS=8 WORKERS=8 FM_NUM_INFERENCE_STEPS=20 \
FM_SELF_CONDITIONING=true FM_CLIP_SAMPLE=true \
  bash scripts/evaluation/eval_dp_fm_joint_navtest.sh
```

- 成功 11989，失败 0，`aggregation_mode=one_stage_fallback`
- 逐 token CSV：本目录 `navtest_epoch22_fm20.csv`（汇总行 token 为 `extended_pdm_score_stage_one`）
- 推理 pkl 未入库（约 671MB）：`exp/train_dp_ch3_scorer_guidance/epoch=22-step=116209_navtest_fm.pkl`
- 权重未入库（约 1.3GB）：`exp/train_dp_ch3_scorer_guidance/epoch=22-step=116209.ckpt`

## 5. navtest 分数

| 指标 | FM-DP ep18 基线 | 本次 ep22 |
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

ep18 基线 CSV（本机）：`exp/train_dp_official_fm_joint/navtest-epoch-18-step-81776-fm20-b8-w8/2026.09.01.16.19.40/2026.09.01.21.16.07.csv`。那次评测命令没有传 `FM_SELF_CONDITIONING`（当时权重的 `input_emb` 仍是 32 列）。本次 ep22 必须传 `true`，和训练一致。

---

## 6. 复盘：这次为什么掉分（2026-10-18 排查）

这次结果 **0.7380 → 0.6369** 不是"λ 调大了"那么简单，代码里有确定性 bug。

### 6.1 确定 bug：引导的时间门控写反了

本仓库的流匹配约定是 `x_t = (1−t)·x1 + t·noise`（`dp_model.py:_fm_interpolate`），
推理也是从 `t=1` 积分到 `t=0`（`dp_model.py` 的 `forward`：`t_val = 1.0 - i*dt`）。
所以 **t=0 是数据、t=1 是纯噪声**，一步估计 `x̂1` 只在**低 t 端**可靠。

旧实现（`800764c` 起）是：

```python
wt = (t01 >= t_min).float()     # t_min = 0.5
```

即**只在 t ∈ [0.5, 1] 施加引导**——恰好是最不可靠、`x̂1` 接近噪声的一半步数。
注释写的「t 很小时 x̂1 仍接近噪声」也与实际约定相反。A、B-1、B-2 三项损失共用这个门控，
等于在约 50% 的训练步上往一个已收敛的模型里注入结构化噪声。5 epoch（约 4.3 万步）足以
把 DAC 拉低 9.5 分、EC 拉低 12.5 分。

**修复**：改为区间门控 `t ∈ [guidance_t_floor, guidance_t_min]`（默认 `[0.1, 0.5]`），
下界避开 `∂x̂1/∂v ∝ t → 0` 的梯度消失区。新增纯函数 `guidance_time_weight()` 与单测
`test_guidance_time_weight_is_low_t_not_high_t` 钉住方向。

### 6.2 另外两处已加固

- **评分器加载失败是静默的**：`strict=False` 下，前缀剥错或缺键时只会打一条 warning，
  然后带着**随机权重**跑完训练。现在键完全不匹配 / 缺键 / 多键都会直接报错
  （`scorer_allow_partial_load=false` 为默认）。
- **没有 warmup**：续训已收敛模型时，引导第一步就按全强度施加。新增
  `guidance_warmup_steps`（默认 2000 步线性 ramp）。
- 另修正：`GUIDANCE_SAMPLES>1` 时第 j>0 条提案没有走 self-conditioning，与主前向口径
  不一致，使 B-2 的 `std` 不可比；现已与主前向对齐。

### 6.3 还没排除的混杂因素

`epoch=18` 的 `input_emb` 是 32 列（当时未开自条件），这次开了
`FM_SELF_CONDITIONING=true`，`input_emb` 被零填充到 64 列。零填充使初始前向不变，
但它与引导同时介入，会让归因变模糊。

**下一步必须做的对照**：从同一个 `epoch=18` 续 5 epoch，三个 λ 全置 0、其余开关不变。

```bash
RESUME_CKPT=.../epoch=18-step=81776.ckpt MAX_EPOCHS=23 \
LAMBDA_SCORE=0 LAMBDA_SAFE_FEAS=0 LAMBDA_COMFORT_DIV=0 \
SCORER_GUIDANCE=false EXP_NAME=train_dp_ch3_control5ep \
  bash scripts/training/train_dp_official_fm.sh
```

若对照也掉到 ~0.64，问题在续训/自条件；若对照保持在 ~0.73，则掉分由引导导致，
可用修好的门控 + 小 λ + warmup 重跑验证。
