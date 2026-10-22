# DP 改进与已有 navtest 分数

本机没有一份「每次改进单独一个分数」的消融表。`scripts/dp/ablation/sweep_dp.sh` 里设计了逐项关掉运动学、轨迹辅助、网格 t、自洽、自条件的格子，这些实验没有跑完，也没有对应 CSV。

下面只列已经评完、能对上 checkpoint 的分数。协议都是 navtest、one-stage fallback、同一套 metric cache。

## 相对官方 DP（0.7305）谁更高

生成器自己出轨迹、`valid=True` 的 navtest 里，高于官方 DP 的只有 **流匹配 epoch 18：0.7380（+0.0075）**。同一条训练的 epoch 11–17 和 19 都在 0.705–0.727，低于 0.7305。加速度辅助没有单独的评测；把它和评价器引导一起续训的 epoch 22 是 0.6369，低于基线。

提案交给官方 AUG 评分器之后，两条我们的提案都远高于「只出 DP 轨迹」的 0.7305，但都略低于官方 AUG 自己的 0.7939：

| 提案 | 评分器 | score |
|---|---|---:|
| 官方 DP | 模型自带（论文 DP） | 0.7305 |
| 我们的 DP epoch 17 | 官方 AUG | 0.7934 |
| 流匹配 epoch 18 | 官方 AUG | 0.7935 |
| 官方权重 | 官方 AUG | 0.7939 |

## 已经评过的

| 方法 | 实际训练里有什么 | score |
|---|---|---:|
| 官方 GTRS-DP | DDPM，论文权重 | **0.7305** |
| 流匹配 epoch 18 | 只把轨迹头改成速度匹配，推理 K=20。这次 Hydra 配置只有 `use_flow_matching=true`，没有 `fm_kin_weight` / `fm_traj_aux_weight` | **0.7380** |
| 同上接到 AUG 评分器 | 词表 + 这条 FM 提案一起选 | **0.7935** |
| 评价器引导续训 epoch 22 | 从 epoch 18 续上，当时代码已含加速度辅助，同时又开了评分引导；时间门控写反 | **0.6369** |

官方 DP CSV：`exp/official_paper_models_navtest/.../dp-navtest-score-only-w8/.../2026.07.25.19.03.51.csv`。

FM epoch 18 CSV：`exp/train_dp_official_fm_joint/navtest-epoch-18-step-81776-fm20-b8-w8/.../2026.09.01.21.16.07.csv`。训练配置：`exp/train_dp_official_fm_joint/2026.08.27.15.46.57/code/hydra/`，override 里没有运动学或轨迹辅助权重。加速度 / jerk、轨迹辅助、自洽、自条件是 `ddf3c3f`（2026-09-12）才进代码的，晚于这次评测（2026-09-01）。

### 官方 DP 对 流匹配 epoch 18

| | 官方 DP | FM ep18 | 差 |
|---|---:|---:|---:|
| score | 0.7305 | 0.7380 | +0.0075 |
| no_at_fault_collisions | 0.9708 | 0.9764 | +0.0056 |
| drivable_area_compliance | 0.9280 | 0.9215 | −0.0065 |
| driving_direction_compliance | 0.9849 | 0.9927 | +0.0078 |
| traffic_light_compliance | 0.9964 | 0.9975 | +0.0011 |
| ego_progress | 0.8686 | 0.8649 | −0.0037 |
| time_to_collision_within_bound | 0.9673 | 0.9719 | +0.0046 |
| lane_keeping | 0.9507 | 0.9519 | +0.0012 |
| history_comfort | 0.9827 | 0.9827 | 0 |
| two_frame_extended_comfort | 0.8284 | 0.8584 | +0.0300 |

这 +0.0075 只对应「DDPM → 流匹配速度损失」。不能算到加速度辅助监督上。

### 同一次流匹配训练的各 epoch

| epoch | score |
|---:|---:|
| 11 | 0.7051 |
| 12 | 0.7081 |
| 13 | 0.7251 |
| 14 | 0.7188 |
| 15 | 0.7266 |
| 16 | 0.7156 |
| 17 | 0.7244 |
| 18 | 0.7380 |
| 19 | 0.7213 |

epoch 10 的评测目录在，CSV 没有写完。

## 写进代码、但没有单独分数的改进

这些都在现在的 `navsim/agents/dp/dp_model.py` 里，默认开着。没有「只加这一项」的 navtest。

| 改进 | 开关 | 做什么 |
|---|---|---|
| 轨迹辅助（漂移 + 航向） | `fm_traj_aux_weight=0.5` | 增量累积误差，航向用 `1−cos Δθ` |
| 加速度 / jerk 辅助 | `fm_kin_weight=0.5` | 增量残差的一阶、二阶差分 |
| 推理网格上采样 t | `fm_lattice_t_prob=0.5`，K 集合 `{2,3,4,5,8,10,20}` | 训练 t 对齐少步推理 |
| x̂1 裁剪 | `fm_clip_sample=true` | 少步外推不再在 cumsum 上放大 |
| 自洽监督 | `fm_self_consistency_p=0.5` | 用模型自己积分出的中间态做剩余速度目标 |
| 自条件 | `fm_self_conditioning=true` | 解码器多吃上一轮 x̂1，输入 32→64 列 |

epoch 22 的 0.6369 同时包含上面这些和写反的评价器引导，不能拆成加速度项的收益。那次分项在 `scripts/dp/runs/ch3_20261014_lambda01/RUN.md`。

---

## 第三章实验总表（截至 2026-10-22）

协议统一：navtest、one-stage fallback、FM 推理 K=20、同一 metric cache。  
补充实验手册：`scripts/dp/runs/ch3_guidance_supplement/RUNBOOK.md`。

### A. 基线与主结果（已定稿）

| ID | 设置 | score | 状态 |
|---|---|---:|---|
| `official_dp` | 论文 GTRS-DP（DDPM） | **0.7305** | 已评 |
| `fm_joint_ep18` | 官方 DP 骨干 → 流匹配速度损失；Hydra 仅 `use_flow_matching=true` | **0.7380** | 已评；自出轨迹峰值 |
| `fm_joint_ep19` | 同上训练曲线下一 epoch | 0.7213 | 已评；过峰 |
| `fm_ep18 + AUG` | FM 提案 + 官方 AUG 评分器选轨 | **0.7935** | 已评；≈ 官方 AUG 0.7939 |
| `guide_bad_ep22` | 从 ep18 续 5 epoch；λ=0.1/0.1/0.05 + 加速度等；**时间门控写反** | **0.6369** | 已评；作废对照 |

时间门控已在 `b2b3d58` 改为区间 `[t_floor, t_min]` + warmup；坏跑不能用来否定引导本身。

### B. Stage 0 续训对照（从 FM ep18 resume，无引导）

目的：排除「再训 5 epoch 本身就会掉分」。  
起点：`epoch=18-step=81776.ckpt`（0.7380）。`SCORER_GUIDANCE=false`，λ 全 0。BS=4。

| 节点 | 相对步数 / ckpt | score | 相对 ep18 |
|---|---|---:|---:|
| +5k | `relstep=005000`（全局 ≈86776） | 0.6462 | −0.0918 |
| +10k | `relstep=010000` | **0.7346** | −0.0034 |
| +15k | `relstep=015000` | 0.7091 | −0.0289 |
| ep22 终 | `epoch=22-step=116208` | **0.7172** | −0.0208 |

汇总 CSV：`exp/ch3_supplement/summary.csv`。  
中间点用 `RelativeStepCheckpoint`（`REL_SAVE_STEPS=5000,10000,15000`）在续训过程中落盘后评测。

**结论（续训对照）**：

- 最终 0.7172 **明显低于** 0.7380，也 **低于** 开 Stage 1 的门槛 0.73（见 RUNBOOK）。
- 曲线「先崩 → 10k 回到接近基线 → 再软掉」，更像 resume / 优化器 / 自条件状态不匹配，而不是引导问题（本 run 未开引导）。
- **因此 Stage 1（`g_xs` / `g_s`）尚未启动**；先换「官方 DP + 重初始化轨迹头、从零训满」的对照。

### C. Stage 0 从零对照（进行中）

| ID | 设置 | 进度 | score |
|---|---|---|---|
| `from0_control` | 加载官方 DP 权重，`reinit_traj_head=true`（unexpected=0，missing=98 仅轨迹头）；**不 resume** 优化器；BS=8；`MAX_EPOCHS=23`；同样 `REL_SAVE_STEPS=5k/10k/15k` | Epoch 0 训练中（约 2026-10-22 14:25 启动） | 尚无 |

日志：`exp/logs/ch3_from0_control_*.log`；产物目录：`exp/ch3_supplement/from0_control/`。

### D. 计划中、尚未跑

| ID | 条件 | 说明 |
|---|---|---|
| `g_xs` | Stage 0 对照 ≥ 0.73 才开 | λ 0.02 / 0.02 / 0.01，区间门控已修 |
| `g_s` | `g_xs` 至少无害后再开 | λ 0.05 / 0.05 / 0.025 |
| Stage 2 消融 | 仅当 Stage 1 有正收益 | warm0 / band03 / n4 / margin03 |

### E. 论文可用证据链（当前）

1. **生成侧**：DDPM 0.7305 → FM 0.7380（+0.0075），归因到流匹配速度损失，不是加速度辅助（后者晚于该 ckpt）。
2. **接 AUG**：FM 提案 0.7935 ≈ 官方 AUG 0.7939，说明提案质量接近官方。
3. **评价器引导**：坏跑 0.6369 已归因到时间门控；**修好后的正收益尚未验证**。续训 λ=0 对照掉到 0.7172，阻断了「在同一 resume 设定下比引导」的归因，改走 from0。

若最终 from0 / 引导标定仍无正收益，第三章退回「生成侧归因」写法，不以引导为创新点主证据。
