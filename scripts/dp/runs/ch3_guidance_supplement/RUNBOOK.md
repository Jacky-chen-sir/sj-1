# 第三章补充实验：评价器引导的归因与标定

本目录是第三章「评价器引导的生成」的补充实验清单与运行脚本。目标只有一个：
**在修好时间门控之后，判断引导到底有没有正收益；如果有，给出可写进论文的权重。**

## 0. 背景（为什么要重跑）

2026-10-14 那次 5-epoch 续训（`scripts/dp/runs/ch3_20261014_lambda01/`）从 **0.7380 掉到 0.6369**。
排查确认主因是时间门控写反（`t >= t_min`，恰好只给最不可靠的一半步数打分），
已在 `b2b3d58` 修正为区间门控 `[t_floor, t_min]` 并加 warmup。
但修正后**尚未验证**，且上一轮**缺少 λ=0 对照**，无法排除「续训本身就会掉分」。

因此本实验的第一目的不是调参，而是**建立可归因的对照**。

## 1. 统一设置

| 项 | 值 |
|---|---|
| 起点 checkpoint | FM-DP `epoch=18-step=81776.ckpt`（navtest 0.7380） |
| 训练轮数 | `MAX_EPOCHS=23`（即再训 5 个 epoch，与上一轮一致） |
| batch / workers | `BS=4` / `NUM_WORKERS=4`（评分器进训练后 24 GB 卡用 8 会 OOM） |
| 精度 | fp32 |
| 评分器 | OPD R34 `step-step=013000.ckpt`，8192 词表，`freeze=true`，`dual_stream=true` |
| 评测 | `eval_dp_fm_joint_navtest.sh`，`K=20`，同一 metric cache |
| 时间区间 | `GUIDANCE_T_FLOOR=0.1`，`GUIDANCE_T_MIN=0.5` |
| warmup | `GUIDANCE_WARMUP_STEPS=2000` |
| 提案数 | `GUIDANCE_SAMPLES=2` |

> ⚠ 所有 run 的 FM 开关必须完全一致（默认全开），唯一变量是引导。

## 2. 实验矩阵

### Stage 0 — 对照（必跑）

| ID | 目的 | 关键 env |
|---|---|---|
| `control` | 只续训、不开引导。判定「5 epoch 续训本身是否掉分」 | `SCORER_GUIDANCE=false`，三个 λ 全 0 |

**判据**：control 的 navtest 若明显低于 0.7380（例如 < 0.73），说明掉分与引导无关，
要先查自条件/优化器续训，不要继续做引导实验。

### Stage 1 — 权重标定（必跑）

| ID | λ_score / λ_safe / λ_div | 目的 |
|---|---|---|
| `g_xs` | 0.02 / 0.02 / 0.01 | 保守档：验证「方向修正后是否至少不伤」 |
| `g_s`  | 0.05 / 0.05 / 0.025 | 中间档：若保守档有效，找更优点 |

上一轮用的 0.1 / 0.1 / 0.05 已确认会掉分，**不再作为主配置**，只作为上界参考。

**判据**：
- `g_xs` ≥ `control`（差值在噪声内）→ 引导至少无害，进入 Stage 2 找增益。
- `g_xs` > `control` + 0.003 → 引导有正收益，可写入论文。
- `control` 与 `g_xs` 都 < 0.73 → 问题不在引导，停止本组实验。

### Stage 2 — 机理消融（仅当 Stage 1 出现正收益时跑）

| ID | 变量 | 目的 |
|---|---|---|
| `warm0` | `GUIDANCE_WARMUP_STEPS=0` | warmup 是否必要 |
| `band03` | `GUIDANCE_T_MIN=0.3` | 更窄的可靠区间是否更好 |
| `n4` | `GUIDANCE_SAMPLES=4` | B-2 的 std 更稳 |
| `margin03` | `SAFE_FEAS_MARGIN=0.3` | 安全 hinge 是否过于激进 |

## 3. 运行

```bash
cd /path/to/sj-1
export START_CKPT=$NAVSIM_EXP_ROOT/train_dp_official_fm_joint/epoch=18-step=81776.ckpt
export SCORER_CKPT=$NAVSIM_EXP_ROOT/training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256-9cd57b3/step-step=013000.ckpt

# 先 dry-run 看命令
STAGE=0 DRY_RUN=1 bash scripts/dp/runs/ch3_guidance_supplement/run_all.sh

# Stage 0 → Stage 1
STAGE=0 bash scripts/dp/runs/ch3_guidance_supplement/run_all.sh
STAGE=1 bash scripts/dp/runs/ch3_guidance_supplement/run_all.sh

# 视结果决定是否跑 Stage 2
STAGE=2 bash scripts/dp/runs/ch3_guidance_supplement/run_all.sh
```

产物：
- 训练：`$NAVSIM_EXP_ROOT/ch3_supplement/<ID>/`
- 评测：`$NAVSIM_EXP_ROOT/ch3_supplement/navtest-<ID>/`
- 汇总：`$NAVSIM_EXP_ROOT/ch3_supplement/summary.csv`

## 4. 每一步都要先看健康度

训练启动后立刻跑（`fm_loss_base` 等字段需要 `86954be` 之后的代码）：

```bash
python scripts/dp/analysis/guidance_health.py exp/logs/<tag>_*.log
```

| 信号 | 处理 |
|---|---|
| `L_score / base > 1` | 立即停，λ 减半 |
| `fm_safety_prod` 长期 < 0.2 | `SAFE_FEAS_MARGIN` 降到 0.2 |
| `fm_guidance_frac ≈ 0` | 区间配置错误 |
| `fm_guidance_warm` 长期很小 | warmup 太长 |

## 5. 成本与顺序

单次 5-epoch 训练在 3×3090 上约 1–2 天（上一轮含一次磁盘满中断，约 30 h）。
- Stage 0 + Stage 1 = 3 次训练（约 3–6 天）。
- Stage 2 是可选的 4 次。

**不要一次排满**：先跑 `control`，它的结果决定后面是否值得跑。

## 6. 论文里怎么用

- `control` 与 `ref(ep18)` 的差 → 说明续训本身的影响。
- `g_xs` 与 `control` 的差 → **引导的净收益**（这是创新点 A 的唯一合法证据）。
- `warm0` 与 `g_xs` 的差 → warmup 的必要性。
- `band03` 与 `g_xs` 的差 → 时间区间设计的依据。

若 `g_xs` 无正收益，第三章退回「生成侧归因分析」的写法，
把 0.7305 → 0.7380 → 0.7935 这条归因链作为实证结论。

## 7. 运行顺序（先读这一节）

### 7.1 要不要先重跑 `sweep_dp.sh` 里的 `base`？

**不需要。** 两者回答的是不同问题：

| 格子 | 配置 | 回答的问题 |
|---|---|---|
| `sweep_dp.sh: base` | **所有** FM 改进关掉 | 「结构化改进作为一组有没有用」 |
| `sweep_dp.sh: full_noguide` | 改进全开、**引导关** | 组件消融的正确参照系 |
| 本目录 `control` | 从 ep18 续训，改进全开、**引导关** | 引导实验的正确参照系 |

引导实验的基线是 ep18（0.7380）+ `control`，不是 `base`。
若先跑 `base`，你得到的只是「关掉全部改进后掉到多少」，对判断引导没有帮助。

### 7.2 严格按这个顺序

```
① git pull                    # 到最新代码（含诊断与校验）
② STAGE=0  → control          # 必跑，1 次训练
③ 判读 control
     ├─ ≥ 0.73 → 继续
     └─ < 0.73 → 停：查续训/自条件，别做引导
④ STAGE=1  → g_xs, g_s        # 2 次训练
⑤ 判读：g_xs 相对 control 的净差
     ├─ > +0.003 → 引导有正收益，进入 ⑥
     └─ ≤ 0     → 引导仍无收益，转「生成—评价归因」写法
⑥ STAGE=2  → warm0/band03/n4/margin03   # 仅当 ⑤ 为正
⑦ 组件消融（支撑「结构化监督」贡献）
     SWEEP_GROUPS="full_noguide comp_no_kin comp_no_aux comp_no_lattice comp_no_sc comp_no_selfcond"
⑧ 步数扫描（最便宜，只需评测，不需训练）
     CKPT=<最终 ckpt> TAG=fm_dp bash scripts/evaluation/eval_dp_fm_step_scan.sh
```

### 7.3 每一步都先看健康度

```bash
python scripts/dp/analysis/guidance_health.py exp/logs/<tag>_*.log
```

Stage 0 没有引导，这条只对 Stage 1/2 有意义。

### 7.4 为什么不能把 `full` 当成组件的参照系

旧版 `sweep_dp.sh` 的 `comp_*` 组继承了 `full` 的 λ=0.2/0.1/0.05 引导。
那样单格差异会同时包含「去掉该项」和「带着一个会掉分的引导」两件事，无法归因。
现已改为相对 `full_noguide`（引导关闭）。
