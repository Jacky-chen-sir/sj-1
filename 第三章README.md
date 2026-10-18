# 第三章 README：基于 Flow Matching 的轨迹生成（FM-DP）

本文件是硕士论文**第三章**的代码运行手册，覆盖：训练、评测、步数扫描、组件消融与创新点 A/B 消融。

对应论文章节：

| 论文小节 | 代码/开关 |
|---|---|
| 3.2 轨迹的参数化表示 | 增量空间参数化（160 维，`(sinθ, cosθ)` 编码） |
| 3.3 Flow Matching 理论基础 | OT 条件路径、边界修正（`fm_sigma_min` / `fm_clip_sample`） |
| 3.4 基于改进 FM 的轨迹生成器 | `navsim/agents/dp/` |
| 3.4.5 自洽监督 | `fm_self_consistency_p` / `fm_sc_max_steps` / `fm_sc_warmup_steps` |
| 3.4.6 自条件输入 | `fm_self_conditioning` / `fm_self_cond_p` |
| 3.4.7 评价器引导的采样（创新点 A） | `scorer_guidance` / `lambda_score` / `guidance_t_min` |
| 3.5 指标结构感知的生成（创新点 B） | `lambda_safe_feas` / `lambda_comfort_div` |
| 3.6 运动学监督（加速度 / jerk） | `fm_kin_weight` |
| 3.7 损失设计 | `fm_traj_aux_weight` / `fm_traj_aux_pos_weight` / `fm_traj_aux_head_weight` |

---

## 1. 文件一览

```
scripts/training/train_dp_official_fm.sh        训练入口（官方 GTRS-DP → FM-DP）
scripts/evaluation/eval_dp_fm_joint_navtest.sh  单 checkpoint 评测（navtest）
scripts/evaluation/eval_dp_fm_step_scan.sh      推理步数扫描（同一 ckpt × 多档 K）
scripts/dp/ablation/sweep_dp.sh                 组件消融 + 创新点 A/B 消融总控

navsim/agents/dp/dp_config.py                   所有 FM 与引导开关的默认值
navsim/agents/dp/dp_model.py                    FM 损失、采样、引导挂载
navsim/agents/dp/scorer_guidance.py             创新点 A/B 的引导模块（新增）
navsim/agents/dp/tests/test_scorer_guidance.py  引导模块的远程单测
```

---

## 2. 前置条件

脚本内**硬编码**了以下路径，跑之前先确认；不一致就改脚本开头或覆盖环境变量：

```bash
source /home/ws/anaconda3/etc/profile.d/conda.sh
conda activate conda_gtrs                  # 训练/评测脚本内部写死

export OPENSCENE_DATA_ROOT=/home/ws/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/ws/navsim_workspace/exp
export NAVSIM_TRAJPDM_ROOT=${OPENSCENE_DATA_ROOT}

# 从头训练（不 resume）时还需官方 DP checkpoint：
ls /home/ws/navsim_workspace/GTRS_official/data/models/gtrs_dp_model.ckpt
# 以及 VoV 骨干：
ls ${OPENSCENE_DATA_ROOT}/models/dd3d_det_final.pth
```

评测还要求以下路径存在（脚本会逐个检查并报错）：

```
<CKPT>
$NAVSIM_EXP_ROOT/navtest_two_stage_metric_cache
$OPENSCENE_DATA_ROOT/navsim_logs/test
$OPENSCENE_DATA_ROOT/sensor_blobs/test/test
```

> ⚠ 训练精度锁定 **fp32**（`PRECISION=32`）。官方 GTRS 全系 fp32，且 fp16 会 NaN。
> 想试 `bf16-mixed` 请先跑 1–2 epoch 对比 val loss，不要直接上长训练。

---

## 3. 快速开始（最短路径）

```bash
cd /path/to/sj-1

# 1) 续训：从旧目标训出的 epoch17 续到 28 epoch
RESUME_CKPT=$NAVSIM_EXP_ROOT/train_dp_official_fm_joint/epoch=17.ckpt \
MAX_EPOCHS=28 REINIT_TRAJ_HEAD=false GPU=0,1,2 \
  bash scripts/training/train_dp_official_fm.sh

# 2) 单点评测（不传 CKPT 会自动取该实验目录下最新的 epoch=*.ckpt）
GPU=0 BS=8 bash scripts/evaluation/eval_dp_fm_joint_navtest.sh

# 3) 步数扫描
CKPT=$NAVSIM_EXP_ROOT/train_dp_official_fm_joint/epoch=27.ckpt TAG=fm_dp \
  bash scripts/evaluation/eval_dp_fm_step_scan.sh
```

---

## 4. 训练

### 4.1 两种启动模式

脚本按是否设置 `RESUME_CKPT` 自动切换：

| 模式 | 触发 | 初始化 | 轨迹头 |
|---|---|---|---|
| **从头训** | 不设 `RESUME_CKPT` | 官方 `gtrs_dp_model.ckpt` | `reinit_traj_head=true`（重新初始化，FM 目标与 DDPM 不同） |
| **续训** | `RESUME_CKPT=<ckpt>` | 该 Lightning ckpt | 默认 `false`（保持权重） |

**续训的 epoch 语义**（脚本注释原话）：`MAX_EPOCHS = 已完成 epoch + 1 + 还想加的轮数`。
例如已完成 17 epoch、还想再训 10 轮，就设 `MAX_EPOCHS=28`。

### 4.2 常用环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GPU` | 全部 GPU | 如 `GPU=0,1,2`；脚本据此设 `CUDA_VISIBLE_DEVICES` |
| `BS` | `8` | **联合训练默认 8**：BS=24 在 24 GB 卡上会 OOM（要存感知激活） |
| `LR` | `1e-4` | 学习率 |
| `MAX_EPOCHS` | `5` | 训练轮数 |
| `PRECISION` | `32` | 精度；保持 fp32 |
| `EXP_NAME` | `train_dp_official_fm_joint` | 实验名，决定输出目录 |
| `RESUME_CKPT` | 空 | 设置后进入续训模式 |
| `REINIT_TRAJ_HEAD` | 见上表 | 是否重新初始化轨迹头 |
| `FREEZE_EXCEPT_TRAJ_HEAD` | `false` | 只训轨迹头（旧模式，不作为默认） |
| `LR_MULT_BACKBONE` | `0.1` | 骨干的相对学习率倍率 |
| `BEV_LOSS_WEIGHT` | `1.0` | BEV 辅助损失权重 |
| `NUM_WORKERS` | `8` | DataLoader worker 数 |

### 4.3 FM 与改进项（默认全开）

| 变量 | 默认 | 对应论文 |
|---|---|---|
| `FM_TRAJ_AUX_WEIGHT` | `0.5` | 3.7 绝对轨迹辅助损失 |
| `FM_TRAJ_AUX_POS_WEIGHT` | `1.0` | 位置项权重 |
| `FM_TRAJ_AUX_HEAD_WEIGHT` | `1.0` | 航向项权重 |
| `FM_KIN_WEIGHT` | `0.5` | 3.6 运动学监督（增量残差的差分） |
| `FM_LATTICE_T_PROB` | `0.5` | 网格 t 采样概率 |
| `FM_LATTICE_K_SET` | `[2,3,4,5,8,10,20]` | 训练时采样的推理步数集合 |
| `FM_CLIP_SAMPLE` | `true` | x̂₁ 裁剪（DDPM `clip_sample` 的 FM 对应），少步采样必开 |
| `FM_SELF_CONSISTENCY_P` | `0.5` | 3.4.5 自洽监督概率 |
| `FM_SC_MAX_STEPS` | `4` | 自洽监督的最大积分步数 |
| `FM_SC_WARMUP_STEPS` | `2000` | 自洽监督的 warmup |
| `FM_SELF_CONDITIONING` | `true` | 3.4.6 自条件输入 |
| `FM_SELF_COND_P` | `0.5` | 自条件概率（配置项，脚本未直接暴露） |

> ⚠ **训练与评测的 `FM_SELF_CONDITIONING` / `FM_CLIP_SAMPLE` 必须一致**：自条件改变
> `input_emb` 的 `in_features`，不传会直接 `size mismatch`；裁剪不带则 K=2/3 会外推。
> 评测脚本默认已取 `true`。

> ⚠ **评测扫的每个 K 都必须在 `FM_LATTICE_K_SET` 里**，否则那个 K 对模型是 OOD、曲线不可信。
> 要扫 50/100 必须先把它们加进 `FM_LATTICE_K_SET` 并重训。

### 4.4 创新点 A / B（评价器引导的生成）

默认**全部关闭**（`lambda_*` 都是 0）；关闭时整块被跳过，行为与改动前逐位一致。

```bash
SCORER_GUIDANCE=true \
SCORER_CKPT=<第 4 章 R34 学生评价器 ckpt> \
LAMBDA_SCORE=0.2 \
LAMBDA_SAFE_FEAS=0.1 \
LAMBDA_COMFORT_DIV=0.05 \
  bash scripts/training/train_dp_official_fm.sh
```

| 变量 | 默认 | 含义 |
|---|---|---|
| `SCORER_GUIDANCE` | `false` | 总开关（关闭时下方参数全部无效） |
| `SCORER_CKPT` | 空 | 评分器 ckpt；**必填**，否则退化为随机初始化 |
| `LAMBDA_SCORE` | `0.0` | A：评分引导权重 |
| `SCORE_TEMPERATURE` | `1.0` | A：softmax 温度，越大越平均地优化整池提案 |
| `LAMBDA_SAFE_FEAS` | `0.0` | B-1：安全可行性约束权重 |
| `SAFE_FEAS_MARGIN` | `0.5` | B-1：四项安全概率乘积的下界 |
| `LAMBDA_COMFORT_DIV` | `0.0` | B-2：舒适多样性约束权重 |
| `COMFORT_DIV_TARGET` | `1.0` | B-2：log 空间舒适分数的目标标准差 |
| `GUIDANCE_T_MIN` | `0.5` | 引导区间的**上界**；见下方「t 的约定」 |
| `GUIDANCE_T_FLOOR` | `0.1` | 引导区间的**下界**；避免 `∂x̂₁/∂v ∝ t → 0` 的梯度消失区 |
| `GUIDANCE_WARMUP_STEPS` | `2000` | 引导权重的线性 warmup（按训练步）；续训已收敛模型时必须有 |
| `GUIDANCE_SAMPLES` | `2` | 每次引导采样的提案数 |
| `GUIDANCE_EVERY_N` | `1` | 每 N 步施加一次引导 |

> 评分器必须是**可微且与 EPDMS 同构**的第 4 章双流评分头；`scorer_freeze=true` 时
> 评分器参数冻结，但梯度仍穿过它回传到生成器。

**t 的约定（很重要）。** 本仓库的流匹配约定是

```
x_t = (1 − t)·x1 + t·noise      # t=0 是数据，t=1 是纯噪声
```

因此一步估计 `x̂₁ = x_t + t_eff·v/(1−σ)` 只在**低 t 端**可靠；t 大时 `x̂₁` 接近噪声，
对它的打分没有意义。引导只在 `t ∈ [GUIDANCE_T_FLOOR, GUIDANCE_T_MIN]`（默认 `[0.1, 0.5]`）
内施加。

> ⚠ **历史 bug（2026-10-14 那次训练）**：旧实现写成 `t ≥ t_min`，方向反了——恰好只给
> 最不可靠的一半步数打分，于是 5 epoch 续训从 **0.7380 掉到 0.6369**（DAC −9.5、EC −12.5）。
> 已修正为区间门控，并加 `test_guidance_time_weight_is_low_t_not_high_t` 钉住方向。
> 该次实验的完整记录见 `scripts/dp/runs/ch3_20261014_lambda01/RUN.md`。

> `SCORER_ALLOW_PARTIAL_LOAD`（默认 `false`）：评分器 ckpt 的键与本模型**完全对不上**、
> 或存在**缺键**时直接报错（缺键意味着那部分仍是随机权重）。多出来的键只提示、不阻断。
> 以前是 `strict=False` 静默继续，会导致带着随机评分器跑完整个训练而无人察觉。

### 4.5 引导模块的单元测试（远程）

```bash
python -m pytest navsim/agents/dp/tests/test_scorer_guidance.py -q
```

本机无 GPU、无 torch，跑不了；需在服务器上执行。

---

## 5. 评测

### 5.1 单 checkpoint

```bash
CKPT=<path>/epoch=27.ckpt GPU=0 BS=8 WORKERS=8 \
  bash scripts/evaluation/eval_dp_fm_joint_navtest.sh
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `CKPT` | 自动取最新 `epoch=*.ckpt` | 要评的 checkpoint |
| `SPLIT` | `navtest` | 数据集划分 |
| `GPU` | `0` | 单卡评测 |
| `BS` | `8` | batch size |
| `WORKERS` | `8` | 评测线程数 |
| `FM_NUM_INFERENCE_STEPS` | `20` | 推理步数 K |
| `USE_FM` | `true` | `false` 时按 DDPM 口径评测（扩散基线对照） |
| `EXP_NAME` | 自动生成 | 输出实验名 |
| `FORCE_RERUN` | `0` | 已有 CSV 时是否强制重跑 |
| `REUSE_SUBSCORE_IF_EXISTS` | `1` | 复用已存的子分数 pickle |

产物：
- 逐 token 子分数 pickle：`${SUBSCORE_PATH}`，默认
  `$NAVSIM_EXP_ROOT/train_dp_official_fm_joint/<ckpt_stem>_<split>_fm.pkl`
- 评测结果 CSV：`$NAVSIM_EXP_ROOT/<EXP_NAME>/`

### 5.2 推理步数扫描

```bash
CKPT=<path>/epoch=27.ckpt TAG=fm_dp \
  bash scripts/evaluation/eval_dp_fm_step_scan.sh
```

- 默认扫 `K = 1 2 3 5 8 10 20`，可用 `FM_STEPS_LIST="1 2 5 10 20"` 覆盖。
- 每个 K 的实验名：`<TAG>/stepscan-<TAG>-K<K>`。
- 扩散基线对照：`TAG=ddpm CKPT=<官方 DP ckpt> USE_FM=false bash ...`

> ⚠ **已知缺口**：脚本末尾提示用 `scripts/evaluation/collect_dp_step_scan.py` 汇总成 CSV，
> 但**该文件目前不存在**。扫描本身能跑完，汇总这一步会失败。需要先补这个脚本，
> 或直接从各 `stepscan-*/` 目录里的 CSV 手动取数。

---

## 6. 消融实验

### 6.1 组列表

```bash
bash scripts/dp/ablation/sweep_dp.sh --list
```

- **组件消融（5.4.1，相对 `full` 逐项关闭）**
  `base`（全关）/ `full`（全开）/ `comp_no_kin` / `comp_no_aux` / `comp_no_lattice` /
  `comp_no_sc` / `comp_no_selfcond`
- **创新点 A（5.4.2，评分引导）**
  `guide_lo`(0.05) / `guide_mid`(0.2) / `guide_hi`(0.5) /
  `guide_tmin03`(0.3) / `guide_tmin07`(0.7)
- **创新点 B（5.4.2，指标结构感知）**
  `struct_feas` / `struct_div` / `struct_both` / `struct_feas_lo`(0.3) / `struct_feas_hi`(0.7)

### 6.2 运行

```bash
# 先 dry-run，确认命令
DRY_RUN=1 RESUME_CKPT=<起点 ckpt> SCORER_CKPT=<评分器 ckpt> \
  bash scripts/dp/ablation/sweep_dp.sh

# 先跑参照系 base + full
RESUME_CKPT=<起点 ckpt> EPOCHS=6 SCORER_CKPT=<评分器 ckpt> \
SWEEP_GROUPS="base full" bash scripts/dp/ablation/sweep_dp.sh

# 再跑组件消融
RESUME_CKPT=<起点 ckpt> EPOCHS=6 SCORER_CKPT=<评分器 ckpt> \
SWEEP_GROUPS="comp_no_kin comp_no_aux comp_no_lattice comp_no_sc comp_no_selfcond" \
  bash scripts/dp/ablation/sweep_dp.sh

# 最后跑 A/B
RESUME_CKPT=<起点 ckpt> EPOCHS=6 SCORER_CKPT=<评分器 ckpt> \
SWEEP_GROUPS="guide_lo guide_mid guide_hi struct_feas struct_div struct_both" \
  bash scripts/dp/ablation/sweep_dp.sh
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `RESUME_CKPT` | **必填** | 所有格子从**同一个起点**续训，保证只差被消融的那一项 |
| `EPOCHS` | `6` | 每个格子训练多少 epoch |
| `SCORER_CKPT` | 空 | A/B 组必需；不设会退化为随机评分器 |
| `SWEEP_GROUPS` | 空 = 全部 | 指定要跑的组 |
| `DRY_RUN` | `0` | `1` 时只打印命令 |
| `BASE_EXP` | `train_dp_official_fm_joint` | 输出根目录 |

产物：`$NAVSIM_EXP_ROOT/<BASE_EXP>/<组名>/`。

> **顺序有讲究**：`base` 与 `full` 必须先跑，它们是其余所有格子的参照系。每组都是
> 「`full` 去掉该项」，这样单格差异才可解释。

---

## 7. 论文实验与命令的对应

| 论文位置 | 要跑什么 |
|---|---|
| 5.2.1 轨迹生成质量对比 | `eval_dp_fm_step_scan.sh`（FM）+ `TAG=ddpm USE_FM=false`（DDPM） |
| 5.4.1 FM-DP 组件消融 | `sweep_dp.sh` 的 `base` / `full` / `comp_*` |
| 5.4.2 创新点 A 消融 | `guide_lo/mid/hi`、`guide_tmin03/07` |
| 5.4.2 创新点 B 消融 | `struct_feas/div/both`、`struct_feas_lo/hi` |

---

## 8. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `Official DP checkpoint not found` | 没设 `RESUME_CKPT` 且官方 ckpt 路径不对；检查 `/home/ws/navsim_workspace/GTRS_official/data/models/gtrs_dp_model.ckpt` |
| `[ERROR] ... is not the sj-dp-fm tree` | 当前 `NAVSIM_DEVKIT_ROOT` 指向了别的 GTRS 副本；脚本会自动改回本仓库 |
| `Missing required path: ...metric_cache...` | 先跑 metric cache 生成（见 `docs/cache.md`） |
| `size mismatch` on `input_emb` | 评测没带 `FM_SELF_CONDITIONING=true`（须与训练一致） |
| CUDA OOM | 降 `BS`（默认已降到 8）；或改 `FREEZE_EXCEPT_TRAJ_HEAD=true` 做只训轨迹头的对照 |
| 扫描结果异常 | 检查该 K 是否在 `FM_LATTICE_K_SET` 内 |
| 引导完全无效果 | 检查 `SCORER_GUIDANCE=true`、`SCORER_CKPT` 非空，且日志里没有 `完全不匹配` 报错；再看 `GUIDANCE_T_MIN` 是否设得比 `GUIDANCE_T_FLOOR` 还小（区间为空） |

---

## 9. 已知缺口

1. **`scripts/evaluation/collect_dp_step_scan.py` 缺失**：步数扫描的汇总步骤跑不了，
   需补脚本或手动从 `stepscan-*/` 取 CSV。
2. **conda 路径写死**为 `/home/ws/anaconda3`，换机器需改脚本。
3. **本机无 GPU**：所有训练/评测只能在 3×RTX 3090 服务器上执行；本机只能做静态检查与渲染图。
