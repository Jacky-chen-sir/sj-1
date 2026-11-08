# 第三章训练 / 评测代码说明（sj-df_model）

> 对应实验目录：`exp/ch3_df/`  
> 一键脚本：`scripts/dp/runs/ch3_df_model/run_phase{0,1,2}.sh`

## 1. 核心改动（模型）

| 文件 | 作用 |
|---|---|
| `navsim/agents/dp/dp_config.py` | `dp_eval_zero_noise`；`fm_center_weight/steps/p/warmup_steps/k_set`；`fm_val_probe(_steps)` |
| `navsim/agents/dp/dp_model.py` | 推理置零起点；训练 `_fm_zero_noise_path` + `_center_consistency_loss`；校验期确定性探针 |
| `navsim/agents/dp/tests/test_center_consistency.py` | warmup / 损失单测 |
| `navsim/agents/dp/tests/test_training_gates.py` | 训练/校验口径单测（2026-11-08 新增） |
| `navsim/planning/training/agent_lightning_module.py` | `DP_PROPOSAL_AGG=first\|medoid`（Phase0）；`_sc_calls` 的 ckpt 持久化 |

### 1.3 训练/校验口径（2026-11-08 修复）

| 问题 | 修法 |
|---|---|
| 自洽分支在**校验**期也生效，一半 val batch 的目标建立在模型自己积分出的扰动状态上，`val/loss` 逐轮不可比 | `sc_alpha = self._fm_sc_alpha() if self.training else 0.0` |
| 引导分支同样在 val 里跑（每步 n_g 次评分器前向） | 整块加 `self.training` 门控 |
| `_sc_calls` 是纯 Python 属性，续训后三处 warmup 全部从 0 重爬 | 写进 ckpt 顶层键 `dp_sc_calls`（`on_save/on_load_checkpoint`），不破坏旧 ckpt 严格加载 |
| `limit_val_batches=0.03`，`val/loss` 的标准误盖过 epoch 间差异 | 脚本层默认 0.1（`LIMIT_VAL_BATCHES`） |
| `ModelCheckpoint` 按 `val/loss_epoch` 挑 ckpt = 抽签 | 改 monitor `val/fm_zero_ade_epoch`（零噪声、固定步、无随机数） |
| 训练 seed 写死在 `competition_training.yaml`，不可覆盖 | `SEED` 环境变量 → `seed=` 覆盖 |
| 中心路径只在 K=4 上训、在 K=20 上评 | `fm_center_k_set`（默认 `[4,8,20]`），每批抽一个 K |

新增的校验期指标（全部确定性，不参与反传）：

| 指标 | 含义 |
|---|---|
| `val/fm_zero_ade` / `fde` | 零噪声路径（`fm_val_probe_steps`=20）与 GT 的落点误差 |
| `val/fm_zero_accel` / `jerk` | 增量的 1/2 阶差分的 RMS，**comfort 的同向代理** |

### 1.4 中心一致性训练

- 仅在 **`self.training`** 且 `fm_center_weight>0` 时计算（验证集不加，避免两臂 val/loss 不可比）。
- `_fm_zero_noise_path`：从 \(x=0\) 做 **`fm_center_steps`（默认 4）** 步 Euler（可带 self-cond），图参与反传。
  步数可由 `fm_center_k_set` 覆盖成集合，每批抽一个，让中心路径在评测会扫到的每个 K 上都成立。
- 损失：相对 GT 的 **漂移 + 航向 + 运动学**（权重对齐 traj aux / `fm_kin_weight`）；warmup 按 `_sc_calls` 线性爬到 `fm_center_weight`。

## 2. 训练入口

| 脚本 | 说明 |
|---|---|
| `scripts/training/train_dp_official_fm.sh` | 官方 DP ckpt → FM；`REINIT_TRAJ_HEAD`、`FM_CENTER_*`、`FM_SELF_CONDITIONING` 等环境变量 |
| `scripts/dp/runs/ch3_df_model/run_phase1.sh` | 两臂 5ep：`p1_base`（w=0）/ `p1_center`（w=0.5）+ zero 评测 |
| `scripts/dp/runs/ch3_df_model/run_phase2.sh` | `p2_center18` 默认 18ep + zero 评测；可选评已有 FM ep18 |

**YAML / 覆盖快照**：见同目录 `configs/README.md`  
（`competition_training.yaml`、`gtrs_diffusion_policy.yaml` 副本 + P1/P2 真实 `overrides.yaml`）。

常用环境变量：

```bash
MAX_EPOCHS / EPOCHS
EXP_NAME                  # 如 ch3_df/p2_center18
REINIT_TRAJ_HEAD=true
FM_CENTER_WEIGHT=0.5 FM_CENTER_STEPS=4 FM_CENTER_P=1.0 FM_CENTER_WARMUP_STEPS=2000
FM_CENTER_K_SET="[4,8,20]"        # 中心路径的训练步数集合；空 = 固定 FM_CENTER_STEPS
FM_SELF_CONDITIONING=true|false   # P0 用 false；P1/P2 实际 true
FM_VAL_PROBE=true FM_VAL_PROBE_STEPS=20   # 校验期零噪声探针（ckpt 选择用）
SEED=0                            # 训练种子，可覆盖（此前写死为 0）
LIMIT_VAL_BATCHES=0.1             # 校验抽样比例（此前 0.03）
SCORER_GUIDANCE=false             # 本章不接评价器
GPU BS NUM_WORKERS
```

每次训练会在 `$NAVSIM_EXP_ROOT/$EXP_NAME/code_rev.txt` 记录
`git_head` / 是否脏工作区 / 启动时间 / seed / epochs —— 上一轮就是因为没有这个，
只能靠 CSV 里有没有 `val/fm_center_*` 才倒推出运行代码与仓库对不上。

数据与设备：`navtrain`，3×3090，bs=8（与已跑实验一致）。

## 3. 评测入口

| 脚本 | 说明 |
|---|---|
| `scripts/evaluation/eval_dp_fm_joint_navtest.sh` | navtest one-stage；读 `DP_EVAL_ZERO_NOISE`、`FM_*`、`NUM_PROPOSALS`、`SEED` |
| `run_phase0.sh` | first×3 / medoid / zero / 官方 DDPM |
| `run_phase1.sh` / `run_phase2.sh` | 训完后自动 zero 评测并追加 summary csv |

Phase1 后续增强（远程 `c28c7cb`）：可评最后 N 个 epoch、K∈{20, center_steps}；`run_phase2` 目前仍只评**最新** epoch、K=20。

## 4. 已跑实验 ↔ 代码开关

| 实验 | 训练代码路径 | 关键开关 |
|---|---|---|
| P0 | 无新训（用已有 FM ep18） | 仅评测：`DP_EVAL_ZERO_NOISE` / `DP_PROPOSAL_AGG` |
| P1 base | `train_dp_official_fm.sh` via `run_phase1` | `FM_CENTER_WEIGHT=0`，5ep |
| P1 center | 同上 | `FM_CENTER_WEIGHT=0.5`，5ep |
| P2 center18 | `run_phase2.sh` | 同 P1 center，**18ep** |

## 5. 文档索引

| 文件 | 内容 |
|---|---|
| `STATUS.md` | 总进度与结论 |
| `RESULTS_PHASE0.md` / `RESULTS_PHASE1.md` / `RESULTS_PHASE2.md` | 分阶段分数 |
| `TENSORBOARD.md` / `tb_export/` | P1/P2 曲线导出与解读 |
| `PLAN.md` | 原实验计划（含 Phase3 ViT-L） |
| `summary_phase{0,1,2}.csv` | 机器可读汇总 |
