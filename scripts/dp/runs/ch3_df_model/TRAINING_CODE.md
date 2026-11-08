# 第三章训练 / 评测代码说明（sj-df_model）

> 对应实验目录：`exp/ch3_df/`  
> 一键脚本：`scripts/dp/runs/ch3_df_model/run_phase{0,1,2}.sh`

## 1. 核心改动（模型）

| 文件 | 作用 |
|---|---|
| `navsim/agents/dp/dp_config.py` | `dp_eval_zero_noise`；`fm_center_weight/steps/p/warmup_steps` |
| `navsim/agents/dp/dp_model.py` | 推理置零起点；训练 `_fm_zero_noise_path` + `_center_consistency_loss` |
| `navsim/agents/dp/tests/test_center_consistency.py` | warmup / 损失单测 |
| `navsim/agents/dp/agent_lightning_module.py` | `DP_PROPOSAL_AGG=first\|medoid`（Phase0） |

### 1.1 零噪声推理

`dp_eval_zero_noise=True` 时，FM 采样起点为 **全零**（确定性 ODE），而非高斯噪声；可与 `num_proposals=1` 合用。

### 1.2 中心一致性训练

- 仅在 **`self.training`** 且 `fm_center_weight>0` 时计算（验证集不加，避免两臂 val/loss 不可比）。
- `_fm_zero_noise_path`：从 \(x=0\) 做 **`fm_center_steps`（默认 4）** 步 Euler（可带 self-cond），图参与反传。
- 损失：相对 GT 的 **漂移 + 航向 + 运动学**（权重对齐 traj aux / `fm_kin_weight`）；warmup 按 `_sc_calls` 线性爬到 `fm_center_weight`。
- **注意**：训练中心路径 K=4，评测默认 K=20——网格不一致，长训时可能放大抖动（P2 已见）。

## 2. 训练入口

| 脚本 | 说明 |
|---|---|
| `scripts/training/train_dp_official_fm.sh` | 官方 DP ckpt → FM；`REINIT_TRAJ_HEAD`、`FM_CENTER_*`、`FM_SELF_CONDITIONING` 等环境变量 |
| `scripts/dp/runs/ch3_df_model/run_phase1.sh` | 两臂 5ep：`p1_base`（w=0）/ `p1_center`（w=0.5）+ zero 评测 |
| `scripts/dp/runs/ch3_df_model/run_phase2.sh` | `p2_center18` 默认 18ep + zero 评测；可选评已有 FM ep18 |

常用环境变量：

```bash
MAX_EPOCHS / EPOCHS
EXP_NAME                  # 如 ch3_df/p2_center18
REINIT_TRAJ_HEAD=true
FM_CENTER_WEIGHT=0.5 FM_CENTER_STEPS=4 FM_CENTER_P=1.0 FM_CENTER_WARMUP_STEPS=2000
FM_SELF_CONDITIONING=true|false   # P0 用 false；P1/P2 实际 true
SCORER_GUIDANCE=false             # 本章不接评价器
GPU BS NUM_WORKERS
```

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
| `PLAN.md` | 原实验计划（含 Phase3 ViT-L） |
| `summary_phase{0,1,2}.csv` | 机器可读汇总 |
