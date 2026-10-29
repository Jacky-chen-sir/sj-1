# 第三章（sj-df_model）进度总表

> 更新：2026-10-29  
> 分支：`sj-df_model`　目标：单独生成模型 navtest > 0.73，越高越好（不接词表/评价器）

## 总览

| 阶段 | 内容 | 状态 | 关键数字 |
|---|---|---|---|
| **Phase 0** | 推断方式对照（零训练） | **完成** | zero **0.7657**（+0.024 vs first） |
| **Phase 1** | 中心一致性 5ep 两臂 | **完成** | center−base **+0.073**（过线） |
| **Phase 2** | center 拉满 18ep | **进行中** | `p2_center18` 已启动 |
| Phase 3 | ViT-L 换骨干 | 未开始 | — |

详细分数见 `RESULTS_PHASE0.md` / `RESULTS_PHASE1.md`。

## Phase 0（已完成，2026-10-22 → 10-23）

- **问题**：历史 0.738 取 `proposals[0]`，不是确定性上限。
- **结论**：medoid +0.016 / zero **+0.024** vs FM first（0.7418±0.0006）；主推 **零噪声路径**。
- **脚本**：`run_phase0.sh`　**结果文**：`RESULTS_PHASE0.md`

| | official DDPM | FM first | medoid | zero |
|---|---:|---:|---:|---:|
| score | 0.7317 | 0.7418±0.0006 | 0.7582 | **0.7657** |

## Phase 1（已完成，训 10-23→10-28，评 10-28）

- **问题**：仅推理换中心不够，训练里是否也应监督零噪声路径？
- **设置**：官方 ckpt + reinit 头；base `center_w=0` vs center `0.5`；各 5ep；评测 zero。
- **结论**：**符合预期且远超门槛**（+0.073 ≫ +0.005）。涨分来自 DAC / comfort / NC / TTC。绝对分低于 P0 zero 属短预算，不否定方法。
- **脚本**：`run_phase1.sh`　**结果文**：`RESULTS_PHASE1.md`

| arm | score (zero) |
|---|---:|
| p1_base | 0.6399 |
| p1_center | **0.7125**（Δ **+0.0726**） |

口径备注：两臂实际 `fm_self_conditioning=true`（与 P0 的 false 不同）；臂间公平，跨阶段比绝对分时注意。

## Phase 2（进行中，2026-10-29 启动）

- **作业**：`p2_center18` = Phase1 有效配方 × **18** epoch → 自动 zero 评测。
- **对照**：已有 FM-only ep18（`train_dp_official_fm_joint/epoch=18-step=81776.ckpt`，P0 zero **0.7657**）；脚本支持 `ARM=fm18|both` 补评。
- **脚本**：`run_phase2.sh`
- **日志**：`exp/logs/ch3_df_phase2_center18_20261029_135213.log`
- **ETA**：按 P1 节奏约 **8–9 天**训 + ~2h 评。

```bash
# 已用（与 P1 同口径）
FM_SELF_CONDITIONING=true ARM=center EPOCHS=18 \
  bash scripts/dp/runs/ch3_df_model/run_phase2.sh
```

## 产物根目录

```
exp/ch3_df/
  summary_phase0.csv  summary_phase1.csv  (summary_phase2.csv 待写)
  p0_* / p1_base / p1_center / p2_center18 /
  navtest-p1_{base,center}-zero /
```

## 论文侧已可写的结论（截至 Phase1）

1. 单样本随机取提案相对确定性中心有 **~2.4** 分退化（表：推断对照）。
2. 把零噪声路径写进训练目标（中心一致性），同预算下相对无该损失的基线 **+7.3** 分（5ep 趋势；长训待 Phase2）。
3. 机制上主要抬 **可行驶区 / 舒适 / 碰撞相关**，略换 progress——与「更稳中心」叙事一致。
