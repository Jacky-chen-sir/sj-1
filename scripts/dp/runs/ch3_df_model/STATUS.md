# 第三章（sj-df_model）进度总表

> 更新：2026-11-08  
> 分支：`sj-df_model`　目标：单独生成模型 navtest > 0.73（不接词表/评价器）

## 总览

| 阶段 | 内容 | 状态 | 关键数字 |
|---|---|---|---|
| **Phase 0** | 推断方式对照（零训练） | **完成** | zero **0.7657**（本章最高） |
| **Phase 1** | 中心一致性 5ep 两臂 | **完成** | center−base **+0.073** |
| **Phase 2** | center × 18ep | **完成，未达预期** | **0.6621**（comfort 崩） |
| Phase 3 | ViT-L 换骨干 | 未开始 | 需工程接入后再训 |

详细：`RESULTS_PHASE0/1/2.md`；代码：`TRAINING_CODE.md`；曲线：`TENSORBOARD.md` + `tb_export/`。

## 分数一览（zero-noise，除非另注）

| 标签 | score |
|---|---:|
| official DDPM first | 0.7317 |
| FM first（×3 mean） | 0.7418 |
| FM medoid | 0.7582 |
| **FM zero（P0，主推）** | **0.7657** |
| p1_base 5ep | 0.6399 |
| p1_center 5ep | 0.7125 |
| **p2_center18** | **0.6621** |

## Phase 0（完成）

推断退化可免费抬分；主推零噪声。见 `RESULTS_PHASE0.md`。

## Phase 1（完成）

同基准短训：center 损失有效（+0.073）。实际 `self_cond=true`。见 `RESULTS_PHASE1.md`。

## Phase 2（完成，失败）

同配方拉满 18ep → **0.6621**，低于 P1 与 P0。  
根因：轨迹加速度/jerk 升高，**two_frame comfort 0.88→0.16**（非评测 bug）。  
**TB**：val loss / ADE 持续下降到 ep17，**看不出**掉分；仅 BEV semantic 略升。见 `TENSORBOARD.md`。  
正在补评 `epoch=04` 验证 early-stop。

## Phase 3（未开始）

换 Depth Anything ViT-L + FM，约 18ep。见 `PLAN.md`。**当前无排队训练。**

## 论文口径（截至 P2）

1. 单样本随机取提案相对确定性中心约 **−2.4** 分（P0）。
2. 中心一致性短训相对同预算基线 **+7.3** 分（P1）；长训会过拟合抖动（P2 负面结果）。
3. 绝对分主结果仍用 **P0 zero 0.7657**；冲更高优先 Phase3，而非再盲训 center×18。

## 产物

```
exp/ch3_df/summary_phase{0,1,2}.csv
exp/ch3_df/p0_* / p1_{base,center} / p2_center18 /
exp/ch3_df/navtest-p1_*-zero / navtest-p2_center18-zero /
```
