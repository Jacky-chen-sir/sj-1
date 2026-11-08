# 第三章（sj-df_model）进度总表

> 更新：2026-11-09  
> 分支：`sj-df_model`　目标：单独生成模型 navtest > 0.73（不接词表/评价器）

## 总览

| 阶段 | 内容 | 状态 | 关键数字 |
|---|---|---|---|
| **Phase 0** | 推断方式对照（零训练） | **完成** | zero **0.7657**（本章最高） |
| **Phase 1** | 中心一致性 5ep 两臂 | **完成** | center−base **+0.073** |
| **Phase 2** | center × 18ep | **完成，未达预期** | ep17 **0.6621**；ep04 **0.6958**（early-stop 成立） |
| **Phase1-v2** | 修复后代码重跑 | **进行中** | `ch3_df_v2` |
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
| P2 ep04（同 run 早停） | **0.6958** |
| **P2 ep17** | **0.6621** |

## Phase 0（完成）

推断退化可免费抬分；主推零噪声。见 `RESULTS_PHASE0.md`。

## Phase 1（完成）

同基准短训：center 损失有效（+0.073）。实际 `self_cond=true`。见 `RESULTS_PHASE1.md`。

## Phase 2（完成，失败；early-stop 已验证）

同配方拉满 18ep → **0.6621**（comfort **0.16**）。  
同 run 的 **ep04 = 0.6958**（comfort **0.89**）≈ P1 center 0.7125 → **长训变差，不是起点就坏**。  
见 `RESULTS_PHASE2.md` §ep04。

## Phase1-v2（进行中）

修复后代码重跑：`ch3_df_v2/p1_{base,center}`，5ep，`fm_center_k_set=[4,8,20]`，`val/fm_zero_ade` 选 ckpt。

## Phase 3（未开始）

换 Depth Anything ViT-L + FM，约 18ep。见 `PLAN.md`。

## 论文口径（截至 P2）

1. 单样本随机取提案相对确定性中心约 **−2.4** 分（P0）。
2. 中心一致性短训相对同预算基线 **+7.3** 分（P1）——**该差值不稳健**，见下。
3. 绝对分主结果仍用 **P0 zero 0.7657**；冲更高优先 Phase3，而非再盲训 center×18。

> ⚠️ **2026-11-08 更正**：
> - P1 的 `+0.073` 主要来自 `two_frame_comfort`（+0.25），而 comfort 是阈值型，
>   P2 同一条曲线在 0.16~0.88 之间摆动。9905 个场景不构成统计功效——
>   **样本单位是 ckpt / seed，不是场景**。
> - P1 与 P2 共用 seed=0，P2 前 5 epoch 就是 P1，不构成独立复现（`overrides.yaml`
>   逐行只差 name / ckpt_path / max_epochs）。
> - P1 base 臂 5ep 掉到 0.6399（低于 P0 的 0.7657），说明短训本身在这个预算下
>   也会掉分，两臂的差值不能直接读成"损失有效"。
>
> 写进论文前必须补齐：至少两个 seed × 至少 3 个 ckpt 的同向变化。

## 产物

```
exp/ch3_df/summary_phase{0,1,2}.csv
exp/ch3_df/p0_* / p1_{base,center} / p2_center18 /
exp/ch3_df/navtest-p1_*-zero / navtest-p2_center18-zero /
```
