# TensorBoard 解读（P1 / P2）

> 导出目录：`tb_export/`（epoch CSV + `tb_key_curves.png`）  
> 原始事件文件仍在 `exp/ch3_df/*/lightning_logs/`（未入库，体积大）。

## 1. 日志位置

| run | events |
|---|---|
| p1_base | `exp/ch3_df/p1_base/2026.10.23.19.54.55/lightning_logs/version_0/` |
| p1_center | `exp/ch3_df/p1_center/2026.10.26.05.13.51/lightning_logs/version_0/` |
| p2_center18 | `exp/ch3_df/p2_center18/2026.10.29.13.52.58/lightning_logs/version_0/` |

本地查看：

```bash
tensorboard --logdir_spec=\
p1_base:exp/ch3_df/p1_base/2026.10.23.19.54.55/lightning_logs,\
p1_center:exp/ch3_df/p1_center/2026.10.26.05.13.51/lightning_logs,\
p2:exp/ch3_df/p2_center18/2026.10.29.13.52.58/lightning_logs
```

## 2. 关键结论（和 navtest 对照）

**TB 曲线看起来一直在变好，但闭环分数变差——这是最关键的信号。**

| 节点 | val/loss | val FM ADE | val center ADE | val BEV sem | navtest zero |
|---|---:|---:|---:|---:|---:|
| p1_center @ ep4 | 1.348 | 0.745 | 0.612 | 0.298 | **0.7125** |
| p2 @ ep4 | 1.358 | 0.715 | 0.566 | 0.300 | （诊断评测中） |
| p2 @ ep17 | **1.293** | **0.638** | **0.493** | **0.353** | **0.6621** |

解读：

1. **开环 ADE / loss 与舒适度脱钩**：ep17 的 ADE、center ADE 全面优于 ep4，但 `two_frame_comfort` 从 ~0.88 掉到 ~0.16。仅看 TB **选不出** 该 early-stop 的 ckpt。
2. **BEV semantic 略升**（0.30→0.35）：长训里感知头有变差苗头，幅度不如 comfort 崩，但是唯一在 TB 里「变坏」的明显曲线。

> ⚠️ **2026-11-08 更正（读 CSV 后）**：上表 "val/loss 1.348 / 1.293" 只是单点，
> 逐 epoch 看这条曲线**没有趋势**：
>
> ```
> ep0 6.25  ep1 3.73  ep2 4.78  ep3 3.26  ep4 1.36  ep5 1.77  ep6 1.60  ep7 2.42
> ep8 1.89  ep9 1.05  ep10 2.34 ep11 1.14 ep12 2.99 ep13 1.28 ep14 1.49 ep15 3.95
> ep16 1.09 ep17 1.29
> ```
>
> 原因有三：(a) `limit_val_batches=0.03`，每轮只验约 3% 的验证批；(b) 校验期自洽
> 分支没有 `self.training` 门控，一半 batch 的目标建立在模型自己积分出的扰动状态上；
> (c) 无 EMA、lr 恒定。三者叠加使 val/loss 的分辨率低于 epoch 间真实差异，
> **不能用来挑 ckpt**。`save_top_k` 按它存，等于抽签。

1'.（原编号 3 作废）**P1 与 P2 不是独立复现**：三次实验的 `overrides.yaml` 逐行只差
`experiment_name` / `ckpt_path` / `max_epochs`（5 vs 18），seed 固定为 0。P2 的前 5 个
epoch 就是 P1（`train_fm_center_ade` 前 5 轮逐位吻合到第三位小数）。同一 RNG 流的不同
截断，不构成第二次观测。

## 3. 曲线摘要（见图 `tb_export/tb_key_curves.png`）

- `val/loss`、`val/fm_x1_ade`、`val/fm_center_ade`：P2 全程单调下降至 ep17。
- `train/fm_center_loss`：warmup 后稳定下降（~0.042→~0.008）。
- `val/bev_semantic_loss`：中后期缓慢抬头（值得盯，但仍非主因）。

## 4. 对实验决策的含义

| 若只看 TB | 会以为 | 实际 |
|---|---|---|
| 选 last ckpt | ep17 最好 | navtest 最差 |
| 选最低 ADE | ep17 | 轨迹 jerk×4，comfort 崩 |

→ 本章 **必须以 navtest（或至少 comfort/jerk 代理）做 ckpt 选择**；不能按 val ADE early-stop 的反方向理解成「再训一会」。  
P2 失败不是 TB 里 loss 发散，而是 **优化目标（中心 ADE）与 PDM comfort 不一致**（"长训过拟合抖动"这条具体机制尚未被验证：P1 与 P2 是同一 RNG 流，不能互证）。

> **2026-11-08 代码修复**（对应本节结论）：
> 1. 校验期关闭自洽分支与引导分支（`self.training` 门控），val/loss 变成确定量；
> 2. `limit_val_batches` 默认 0.03 → 0.1（脚本层）；
> 3. 新增 `val/fm_zero_ade`：零噪声路径、固定步数、无随机抽样，专供 ckpt 选择；
>    `ModelCheckpoint` 已改为 monitor 它；
> 4. 新增 `val/fm_zero_accel` / `fm_zero_jerk`：comfort 的同向代理，能在 TB 里直接看抖动。

## 5. 导出文件

```
tb_export/
  p1_base_epoch_scalars.csv
  p1_center_epoch_scalars.csv
  p2_center18_epoch_scalars.csv
  tb_key_curves.png
```
