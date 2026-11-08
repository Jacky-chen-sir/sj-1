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

1. **开环 ADE / loss 与舒适度脱钩**：ep17 的 ADE、center ADE、总 loss 全面优于 ep4，但 `two_frame_comfort` 从 ~0.88 掉到 ~0.16。仅看 TB **选不出** 该 early-stop 的 ckpt。
2. **BEV semantic 略升**（0.30→0.35）：长训里感知头有变差苗头，幅度不如 comfort 崩，但是唯一在 TB 里「变坏」的明显曲线。
3. **P1 center vs P2 @ ep4**：TB 几乎重合（同配方可复现）；掉分发生在 **ep4 之后继续训**。
4. **p1_base 无 `fm_center_*`**：符合 `center_w=0`；两臂前几 ep 的 dp/ADE 走势接近，差异主要在 center 项。

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
P2 失败不是 TB 里 loss 发散，而是 **优化目标（中心 ADE）与 PDM comfort 不一致 + 长训过拟合抖动**。

## 5. 导出文件

```
tb_export/
  p1_base_epoch_scalars.csv
  p1_center_epoch_scalars.csv
  p2_center18_epoch_scalars.csv
  tb_key_curves.png
```
