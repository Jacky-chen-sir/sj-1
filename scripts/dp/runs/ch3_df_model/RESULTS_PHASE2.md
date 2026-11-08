# Phase 2 结果：中心一致性长预算（18 epoch）

> 状态：**已完成，未达预期**。  
> 协议：与 Phase1 同配方；navtest、`DP_EVAL_ZERO_NOISE=true`、K=20、`fm_self_conditioning=true`。  
> 机器汇总：`exp/ch3_df/summary_phase2.csv`（本目录副本 `summary_phase2.csv`）。

## 1. 设置

| 项 | 值 |
|---|---|
| 作业名 | `ch3_df/p2_center18` |
| 初始化 | 官方 `gtrs_dp_model.ckpt` + `reinit_traj_head=true` |
| 预算 | **18** epoch（最终 `epoch=17-step=77472`） |
| `fm_center_weight` / steps / p / warmup | **0.5** / 4 / 1.0 / 2000 |
| `FM_SELF_CONDITIONING` | **true**（对齐 P1，非 P0） |
| 评测 | zero-noise，`num_proposals=1`，seed=3000 |
| 训练时段 | 2026-10-29 13:52 → 2026-11-07 10:33 |
| 评测时段 | 2026-11-07 10:34 → 14:13 |
| 日志 | `exp/logs/ch3_df_phase2_center18_20261029_135213.log` |

一键：`FM_SELF_CONDITIONING=true bash scripts/dp/runs/ch3_df_model/run_phase2.sh`

## 2. 总分对照

| 模型 | epochs | score (zero) | vs P0 zero | vs P1 center |
|---|---:|---:|---:|---:|
| P0 FM zero（无 center 损失） | 18 (旧) | **0.7657** | — | — |
| P1 center | 5 | **0.7125** | −0.053 | — |
| **P2 center18** | 18 | **0.6621** | **−0.104** | **−0.050** |

**预期**：拉满后逼近/超过 P0 zero，至少不低于 P1。  
**实际**：明显低于二者 → **长训策略失败**（非评测管道错误，见 §4）。

## 3. 分项（stage_one）

| 子指标 | P1 center (5ep) | P2 (18ep) | Δ |
|---|---:|---:|---:|
| score | 0.7125 | 0.6621 | **−0.050** |
| no_at_fault_collisions | 0.9725 | 0.9732 | +0.001 |
| drivable_area_compliance | 0.9045 | 0.9143 | +0.010 |
| ego_progress | 0.8761 | 0.8782 | +0.002 |
| time_to_collision_within_bound | 0.9618 | 0.9641 | +0.002 |
| lane_keeping | 0.9158 | 0.9324 | +0.017 |
| **two_frame_extended_comfort** | **0.8765** | **0.1621** | **−0.714** |

拖分主因：**two_frame comfort 崩溃**。NC/DAC 等并未变差。

## 4. 诊断（排除实现 bug）

| 检查 | 结论 |
|---|---|
| P1 vs P2 评测 Hydra | 仅 ckpt / `experiment_name` 不同；zero、K、self_cond、nprop 一致 |
| 汇总 csv | 唯一文件，分数与 summary 一致 |
| 中心损失 `self.training` 门控 | 有意设计，不影响推理 |
| 轨迹运动学（proposals） | P2 相对 P1：加速度约 **×3**，jerk 约 **×4** |
| comfort 场景级 | 9905 场景中 **7196** 从通过→失败（1→0） |

→ **真实轨迹变抖**，不是 scorer / 脚本取错 ckpt。  
假设：**同配方下 center 损失短训有益，长训过拟合高频抖动**；「早期更高」待中间 epoch 验证。

### 进行中的验证

- 评测 `p2_center18/epoch=04-step=21520.ckpt`（与 P1 同 step）→ `exp/ch3_df/navtest-p2_ep04-zero/`  
- 若 ≈0.71：坐实 early-stop；论文用 ~5ep，弃用 ep17。

## 5. 结论与下一步

1. **不要**再盲开 center×18；P2 作为负面结果保留。
2. 主推绝对分仍用 **P0 zero = 0.7657**；方法消融用 **P1 ±0.073**。
3. 可选：扫 ep8/12、降 `fm_center_weight`、训评 K 对齐；冲分走 **Phase 3 ViT-L**。

## 6. 产物路径

```
exp/ch3_df/p2_center18/epoch={00..17}-*.ckpt
exp/ch3_df/navtest-p2_center18-zero/   # ep17 评测
exp/ch3_df/summary_phase2.csv
exp/logs/ch3_df_phase2_center18_20261029_135213.log
```
