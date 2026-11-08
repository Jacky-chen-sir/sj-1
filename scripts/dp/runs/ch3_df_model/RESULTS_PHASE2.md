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

| 模型 | epochs | score (zero) | comfort | vs P1 center |
|---|---:|---:|---:|---:|
| P0 FM zero（无 center 损失） | 18 (旧) | **0.7657** | 0.8856 | — |
| P1 center | 5 | **0.7125** | 0.8765 | — |
| **P2 ep04**（同 step 早停对照） | 5 | **0.6958** | **0.8918** | −0.017 |
| **P2 ep17**（拉满） | 18 | **0.6621** | **0.1621** | −0.050 |

**预期**：拉满后逼近/超过 P0 zero，至少不低于 P1。  
**实际**：明显低于二者 → **长训策略失败**（非评测管道错误，见 §4）。

> ⚠️ **2026-11-08 更正**：`+0.073` 与 `−0.050` 这两个差值都**不能当稳健估计**。
> - comfort 是阈值型指标（单场景过/不过），相邻 ckpt 在 0.16~0.88 之间摆动；
> - P1 与 P2 用同一 seed=0，P2 的前 5 epoch 就是 P1，两者不是独立观测；
> - 三次实验的 `overrides.yaml` 逐行只差 name / ckpt_path / max_epochs，其余一致。
>
> 要看的是「跨 ckpt 是否稳定」，而不是某两个数字之差。下列结论目前只有
> 「ep17 的 comfort 明显低于 ep4」这一条有直接证据。

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
可能机制（**未验证**）：
1. **训练/评测步数不一致**：中心损失只在 K=4 的路径上施加，评测用 K=20。
   模型在 4 步里要把 20 步的积分误差一并补掉，学到的速度场在细步长下偏。
2. 自洽分支在校验期也在生效（当时无 `self.training` 门控），`val/loss` 的
   ckpt 选择信号本身是随机的。

两条都已在 2026-11-08 的代码里修掉（`fm_center_k_set` / 校验门控 / `val/fm_zero_*`）。
旧 ckpt 可以直接拿来验 (1)：`K_LIST="4" ARMS="p2_center18" EPOCHS_LIST="17" bash eval_ckpts.sh`，
若 K=4 的分数明显高于 K=20，则 (1) 成立。

### ep04 验证（2026-11-09 完成）

评测：`p2_center18/epoch=04-step=21520.ckpt` → `exp/ch3_df/navtest-p2_ep04-zero/`  
（与 P1 同 step；zero-noise、K=20、`self_cond=true`、seed=3000）

| | score | two_frame_comfort | DAC |
|---|---:|---:|---:|
| P1 center ep4 | 0.7125 | 0.8765 | 0.9045 |
| **P2 ep04** | **0.6958** | **0.8918** | 0.8781 |
| P2 ep17 | 0.6621 | 0.1621 | 0.9143 |

**结论**：early-stop 方向成立——**同一次 18ep 训练里，ep04 ≫ ep17**（comfort 0.89 vs 0.16）；
ep04 与独立 P1 center 接近（−0.017）。不是「方法从一开始就坏」，而是 **拉长训练后轨迹变抖**。

## 5. 结论与下一步

1. **不要**再盲开 center×18；选 ckpt 看 comfort / `val/fm_zero_jerk`，不要只看 ADE。
2. 主推绝对分仍用 **P0 zero = 0.7657**。
3. 已用修复后代码重跑短训：`ch3_df_v2`（`k_set=[4,8,20]`、确定性探针选 ckpt）。
4. 可选：扫 ep8/12；冲分走 **Phase 3 ViT-L**。

## 6. 产物路径

```
exp/ch3_df/p2_center18/epoch={00..17}-*.ckpt
exp/ch3_df/navtest-p2_center18-zero/   # ep17 → 0.6621
exp/ch3_df/navtest-p2_ep04-zero/       # ep04 → 0.6958
exp/ch3_df/summary_phase2.csv
exp/logs/ch3_df_phase2_center18_20261029_135213.log
exp/logs/ch3_df_p2_diag_ep04_*.log
```
