# Phase 1 结果：中心一致性训练（5 epoch 趋势对照）

> 状态：**已完成**。判据通过 → **进入 Phase 2**（`p2_center18`，18 epoch）。  
> 协议：navtest、one-stage fallback、`DP_EVAL_ZERO_NOISE=true`、同一 metric cache。  
> 机器汇总：`exp/ch3_df/summary_phase1.csv`（本目录副本 `summary_phase1.csv`）。

## 1. 设置（与 `run_phase1.sh` / 实际跑通配置一致）

两臂**同一基准**，只差 `fm_center_weight`：

| 项 | 值 |
|---|---|
| 初始化 | 官方 `gtrs_dp_model.ckpt` + `reinit_traj_head=true` |
| 预算 | **5** epoch（`epoch=04-step=21520`） |
| GPU / BS | 3×3090，bs=8 |
| `fm_center_weight` | base **0** / center **0.5** |
| `fm_center_steps` / `p` / warmup | 4 / 1.0 / 2000 |
| `FM_SELF_CONDITIONING` | **true**（启动环境带入；两臂一致。PLAN 默认 false，与 P0 口径不同，见 §4） |
| 评测 | zero-noise，`num_proposals=1`，seed=3000，K=20 |
| 对照参考 | Phase0 `p0_zero` = **0.7657**（成熟 FM ep18，非同训程） |

## 2. 总分

| arm | epochs | score (zero) | 相对 base |
|---|---:|---:|---:|
| `p1_base` | 5 | **0.6399** | — |
| `p1_center` | 5 | **0.7125** | **+0.0726** |

**判据**（PLAN：`P1_center − P1_base > +0.005`）：**通过**（实测约 **14×** 门槛）。

相对 Phase0 成熟 zero（0.7657）：center **−0.053**，base **−0.126**。  
属短预算重训头预期落差，**不是**相对判据失败；冲绝对分是 Phase 2 的事。

## 3. 分项（stage_one 汇总行）

| 子指标 | base | center | Δ (c−b) | p0_zero（参考） |
|---|---:|---:|---:|---:|
| score | 0.6399 | 0.7125 | **+0.0726** | 0.7657 |
| no_at_fault_collisions | 0.9506 | 0.9725 | +0.0219 | 0.9824 |
| drivable_area_compliance | 0.8618 | 0.9045 | **+0.0427** | 0.9439 |
| driving_direction_compliance | 0.9744 | 0.9921 | +0.0176 | 0.9949 |
| traffic_light_compliance | 0.9960 | 0.9977 | +0.0018 | 0.9983 |
| ego_progress | 0.8847 | 0.8761 | −0.0086 | 0.8552 |
| time_to_collision_within_bound | 0.9448 | 0.9618 | +0.0170 | 0.9796 |
| lane_keeping | 0.9297 | 0.9158 | −0.0139 | 0.9600 |
| history_comfort | 0.9816 | 0.9827 | +0.0011 | 0.9836 |
| two_frame_extended_comfort | 0.6247 | 0.8765 | **+0.2518** | 0.8856 |

解读：

- 涨分主要来自 **DAC**、**two_frame comfort**、NC/TTC/方向合规；与「中心轨迹更稳、更合规」一致。
- 略牺牲 ego_progress 与 lane_keeping（与 Phase0 zero 相对 first 的取舍同向）。
- base 的 two_frame comfort 很低（0.62），说明无 center 损失时短训零噪声路径抖动大；center 拉回至接近 p0_zero（0.88）。

## 4. 是否符合预期

| 问题 | 结论 |
|---|---|
| 相对判据（center > base +0.005）？ | **是，远超预期（+0.073）** |
| 方法假设（中心一致性有效）？ | **是**；安全/合规/舒适分项同向 |
| 绝对分追上 P0 zero？ | **否，且本阶段不要求**；5ep 重 init 头天然偏低 |
| 与 PLAN 默认 `self_cond=false`？ | 实际两臂均为 `true`；**臂间对照仍公平**，与 P0 不可直接混表 |

→ **进入 Phase 2**：同配方拉满 18 epoch（`p2_center18`），对照已有 `P2_fm18`（ep18 zero 0.7657）。

## 5. 产物路径

```
exp/ch3_df/p1_base/epoch=04-step=21520.ckpt
exp/ch3_df/p1_center/epoch=04-step=21520.ckpt
exp/ch3_df/navtest-p1_base-zero/   # csv 2026.10.28.19.50.45
exp/ch3_df/navtest-p1_center-zero/ # csv 2026.10.28.23.28.25
exp/ch3_df/summary_phase1.csv
exp/logs/ch3_df_phase1_e5_20261023_195410.log
```

训练时段：base ≈ 2026-10-23 → 10-26；center ≈ 10-26 → 10-28；评测 10-28 当天完成。
