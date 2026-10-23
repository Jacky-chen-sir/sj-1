# Phase 0 结果：推断方式对照（2026-10-22 → 10-23）

> 状态：**已完成**。判据通过 → **进入 Phase 1**（中心一致性训练）。  
> 协议：navtest、one-stage fallback、同一 metric cache。  
> 机器汇总：`exp/ch3_df/summary_phase0.csv`（本目录亦有副本 `summary_phase0.csv`）。

## 1. 设置（与 `run_phase0.sh` 一致）

| 项 | 值 |
|---|---|
| FM ckpt | `train_dp_official_fm_joint/epoch=18-step=81776.ckpt` |
| 官方 ckpt | `GTRS_official/data/models/gtrs_dp_model.ckpt` |
| `FM_SELF_CONDITIONING` | `false`（对齐 ep18） |
| FM 推理步数 K | 20 |
| 官方 DDPM 步数 | 100 |
| first / zero | `num_proposals=1` |
| medoid / official | `num_proposals=100` |
| seed | first: 1001/1002/1003；medoid 2000；zero 3000；official 1001 |

说明：`first` 用 1 提案与历史「100 取第 0 条」**不等价**（RNG 按 `B×P` 排布）；与历史 0.7380 的比较是分布层面的。官方对照用 100 提案以对齐历史 0.7305 口径。

## 2. 总分

| tag | 推断 | score | 相对 FM first 均值 |
|---|---|---:|---:|
| `p0_official_first` | 官方 DDPM，`proposals[0]` | **0.7317** | −0.0101 |
| `p0_first_1` | FM 单样本 seed=1001 | 0.7412 | — |
| `p0_first_2` | FM 单样本 seed=1002 | 0.7424 | — |
| `p0_first_3` | FM 单样本 seed=1003 | 0.7417 | — |
| FM first 汇总 | mean ± std（n=3） | **0.7418 ± 0.0006** | — |
| `p0_medoid` | 100 条 medoid | **0.7582** | **+0.0164** |
| `p0_zero` | 零噪声确定性路径 | **0.7657** | **+0.0239** |

**判据**（PLAN：medoid 或 zero 相对 first > +0.01）：**通过**。  
单样本方差极小（std 0.0006）；涨分来自换推断方式，不是运气。

## 3. 分项（stage_one 汇总行）

| 子指标 | official | first 均值 | medoid | Δmed | zero | Δzero |
|---|---:|---:|---:|---:|---:|---:|
| score | 0.7317 | 0.7418 | 0.7582 | +0.0164 | 0.7657 | +0.0239 |
| no_at_fault_collisions | 0.9712 | 0.9770 | 0.9792 | +0.0022 | 0.9824 | +0.0054 |
| drivable_area_compliance | 0.9283 | 0.9259 | 0.9389 | +0.0129 | 0.9439 | +0.0179 |
| driving_direction_compliance | 0.9859 | 0.9930 | 0.9947 | +0.0017 | 0.9949 | +0.0018 |
| traffic_light_compliance | 0.9970 | 0.9977 | 0.9981 | +0.0004 | 0.9983 | +0.0007 |
| ego_progress | 0.8694 | 0.8635 | 0.8647 | +0.0012 | 0.8552 | −0.0084 |
| time_to_collision_within_bound | 0.9683 | 0.9721 | 0.9758 | +0.0037 | 0.9796 | +0.0075 |
| lane_keeping | 0.9506 | 0.9536 | 0.9598 | +0.0062 | 0.9600 | +0.0065 |
| history_comfort | 0.9828 | 0.9829 | 0.9835 | +0.0006 | 0.9836 | +0.0007 |
| two_frame_extended_comfort | 0.8309 | 0.8563 | 0.8848 | +0.0285 | 0.8856 | +0.0293 |

解读：

- 涨分主要来自 **DAC** 与 **two_frame comfort**；zero 略牺牲 progress，换更稳合规。
- 场景级 DAC=0：first≈891 → medoid 733 → zero **673** / 11989。
- 官方 0.7317 ≈ 历史复现 0.7305；FM first 0.7418 略高于历史单次 0.7380（设置/种子不同，属分布层差异）。

## 4. 结论与下一步

1. **历史 0.738 是单样本推断分，不是模型上限**；换成 medoid / zero 可免费抬 **+1.6 / +2.4** 分。
2. zero（确定性、1 次前向）略优于 medoid（100 次前向），论文主推 **零噪声确定性路径**，medoid 作多样本对照。
3. **进入 Phase 1**：中心一致性训练，让零噪声路径在训练里也成为好解；评测默认改为 zero（或与 first 对照表）。

## 5. 产物路径

```
exp/ch3_df/p0_first_{1,2,3}/
exp/ch3_df/p0_medoid/
exp/ch3_df/p0_zero/
exp/ch3_df/p0_official_first/
exp/ch3_df/summary_phase0.csv
exp/logs/ch3_df_phase0_20261022.log
```

每档另有 `run_meta.txt`（seed / agg / nprop / ckpt）。
