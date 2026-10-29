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

## 6. 复核意见（2026-10-29）

结论成立（+0.073 远大于判据），Phase 2 照常跑。以下三点写论文前要补或要注意：

**① 只评了一个 ckpt、只评了 zero。**
`summary_phase1.csv` 是旧版 `run_phase1.sh` 的输出（表头 `arm,epochs,score,...`），每臂只评了
`epoch=04` 的 zero。同一训程相邻 ckpt 波动可达 0.01–0.03（历史 ep11–19 在 0.705–0.738），
+0.073 远大于这个量级，结论不会翻；但论文表格需要“最后 3 个 epoch 的均值±标准差”。
另外中心损失**直接优化的就是 zero 路径**，只报 zero 容易被问“是不是只是对齐了评测规则”。
补一组 `first`（单样本）：若 center 在 first 上也涨，说明整个分布变好了，论证更硬。

```bash
# 两臂最后 3 个 epoch × {zero, first}，空闲卡上跑，不影响 Phase 2
ARMS="p1_base p1_center" GPU=<空闲卡> bash scripts/dp/runs/ch3_df_model/eval_ckpts.sh
```

**② base 臂 5ep 的绝对分偏低，增益幅度会随训练变长收窄。**
p1_base 0.640 明显低于 P0 first（0.742）；差距主要在 two_frame_comfort（0.62 vs 0.86）。
说明短训时零噪声路径本身还很抖，中心损失恰好在修这一点。长训后 base 自己也会变好，
所以 +0.073 是“5ep 同预算”的结论，论文里要注明预算；长预算增益以 Phase 2 为准。

**③ Phase 2 目前缺同配方的 base 臂。**
`p2_center18` 的对照是 FM-only ep18（0.7657），但两者配方不同（ep18 无 aux/kin/自洽，
self_cond=false；P2 全配方 + self_cond=true），差值不能归因于中心损失。
- 若 Phase 2 > 0.7657：可直接写“完整方法超过 FM 基线”，不需要归因到单项。
- 若要归因中心损失的长预算增益：需要 `p2_base18`（center_w=0，其余同 P2），再占 8–9 天。
  建议先等 P2 结果，再决定是否补。

**Phase 2 选点。** 训练中可用空闲卡评已落盘的中间 epoch，不必等到最后：

```bash
ARMS="p2_center18" EPOCHS_LIST="8 12 15 17" INFER="zero" GPU=<空闲卡> \
  bash scripts/dp/runs/ch3_df_model/eval_ckpts.sh
```

注意：`run_phase2.sh` 只评“最新”一个 ckpt，且用 `find|head` 取 csv（重跑时可能取到旧的）。
它正在跑，没有改；评测改用上面的 `eval_ckpts.sh`。
