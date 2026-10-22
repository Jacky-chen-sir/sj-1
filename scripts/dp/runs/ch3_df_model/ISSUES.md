# `sj-df_model` 审阅意见（更新 2026-10-22 晚）

相对 `a7d829a` 再审。此前标注的 A–F 大多已在分支里改过；本轮又抓到 **一个会让「显式 seed」形同虚设的 bug**，已修。

---

## 本轮新发现（已修）

### H. 只 `export PL_GLOBAL_SEED` **不会**固定采样 —— **已修**

`a7d829a` 的注释写成「Lightning Trainer(seed_everything=True) 读 PL_GLOBAL_SEED」。  
本环境是 **PL 2.2.1**，`Trainer` **没有** `seed_everything` 参数；`run_pdm_score_gpu_v2.py` 原先也**从不**调用 `pl.seed_everything`。

因此：

- 只 export 环境变量 → **主进程 RNG 不受影响**；
- `SEED=1001/1002/1003` 对 first×N **原本无效**；
- 「不设 seed 会方差=0」也不对：未调用 `seed_everything` 时每次进程启动是未固定 RNG，方差来自偶然，不可复现。

**修复**：

1. `run_pdm_score_gpu_v2.py`：若 `PL_GLOBAL_SEED` 非空，则 `pl.seed_everything(...)`；
2. `eval_dp_fm_joint_navtest.sh`：仅当调用方传了 `SEED=` 才 export；未传则不固定（保持历史评测习惯）。

`run_phase0.sh` 各档已传 `SEED=`，修好后 first×N 才真正可复现。

### I. 官方对照用 `num_proposals=1` 对不上 0.7305 → **已改 100**

历史官方分是默认 100 提案取 `[0]`。确认基线应用 `NUM_PROPOSALS=100`（已改 `run_phase0.sh`）。

---

## 此前已关闭（无需再改）

| ID | 内容 |
|---|---|
| P0 | EXP_NAME 碰撞 / zero×100 / 无驱动脚本 |
| A/B | first 省算力用 1 提案、独立 subscore |
| C/D | 显式 seed 意图、官方 DDPM 档（机制见 H） |
| E/F | 措辞降调 |

关于 `num_proposals=1` ≠「100 取第 0 条」：RNG 按 `(B*P,…)` 排布，表述正确；与 0.7380 比是分布层比较，增益贴 0.01 时再补 100 提案配对。

---

## 残余（不挡开跑）

| 点 | 说明 |
|---|---|
| `DP_PROPOSAL_AGG` 仅环境变量 | 已 export+echo；未进 Hydra |
| CUDA 非确定性 | 同 seed 仍可能有微小浮点差；估方差足够 |
| medoid 距离在插值轨迹 | 可接受 |
| CH3_DRAFT 个别「分布中心」 | 草案语气，出表后再收 |

---

## 结论

| 项 | 状态 |
|---|---|
| Phase 0 能否开跑 | **可以**（SEED 现已真正生效） |
| 阻塞项 | **无** |
| 下一步 | `run_phase0.sh` 出表 → 再决定 Phase 1 |
