# `sj-df_model` 审阅意见（更新 2026-10-22）

相对 `78c363e`（已修上一轮三个 P0）再审一遍。  
**上一轮已关闭**：互斥 `EXP_NAME` + `run_phase0.sh`、zero→`num_proposals=1`、术语降调、SURVEY 口径、与 `sj-opd` 关系。

下面是**现在还不对 / 仍会踩坑**的点。

---

## 本轮已补（相对 `78c363e`）

### A. `first` 只用第 0 条却生成 100 条 → **已在 `run_phase0.sh` 设 `NUM_PROPOSALS=1`**

`medoid` 仍为 100；`zero` 显式 1。

> ⚠ 修正审阅里的一个说法（本轮核对代码时发现）：`num_proposals=1` **并不等价**于
> 「100 条取第 0 条」。`dp_model.py` 的噪声形状是 `(B*num_proposals, H, A)` 且按行优先生成，
> P 不同时每个场景（除 b=0）拿到的随机数不同。因此：
> - 与历史 0.7380 的比较是**分布层面**的，不是逐场景配对；
> - 若 medoid/zero 的增益接近 0.01 判据，应再用 `NUM_PROPOSALS=100` 补一组配对确认（seed 相同）。

### B. 共享 `SUBSCORE_PATH` → **已改为每档 `$REL/$tag/proposals.pkl`**

### C. `first×N` 没有显式种子策略 → **已修**

Lightning 2.x 的 `Trainer(seed_everything=True)` 读 `PL_GLOBAL_SEED`，**缺省固定为 0**。
所以不显式给 seed 时，重复评测会抽出**完全相同**的提案，方差会被估成 0（比"不可复现"更糟）。

- `eval_dp_fm_joint_navtest.sh` 新增 `SEED`（导出为 `PL_GLOBAL_SEED`）并打印；
- `run_phase0.sh` 给 `p0_first_i` 传 `SEED=1000+i`，medoid/zero 也各给固定 seed。

### D. PLAN 写了「官方 DP first 确认 0.7305」，一键脚本没跑 → **已修**

`run_phase0.sh` 新增 `p0_official_first`（`USE_FM=false` + `OFFICIAL_CKPT`，默认
`/home/ws/navsim_workspace/GTRS_official/data/models/gtrs_dp_model.ckpt`）；
`RUN_OFFICIAL=0` 可跳过。`summarize_phase0.py` 把它单列，**不计入** FM first 的均值/标准差。

---

## 表述上仍偏满（不挡跑，写论文前改）

### E. SURVEY §2.1 仍写「几乎免费的提升被忽略了」 → **已改**

现文已改为待验证口径，与 PLAN 一致。

### F. PLAN 改动表仍写「确定性中心路径」 → **已改**

PLAN 表格统一为「确定性路径（先验零起点；≠条件均值）」，Phase 0 表里的「先验均值起点」
也一并改掉——零起点不是条件均值，这一句在正文里不能当定理写。

### G. `ISSUES.md` 旧版把已修 P0 仍写成「开放」

（本文已改成「已关闭 + 剩余」；勿按旧 P0 清单再改一轮。）

---

## 可接受的残余（知悉即可）

| 点 | 说明 |
|---|---|
| `DP_PROPOSAL_AGG` 仅环境变量 | 评测脚本已 `export` 并 echo；未进 Hydra config，漏设会静默 `first` |
| `REUSE_SUBSCORE_IF_EXISTS` | 本评测路径里基本是死变量；真正复用靠 `SKIP_INFER` |
| zero ≠ 条件均值 | 文档已写清；勿在正文当定理 |
| medoid 距离在插值轨迹、输出取 raw proposals | 索引对齐，可接受 |
| 与历史 0.7380 比绝对分 | 必须 `FM_SELF_CONDITIONING=false`（脚本默认已是） |

---

## 当前结论

| 项 | 状态 |
|---|---|
| 方向 / 开关骨架 | 可用，可以开跑 Phase 0 |
| 上一轮 P0（撞名 / zero×100 / 无驱动） | **已修** |
| A/B（first=1 提案、独立 subscore） | **已修** |
| C（显式 seed） | **已修** |
| D（官方 DDPM 对照） | **已修** |
| 剩余 | 无阻塞项；出表后再决定 Phase 1 |

**建议跑法**：直接 `run_phase0.sh`（含官方对照）；出表后再决定 Phase 1。
