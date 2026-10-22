# `sj-df_model` 审阅意见（更新 2026-10-22）

相对 `78c363e`（已修上一轮三个 P0）再审一遍。  
**上一轮已关闭**：互斥 `EXP_NAME` + `run_phase0.sh`、zero→`num_proposals=1`、术语降调、SURVEY 口径、与 `sj-opd` 关系。

下面是**现在还不对 / 仍会踩坑**的点。

---

## 本轮已补（相对 `78c363e`）

### A. `first` 只用第 0 条却生成 100 条 → **已在 `run_phase0.sh` 设 `NUM_PROPOSALS=1`**

`medoid` 仍为 100；`zero` 显式 1。

### B. 共享 `SUBSCORE_PATH` → **已改为每档 `$REL/$tag/proposals.pkl`**

### 仍开放

### C. `first×3` 没有显式种子策略

`run_pdm_score_gpu_v2.py` 不调用 `seed_everything`，三次重复依赖进程启动时的 RNG，通常会不同，但：

- 不可复现；
- 也没在日志里记下 seed；
- 若以后某处加了全局固定 seed，三次会撞成同一分，方差估成 0。

**建议**：`p0_first_i` 设 `SEED=1000+i`（或等价 Hydra `seed=`），并写进 `EXP_NAME`/日志。

### D. PLAN 写了「官方 DP first 确认 0.7305」，一键脚本没跑

`run_phase0.sh` 只有 FM ep18 三档。官方 DDPM 对照需另开（`USE_FM=false` + 官方 ckpt），否则表不完整。

---

## 表述上仍偏满（不挡跑，写论文前改）

### E. SURVEY §2.1 仍写「几乎免费的提升被忽略了」

与 PLAN「待验证假设」口径不一致。应改成「可能的廉价对照 / 待 Phase 0 验证」。

### F. PLAN 改动表仍写「确定性中心路径」

`dp_model.py` 注释已降调；PLAN §1 表格一行仍用「中心路径」。统一成「确定性路径（先验零起点）」。

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
| 剩余 | first 浪费算力、共享 subscore、无 seed、缺官方 DP 对照、个别措辞 |

**建议跑法**：用 `run_phase0.sh`，并先补上 A/B（C 更稳）；出表后再决定 Phase 1。
