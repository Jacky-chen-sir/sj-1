# `sj-df_model` 审阅意见（2026-10-22）

分支意图清楚：Phase 0 用「first / medoid / zero-noise」对照单样本推断，不训模型。  
下面只列**我觉得不对或会踩坑**的点，按严重程度排。

---

## P0 — 不修会直接搞坏实验

### 1. 默认 `EXP_NAME` 不含推断方式 → 评测会互相 SKIP / 覆盖

`eval_dp_fm_joint_navtest.sh` 默认目录只有 `ckpt + fm_steps + bs`，**不含** `DP_PROPOSAL_AGG` / `DP_EVAL_ZERO_NOISE`。  
脚本又有「已有 CSV 则 SKIP」逻辑。

按 PLAN 里三条命令原样跑（都不设 `EXP_NAME`）时：

- 第一次 `first` 写完 CSV 后，
- 再跑 `medoid` / `zero` 会被 **SKIP**，或写进同一目录，**对照表作废**。

`summarize_phase0.py` 期望的是 `ch3_df/p0_first_*`、`p0_medoid`、`p0_zero`，但 PLAN 用法里**没有**对应的 `EXP_NAME=ch3_df/...`，驱动脚本也缺失。

**要求**：Phase 0 每次必须显式设互不冲突的 `EXP_NAME`（见下方推荐命令）。  
评测脚本侧已补：默认名带上 `-agg*` / `-zero` 后缀，降低误用概率；仍以显式 `EXP_NAME` 为准。

### 2. `zero_noise=true` 仍用 `num_proposals=100` → 100 条完全相同

`dp_model.py` 在置零时对 `B * NUM_PROPOSALS` 整块 `zeros`。  
默认 100 提案 → **100 条同一条轨迹**，算力白烧；再叠 `medoid` 无意义（距离矩阵全 0）。

**要求**：`P0_zero` 应 `++agent.config.num_proposals=1`（或等价）。  
PLAN 原先没写。

### 3. 缺少 Phase 0 一键脚本

有 `summarize_phase0.py`，没有 `run_phase0.sh`。  
手册三条命令也不导出 `FM_SELF_CONDITIONING` / 独立 `EXP_NAME`，和 ep18 历史口径容易不一致。

---

## P1 — 概念 / 表述过满

### 4. 「零噪声 = 分布中心」说重了

FM 从 `x_T=0` 积分，只是一条**确定性路径**，一般**不等于** \(\mathbb{E}[x\mid c]\)，也不是后验众数。  
DDPM 从全零起步更是相对标准 \(x_T\sim\mathcal{N}(0,I)\) 的 **OOD 起点**。  
草案/PLAN 里「中心路径 / 分布中心 / 免费涨分」容易写成已证结论；Phase 0 前应降调为**假设**。

### 5. 「中位数轨迹」≠ 实现

实现是 **medoid**（到其余样本距离和最小），不是逐时刻坐标中位数，也不是均值轨迹。  
SURVEY / CH3_DRAFT 混用「中位数」会误导审稿人；对外统一叫 medoid / 中心样本。

### 6. 「0.7305 / 0.7380 带采样噪声，换中心就可能免费涨分」

- 单次 `proposals[0]` 确有随机性，重复评可估方差 —— 这部分成立。  
- **涨分是待证假设**，不是核心发现。medoid 也可能更保守（进度↓、合规↑），总分不一定升。  
- 与已有 ep18=0.7380 比绝对分时，必须对齐当时推理开关（见下条）。

### 7. SURVEY 分数口径混用

表里 TransFuser 等写 **PDMS ~84**，本仓库目标写 **navtest ~0.73**。  
不是同一套指标/归一化，并列表易让人以为「还差 10 分」。需标注 benchmark / metric。

---

## P2 — 工程细节

### 8. `DP_PROPOSAL_AGG` 只读环境变量

评测脚本 Hydra 只透传了 `dp_eval_zero_noise`，**没有**把 agg 写进 config / 日志。  
漏 `export` 时静默退回 `first`，目录名若也不区分，难排查。  
建议：进 `DPConfig` + 评测 override，或至少在脚本 `echo` 里打印当前 agg。

### 9. 与 ep18 历史分数对齐

当前评测默认 `FM_SELF_CONDITIONING=true`（config 默认亦 true）。  
出 0.7380 的那次训练 override **只有** `use_flow_matching`（自条件是后加的）。  
Phase 0 若要和 0.7380 比绝对分：`FM_SELF_CONDITIONING=false`（或与当时 ckpt 结构一致）。  
三档相对比较时三者一致即可，但正文不要把「换推断」和「换自条件」搅在一起。

### 10. 与 `sj-opd` 引导线的关系没写清

`sj-opd` 上第三章曾是「评价器引导 + 时间门控」；本分支改成「确定性推断 / 中心一致性」。  
PLAN 未说明是**替换主线**还是**并行备选**。写论文前要定口径，避免两章抢同一个「第三章」。

### 11. medoid 距离

对插值轨迹 XY 做 pairwise MSE 再 sum，实现合理；`final_pose` 取的是未插值 `proposals[sel]`，与距离空间不一致但索引对齐，可接受。  
P=100 时 `[P,P,T,2]` 临时张量不大；若以后加大 P 再优化。

---

## 推荐的 Phase 0 命令（修完用法后）

```bash
CKPT=$NAVSIM_EXP_ROOT/train_dp_official_fm_joint/epoch=18-step=81776.ckpt
FM_SELF_CONDITIONING=false   # 与出 0.7380 时对齐；三档保持一致

# first ×3（估方差）
for i in 1 2 3; do
  FORCE_RERUN=1 DP_PROPOSAL_AGG=first DP_EVAL_ZERO_NOISE=false \
  EXP_NAME=ch3_df/p0_first_$i \
  CKPT=$CKPT bash scripts/evaluation/eval_dp_fm_joint_navtest.sh
done

# medoid
FORCE_RERUN=1 DP_PROPOSAL_AGG=medoid DP_EVAL_ZERO_NOISE=false \
EXP_NAME=ch3_df/p0_medoid \
CKPT=$CKPT bash scripts/evaluation/eval_dp_fm_joint_navtest.sh

# zero-noise（单提案）
FORCE_RERUN=1 DP_PROPOSAL_AGG=first DP_EVAL_ZERO_NOISE=true \
EXP_NAME=ch3_df/p0_zero \
CKPT=$CKPT \
bash -c 'source scripts/evaluation/eval_dp_fm_joint_navtest.sh'  # 需在 hydra 加 num_proposals=1
# 更稳妥：改评测脚本支持 NUM_PROPOSALS=1，或临时 ++agent.config.num_proposals=1
```

汇总：

```bash
python scripts/dp/runs/ch3_df_model/summarize_phase0.py \
  --root $NAVSIM_EXP_ROOT/ch3_df \
  --out  $NAVSIM_EXP_ROOT/ch3_df/summary_phase0.csv
```

---

## 小结

| 级别 | 结论 |
|---|---|
| 方向 | 可做：先量化单样本方差，再谈中心推断 / 一致性训练 |
| 代码骨架 | first/medoid/zero 开关基本可用 |
| 现在不对的 | 默认实验名碰撞、zero×100 提案、手册与 summarize 路径脱节、把假设写成「核心发现」、术语与指标口径不严谨 |

**建议**：先按上面命令把 Phase 0 跑通并出表，再决定要不要 Phase 1；在那之前不要把「免费涨分」写进正文。
