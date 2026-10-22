# 第三章重构计划：确定性推断与中心一致性

> 分支：`sj-df_model`
> 设备：3×RTX 3090　目标：**单独生成模型 > 0.73，越高越好**
> 定位：不接词表、不接评价器，纯轨迹生成。

## 0. 核心发现

DP 单独评测原本取 `proposals[0]`（100 条随机采样中的第一条），代码注释写着
“randomly choose a dp proposal”。这意味着：

1. **已有的 0.7305 / 0.7380 带有采样噪声**，不是模型的确定性输出；
2. **另外 99 条提案被丢弃**；
3. 换成确定性路径或 medoid 是**待验证的提升假设**，不是已证结论。

## 1. 代码改动（已在 `sj-df_model` 落地）

| 文件 | 改动 |
|---|---|
| `dp_config.py` | 新增 `dp_eval_zero_noise`（推理起点置零） |
| `dp_model.py` | 置零时用 `torch.zeros` 作为 ODE 起点（确定性路径；≠条件均值） |
| `agent_lightning_module.py` | `DP_PROPOSAL_AGG` 环境变量：`first`（默认）/ `medoid` |
| `eval_dp_fm_joint_navtest.sh` | 暴露 `DP_EVAL_ZERO_NOISE` |

**用法**（必须带互不冲突的 `EXP_NAME`；细节与已知问题见同目录 `ISSUES.md`）：

```bash
CKPT=$NAVSIM_EXP_ROOT/train_dp_official_fm_joint/epoch=18-step=81776.ckpt
# 与历史 0.7380 对齐时建议 FM_SELF_CONDITIONING=false；三档保持一致
export FM_SELF_CONDITIONING=false

# 官方行为（取第 0 条）×3 估方差
for i in 1 2 3; do
  FORCE_RERUN=1 DP_PROPOSAL_AGG=first EXP_NAME=ch3_df/p0_first_$i CKPT=$CKPT \
    bash scripts/evaluation/eval_dp_fm_joint_navtest.sh
done

# medoid（到其余提案距离和最小；不是坐标中位数）
FORCE_RERUN=1 DP_PROPOSAL_AGG=medoid EXP_NAME=ch3_df/p0_medoid CKPT=$CKPT \
  bash scripts/evaluation/eval_dp_fm_joint_navtest.sh

# 零噪声确定性路径（脚本在 zero 时自动 num_proposals=1）
FORCE_RERUN=1 DP_EVAL_ZERO_NOISE=true EXP_NAME=ch3_df/p0_zero CKPT=$CKPT \
  bash scripts/evaluation/eval_dp_fm_joint_navtest.sh
```

## 1.5 与 `sj-opd` 分支的关系

本分支的第三章主线**取代** `sj-opd` 上的「评价器引导 + 时间门控」线。
`sj-opd` 的引导实验保留为独立对照，不进入第三章正文；论文里只保留一套第三章口径。

## 2. 实验计划

### Phase 0 — 推断方式对照（零训练，必跑）

用现有 FM-DP `epoch=18-step=81776.ckpt`（0.7380）跑三档：

| ID | 推断 | 说明 |
|---|---|---|
| `P0_first` | `proposals[0]` | 现状，且**重复 3 次**以估计方差 |
| `P0_medoid` | 100 条的 medoid（到其余样本距离和最小） | 降方差 |
| `P0_zero` | 零噪声确定性路径（先验均值起点） | 确定性输出 |

同时用官方 DP（DDPM）ckpt 跑一次 `first`，确认基线 0.7305。

**判据**：
- 若 `medoid` 或 `zero` 比 `first` 高 > 0.01 → 推断退化是本 chapter 的核心问题，进入 Phase 1。
- 若三者接近 → 生成式模型的单样本已经足够好，转 Phase 3（换骨干）。

**成本**：3–5 次评测，每次约数小时（纯推理），**不需要训练**。

**一键执行**：

```bash
CKPT=$NAVSIM_EXP_ROOT/train_dp_official_fm_joint/epoch=18-step=81776.ckpt \
  bash scripts/dp/runs/ch3_df_model/run_phase0.sh
```

脚本自动：
- 三档使用互不冲突的 `EXP_NAME`（`ch3_df/p0_*`）；
- 统一 `FM_SELF_CONDITIONING=false`（与 ep18 对齐）；
- `first` / `zero` 用 `num_proposals=1`，`medoid` 用 100；
- 每档独立 `SUBSCORE_PATH`，避免互相覆盖 pickle；
- 跑完调用 `summarize_phase0.py` 出表。

> 官方 DDPM `first` 确认 0.7305 尚未纳入本脚本，需另跑（`USE_FM=false` + 官方 ckpt）。

### Phase 1 — 中心一致性训练（Phase 0 有效才跑）

目标：让“中心轨迹”本身成为一个好解，而不是只在推理时聚合。

| ID | 配置 | 预算 |
|---|---|---|
| `P1_center` | 在 FM 主损失之外，对**零噪声路径的还原轨迹**加辅助监督（与轨迹辅助同口径） | 18 epoch |
| `P1_base` | 同一配方但不加中心约束 | 18 epoch |

两臂都从官方 ckpt + 重初始化轨迹头起，同预算。

**判据**：`P1_center − P1_base > +0.005`。

### Phase 2 — 长预算复现（按需）

| ID | 配置 |
|---|---|
| `P2_fm18` | FM-only 18 epoch（= ep18，已有 0.7380） |
| `P2_center18` | Phase 1 的有效配置跑满 18 epoch |

### Phase 3 — 换基座（底线保障）

把 DP 的 BEV 骨干从 VoV-99 扩到 **Depth Anything ViT-L**（`da_vitl16.pth` 权重已有）。

需要改 `HydraBackboneBEV`：新增 `backbone_type == 'vit'` 分支，接 `DAViT`，
并对齐 `img_feat_c`、`avgpool` 与 BEV 融合层的维度。**这是工程改动，需在服务器上验证。**

| ID | 配置 |
|---|---|
| `P3_vitl` | ViT-L 骨干 + FM 解码器，18 epoch |

预期：公开结果显示 R34 → ViT-L 有 +3–4 分空间（在带评分器的系统里）。

## 3. 执行顺序

```
① Phase 0（零训练，先做）
   P0_first ×3（估方差） / P0_medoid / P0_zero
② 判读：
   ├─ medoid 或 zero 明显更好 → Phase 1
   └─ 三者接近 → 直接 Phase 3
③ Phase 1（18 epoch × 2 臂）
④ Phase 3（换 ViT-L，工程 + 18 epoch）
```

## 4. 成本

| 阶段 | 训练 | 推理 | 时间 |
|---|---|---|---|
| Phase 0 | 0 | 4–5 次 | 1 天 |
| Phase 1 | 2 × 18 epoch | 2 次 | 8–10 天 |
| Phase 3 | 1 × 18 epoch | 1 次 | 5 天 + 工程 |

## 5. 论文怎么写

**主线**：
> 生成式轨迹规划器在推理时从先验随机采样一条轨迹，单样本方差大、且丢弃了其余样本。
> 本文（1）量化这一退化；（2）提出中心推断（零噪声 / medoid）；（3）用中心一致性训练
> 让分布中心本身成为好解；（4）在更换更强视觉基座后进一步抬高上限。

**与现有工作的区别**：
- GoalFlow / DiffusionDrive / MeanFuser 关注“怎么生成 / 怎么用一步”，**没有研究单样本推断的统计**；
- 本文不改生成框架、不接评分器，只改“从分布里取哪个点”，并把它写进训练目标。

**实验**：
- 表 1：推断方式对照（first ×3 / medoid / zero），带均值±标准差；
- 表 2：中心一致性训练 vs 基线；
- 表 3：骨干升级（VoV-99 vs ViT-L）；
- 图：单样本轨迹散布 vs 中心轨迹。

## 6. 风险

| 风险 | 应对 |
|---|---|
| 分布多模态，均值落在模态之间 | 用 medoid，不用均值 |
| medoid 提升不明显 | 转 Phase 3（换骨干） |
| ViT-L 接入出现维度/显存问题 | 先只换骨干、冻结解码器跑 1 epoch 验证 |
| 18 epoch 成本高 | 先用 5 epoch 看趋势，再决定是否拉满 |
