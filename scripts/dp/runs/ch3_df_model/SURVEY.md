# 第三章调研：单独训练的轨迹生成模型（截至 2026-10）

> 范围：**只训练一个决策模型、直接输出轨迹**，不使用轨迹词表、不使用评价器。
> 评测：NAVSIM navtest。
> 目标：分数必须 > 0.73（官方 GTRS-DP 单体），越高越好。

## 1. 方法谱系
> **口径说明**：下表的 PDMS/EPDMS 是文献报告值（NAVSIM navtest，官方评测口径与骨干随论文而定）；
> 本仓库第三章的 0.73–0.74 是**自复现管线**（one-stage fallback）下的分数，
> 两者**不可直接相减**，也不能据此得出「还差 10 分」的结论。


### 1.1 回归式（确定性）

| 代表 | 做法 | 参考分数 |
|---|---|---|
| TransFuser (2023) | Transformer 融合图像 + LiDAR，直接回归一条轨迹 | 84.0 PDMS（官方口径） |
| PARA-Drive (2024) | 并行回归 + 多任务 | 84.0 PDMS |
| UniAD (2023) | 感知-预测-规划串行 | 83.4 PDMS |

**特点**：稳定、延迟低；单条输出，无法表达多模态；路口等场景易退化为模态平均。

### 1.2 生成式（扩散 / 流匹配）

| 代表 | 做法 | 参考分数 |
|---|---|---|
| DiffusionDrive (CVPR 2025) | 锚点条件截断扩散，20 锚点，2 步去噪 | 88.1 PDMS |
| GoalFlow (CVPR 2025) | 目标点约束 + 流匹配，单步 | 90.3 PDMS |
| MeanFuser (2026-02) | 高斯混合噪声 + MeanFlow 一步生成 + 自适应重建 | — |
| Adaptive Time Step FM (2026-02) | 自适应时间步的流匹配 | — |
| CATG (2025-10) | 约束感知流匹配 | — |
| GuideFlow (2025-11) | 约束引导流匹配 | — |
| WAM-Flow (2025-12) | 离散流匹配 + 粗到细 | — |
| FlowR2A (2026-06) | 奖励条件流匹配（仿真奖励作为条件） | — |

**特点**：多模态表达强；但**单次随机采样**的方差大，且多数高分方法最终仍依赖一个选择/评分环节。

### 1.3 世界模型 / VLA（2026 主流）

`World4Scorer`、`WAC`、`PlanWAM`、`MORE`、`DriveWAM`、`MindVLA` 等。
**结论：3×3090 不可行**，且与“单独生成模型”的定位不符，本工作不采用。

## 2. 现有工作的共同盲点

### 2.1 单样本推断的随机性没有被研究

生成式规划器在推理时从先验采样一条轨迹。**同一 checkpoint、同一场景，不同随机种子会给出不同的轨迹和不同的分数**。但文献普遍只报一次评测的结果，没有：

- 量化单样本推断的方差；
- 比较“随机样本”与“确定性路径 / medoid 中心样本”；
- 把“推断退化”作为独立问题处理。

因此值得做一次**零训练对照**（多采几条取 medoid / 零噪声确定性路径 vs 只取第一条），
看单样本方差有多大、中心推断是否真能抬分——这是假设，不是已有结论。

### 2.2 代码层面的证据

本仓库的 DP 单独评测（`agent_lightning_module.py:predict_step_dp_traj`）原本是：

```python
# randomly choose a dp proposal
final_pose = proposals[0]
```

而模型一次生成 `num_proposals = 100` 条。**99 条被丢弃，分数取决于第 0 条。**
这意味着已有的 0.7305 / 0.7380 本身就带有采样噪声。

## 3. 可用的基座

| 基座 | 权重 | 现状 |
|---|---|---|
| VoV-99（当前） | `dd3d_det_final.pth` | 已在用，DP 的 BEV 骨干只支持它 |
| Depth Anything ViT-L | `da_vitl16.pth` | 权重已有；`HydraBackbone` 支持 `vit`，但 DP 的 `HydraBackboneBEV` 尚未接入 |
| V2-99 | 需单独下载 | `vov.py` 中有 V2-99 的加载分支，未验证 |
| EVA-ViT-L | 需下载 | `HydraBackbone` 支持 `sptr` |

**结论**：最现实的“换基座”是 **DA-ViT-L**（权重已就位），其次是 V2-99。

## 4. 候选创新点

| 候选 | 新颖性 | 成本 | 风险 |
|---|---|---|---|
| **A. 确定性推断 / 中心样本** | 中高：没人系统研究生成式规划器的单样本方差 | 极低（只改推理） | 若分布多模态，均值可能落在模态之间；用 medoid 规避 |
| B. 中心一致性训练 | 高：训练时约束“中心轨迹”的质量 | 中（5–18 epoch） | 需要 A 先验证 |
| C. 换 ViT-L 骨干 | 低（工程） | 中 | 需改 BEV 骨干 |
| D. MeanFlow 一步生成 | 中（MeanFuser 已做） | 高 | 与已有工作重叠 |

## 5. 建议

**主线 = A + B**：先证明“单样本推断有显著方差、中心推断更好”，再把“中心一致性”写进训练目标。
**底线 = 换 ViT-L 骨干（C）**：如果 A/B 的增益不够，用更强的感知基座把分数抬上去。

## 6. 参考文献（arXiv / 会议）

- TransFuser: arXiv 2205.15997
- PARA-Drive: CVPR 2024
- UniAD: CVPR 2023, arXiv 2212.10156
- DiffusionDrive: CVPR 2025, arXiv 2411.15139
- GoalFlow: CVPR 2025, arXiv 2503.05689
- MeanFuser: arXiv 2602.20060
- Adaptive Time Step FM: arXiv 2602.10285
- CATG: arXiv 2510.26292
- GuideFlow: arXiv 2511.18729
- WAM-Flow: arXiv 2512.06112
- FlowR2A: arXiv 2606.24231
- World4Scorer: arXiv 2609.36438
- WAC: arXiv 2610.22825
- PlanWAM: arXiv 2610.11382
- MORE: arXiv 2610.14871
- GTRS: arXiv 2506.06664
- NAVSIM: arXiv 2406.15349
