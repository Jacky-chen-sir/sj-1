# OPD 事后分析流水线

论文第 4 章里四个"零训练成本"的分析（效率表 / 师生排序一致性 / 挑战子集分解 / bootstrap 置信区间）。
**都不需要再训模型**：要么只跑前向，要么只读已有 pickle，要么只读两张 CSV。

## 总览

```
                        ┌─ run_efficiency_profile.py   → efficiency.csv          (#7 效率表)
训练好的 ckpt ──────────┤
                        └─ run_teacher_opd_cache.py    → {token}.pkl 目录         (#8 的输入)
                                                          │
                                                          └─ rank_agreement.py   → rank_agreement.csv (#8 排序一致性)

metric cache ──── run_scene_attributes.py ──→ scene_attributes.csv ─┐
                                                                    ├─ challenge_subsets.py (#9 挑战子集)
预测 pickle ──── 打分脚本 ──→ per_token.csv ────────────────────────┤
                                                                    └─ bootstrap_ci.py      (#10 置信区间)
```

`per_token.csv` 是新增的产物：打分脚本 `run_pdm_score_gpu_v2_aug.py` 现在除了写最终 CSV，
还会把**逐 token 的最终分数 + 场景元信息**（`log_name` / `frame_type` / 起止点）左连接后落盘。
最终 CSV 在聚合时把元信息丢掉了，而 `raw_pdm_score.pkl` 里还没有两帧扩展舒适度，
所以此前没有任何一张表能同时给出"最终 score"和"log_name"——#9 分桶和 #10 的 log 级
cluster bootstrap 都需要它。

---

## 一键跑完（推荐）

下面各节是分步说明；**跑的时候用总驱动**，它把 表 6（效率）与 表 7（分桶 + 显著性）串成一条链，
省得漏掉步骤或让两个方法的 `POLICY` 取到不同的值：

```bash
BASE_PKL=$ABL/evals/opd/base_13k.pkl OURS_PKL=$ABL/evals/opd/opd_13k.pkl \
  bash scripts/opd/analysis/run_zero_cost_analysis.sh
```

给了 `TEACHER_CKPT` / `STUDENT_CKPT` 就顺带跑效率表；`DRY_RUN=1` 只打印命令。
产物：`efficiency.csv`（表 6）、`challenge_subsets.csv` + `bootstrap_ci.csv`（表 7）。

---

## 0. 前置：环境变量

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/sj-1
export NAVSIM_EXP_ROOT=/path/to/exp
export OPENSCENE_DATA_ROOT=/path/to/dataset
export ABL=$NAVSIM_EXP_ROOT/ablation        # 本仓库约定的消融根目录
cd $NAVSIM_DEVKIT_ROOT
```

---

## 1. 逐 token 打分表（#9 / #10 的输入）

对每个要比较的方法各跑一次。`SKIP_INFER=1` 表示只对已存好的预测 pickle 打分、不跑模型，
所以这一步是纯 CPU、可并行的。

```bash
PKL=$ABL/evals/opd/step13000_subscores.pkl  TAG=opd_13k  POLICY=non_reactive \
  bash scripts/opd/evaluation/rescore_pickle.sh
```

产物在 `$NAVSIM_EXP_ROOT/rescore/<TAG>-<POLICY>/`：
- `<timestamp>.csv`   —— 官方口径的最终表（含 4 行汇总）
- `per_token.csv`     —— **逐 token + 元信息**，下面几步用它
- `raw_pdm_score.pkl` —— 逐 token 聚合前原始行

> 注意：`POLICY` 必须两个方法取**同一个值**，否则比较的是打分口径而不是模型。这一列在
> 论文里要写清楚（见 `scripts/opd/evaluation/rescore_pickle.sh` 顶部的说明）。

---

## 2. 场景属性表（#9 / #10 的分桶维度）

只用 metric cache，不需要模型和 GPU，跑一次就够，之后所有分桶分析复用。

```bash
python navsim/planning/script/run_scene_attributes.py \
  train_test_split=navtest \
  ++attrs.out=$ABL/analysis/scene_attributes.csv \
  ++attrs.workers=16
```

属性含义（定义都写在脚本 docstring 里）：`is_intersection`（route 含 lane connector）、
`has_vru`、`is_lane_change`（人类轨迹横向偏移 > 3.5 m）、`n_objects`、`ego_speed`。

---

## 3. #7 效率表

参数量 / FLOPs / 每样本延迟 / FPS。教师与学生用**同一个脚本、同一张卡、同一条
`agent.forward` 路径**，口径天然一致；`batch_size` 由 `dataloader.params.batch_size` 控制，
延迟按每样本折算。同一个 `profile.out` 追加写入，多次调用拼成一张表。

```bash
# 学生 ResNet34
python navsim/planning/script/run_efficiency_profile.py \
  agent=gtrs_aug_opd_r34 agent.checkpoint_path=<R34 ckpt> \
  agent.config.inference.model=student \
  ++profile.tag=student_r34 ++profile.out=$ABL/analysis/efficiency.csv

# 教师 ViT-L
python navsim/planning/script/run_efficiency_profile.py \
  agent=gtrs_aug_drivesuprim_vit agent.checkpoint_path=<ViT-L ckpt> \
  agent.config.inference.model=teacher \
  ++profile.tag=teacher_vitl ++profile.out=$ABL/analysis/efficiency.csv
```

> FLOPs 用 `torch.utils.flop_counter`（torch ≥ 2.1），只覆盖 conv/matmul 等主要算子，
> 不含 softmax 等逐元素开销。论文里要写明这个口径，且师生两侧用的是同一份代码。
> 想改 batch：`dataloader.params.batch_size=8`。

---

## 4. #8 师生排序一致性

蒸馏有没有起作用，取决于学生是否学到了教师的**排序**，而不只是"选了哪条"。这一步比较
同一批场景、同一个 8192 条词表上，教师与学生的粗筛分数。

**输入是 per-token pickle 目录**（`run_teacher_opd_cache.py` 的产物），所以不需要模型、
不需要 GPU，在任意机器上都能重跑。教师和学生用**同一个脚本**产出，格式一致
（`coarse` [V] fp32 + `topk_idx` [K] int32）：

```bash
# 教师缓存（全量 navtrain，见 scripts/opd/training/cache_teacher.sh）
agent=gtrs_aug_drivesuprim_vit agent.config.inference.model=teacher \
  bash scripts/opd/training/cache_teacher.sh

# 学生缓存：同一条命令，换成学生的 agent / ckpt，输出目录换一个
```

然后把学生缓存目录和教师目录一起喂给分析脚本：

```bash
python scripts/opd/analysis/rank_agreement.py \
  --teacher $ABL/opd_cache/teacher_navtest \
  --student base_5k=$ABL/opd_cache/base_step5000 \
  --student opd_5k=$ABL/opd_cache/opd_step5000 \
  --student opd_13k=$ABL/opd_cache/opd_step13000 \
  --out $ABL/analysis/rank_agreement.csv \
  --max-scenes 3000 --workers 8
```

输出逐场景明细 + `<stem>_summary.csv` 汇总，指标含义：

| 指标 | 读法 |
|---|---|
| `spearman` / `kendall` | 全词表排序相关性——分布形状学没学到 |
| `recall@K` | 学生 top-K 与教师 top-K 的交集比例——精排阶段的可达上界 |
| `top1_agree` | 最终选出的轨迹是否相同——端到端行为一致性 |
| `teacher_top1_in_student@K` | 教师的第一名落在学生前 K 名里——比集合召回更贴近实际 |
| `teacher_entropy` / `student_entropy` | 教师分布有多尖，学生有没有学得过平/过尖 |

`--max-scenes 3000` 是等距采样（不是随机），保证可复现；`--workers` 按核数给。

---

## 5. #9 挑战子集分解

只报一个 navtest 总分会掩盖改进发生在哪里。这一步按场景属性分桶，回答"提升是不是集中在
困难场景"——正是审稿人会追问的。**纯 CPU，读两张 CSV**。

```bash
python scripts/opd/analysis/challenge_subsets.py \
  --attrs $ABL/analysis/scene_attributes.csv \
  --run base_13k=$ABL/rescore/base_13k-non_reactive/per_token.csv \
  --run opd_13k=$ABL/rescore/opd_13k-non_reactive/per_token.csv \
  --out $ABL/analysis/challenge_subsets.csv
```

分桶维度：路口/非路口、有/无 VRU、换道/非换道、车流密度三分位、自车速度三分位。
每个维度内部的桶互不重叠、并集近似全覆盖，可以直接横向比。终端会打印一张
"行=桶、列=方法"的 EPDMS 透视表；两个 `--run` 时额外打印按提升排序的 Δ 表。

---

## 6. #10 Bootstrap 置信区间

单种子 + 单点数字在顶会上说服力不够。这一步给出 ΔEPDMS 的 95% CI、获胜场景比例与双侧 p 值。

```bash
python scripts/opd/analysis/bootstrap_ci.py \
  --baseline $ABL/rescore/base_13k-non_reactive/per_token.csv \
  --ours     $ABL/rescore/opd_13k-non_reactive/per_token.csv \
  --attrs    $ABL/analysis/scene_attributes.csv \
  --n-boot 10000 --out $ABL/analysis/bootstrap_ci.csv
```

**默认按 `log_name` 做 cluster bootstrap，而不是按场景独立重采样。** navtest 的场景来自
有限条 log，同一条 log 内相邻帧高度相关（同一段路、同一批交通参与者），按场景独立重采样
会把有效样本量高估好几倍、把置信区间压得过窄。cluster bootstrap 是这里统计上正确的做法，
也是审稿人会追问的点。`--cluster none` 可以退回按场景重采样做对照——**论文里两个数都报**，
并说明为什么以 cluster 的为准。

输出列：`delta` / `ci_lo` / `ci_hi` / `boot_p` / `sign_test_p` / `win_rate` / `tie_rate`，
以及 8 项子指标各自的一行（`metric` 列）和（给了 `--attrs` 时）各分桶的一行。

> bootstrap 衡量的是**测试集采样**带来的不确定性，**不包含训练随机性**。多种子方差要
> 分别跑每个种子的 `per_token.csv`，再对 Δ 的种子间标准差单独报告——两者不要混为一谈。

---

## 7. 决策过程 PCA 可视化（第 5 章 5.3 节）

`decision_pca.py` 对应论文 5.3 节的四个小节，用子命令分开跑。**四个子命令都不需要训练**：
`vocab` / `generation` 只读 npy，`scoring` 只读已有 pickle，`fullchain` 只读一个场景的缓存。

```bash
# 5.3.2 词表空间结构：8192 条锚点是否按驾驶意图聚成簇
python scripts/opd/analysis/decision_pca.py vocab \
    --vocab $NAVSIM_DEVKIT_ROOT/traj_final/test_8192_kmeans.npy \
    --out figs/fig5_2_vocab_pca.png --csv figs/fig5_2_vocab_pca.csv --plot

# 5.3.3 生成侧：FM 提案是否覆盖词表簇之间的空隙（投影基只在词表上拟合，其余复用同一组基）
python scripts/opd/analysis/decision_pca.py generation \
    --vocab $NAVSIM_DEVKIT_ROOT/traj_final/test_8192_kmeans.npy \
    --fm /exp/fm_proposals.npy --ddpm /exp/ddpm_proposals.npy --gt /exp/gt_traj.npy \
    --out figs/fig5_3_generation_pca.png --plot

# 5.3.4 评价侧：师生粗筛分布是否靠拢（Top-1 一致率 / Spearman / Recall@K）
python scripts/opd/analysis/decision_pca.py scoring \
    --teacher /exp/opd_cache/teacher_navtest \
    --student base_13k=/exp/opd_cache/base_step13000 \
    --student opd_13k=/exp/opd_cache/opd_step13000 \
    --out figs/fig5_4_scoring_pca.png --csv figs/fig5_4_scoring.csv --plot

# 5.3.5 全链路：单场景的「候选云 → Top-K → argmax 输出」
python scripts/opd/analysis/decision_pca.py fullchain \
    --vocab $NAVSIM_DEVKIT_ROOT/traj_final/test_8192_kmeans.npy \
    --cache /exp/opd_cache/opd_step13000 --scene <token> \
    --out figs/fig5_5_fullchain_pca.png --plot
```

**为什么用 PCA 而不是 t-SNE**：PCA 是线性的，投影后的坐标有明确的方差解释含义，且可以用
**同一组基**去投影不同来源的样本——这正是 5.3.3 与 5.3.5 要求「词表与提案在同一坐标系」的前提。
t-SNE 的坐标不可解释、也不能对样本外数据直接投影，只适合作定性对照。

**注意**：`vocab` / `generation` 两个子命令的输入 npy 支持 `[V,H,3]`（绝对位姿）与
`[V,H,4]`（增量编码 `Δx, Δy, sinθ, cosθ`）两种格式，后者会自动 cumsum 还原。

---

## 依赖

`numpy`、`pandas`（必需）；`scipy`（可选，`rank_agreement.py` 的 Kendall τ 全量计算用它，
没有就退化到等距采样）。打分脚本和 `run_scene_attributes.py` 需要完整的 navsim 环境。
