# Ch5 锁档实验数据（论文用）

本目录入库第五章定稿所依据的**可复核数字与图**，不含 ckpt / metric cache / 逐场景原始评测大表。

## 目录

| 路径 | 内容 |
|------|------|
| `summaries/` | 生成器 Phase0/1/2、Phase1-v2、p1_official（官方 5ep，可能仍空）总分汇总 |
| `efficiency/` | 表 5-2：GPU0 师生参数量 / FLOPs / 延迟 p50 / 峰值显存 |
| `evals_summary/` | 评价器同迭代 `same_iter_scores.csv`；门控与同迭代紧凑汇总 `locked_eval_summaries.csv` |
| `results_pub/` | 论文用柱图 / Pareto / PCA / TB / 效率图（PNG+PDF） |
| `tb_csv/` | Phase1-v2 等训练曲线 epoch 宽表 |
| `decision_viz/` | BEV\|PV 决策可视化（hero + plate） |

## 口径提示

- 生成器分数多为 0–1 小数（×100 即论文百分制）。
- `locked_eval_summaries.csv` 中 `epdms` 等已换算为百分制；`zero_pct` 为场景零分占比。
- 效率：`agent.forward`、batch=1、RTX 3090 GPU0；延迟取 p50。

复现脚本见 `scripts/opd/analysis/polish_ch5_figs_thesis.py`、`fill_thesis_ch5.py`、`run_efficiency_profile.py`。
