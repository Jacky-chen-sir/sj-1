#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成论文图 5：粗筛召回率与最终 EPDMS 的关系。

## 这张图要回答什么

第 4 章 4.5 节的用途是**诊断**，不是报成绩：它检验蒸馏是否真的传递了教师的**判别结构**，
而不是碰巧把指标刷高了。判据是两条：

  1. 若"教师最优候选落入学生前 K 名"的比例（下称召回率）与 EPDMS 呈**单调关系**，
     说明评价器的性能确实由它在头部候选上的判别质量决定；
  2. 若本文方法在**同等召回率**下取得更高 EPDMS（点落在拟合线上方），
     说明增益不只来自"把教师的 Top-1 抄过来"，而包含对整池候选的排序改善。

若两条都成立，则分解式蒸馏传递的是结构而非答案。

## 输入

  --rank   `rank_agreement.py` 产出的 `*_summary.csv`
           需要 `run` 列与 `teacher_top1_in_student@{K}_mean` 列
  --scores 每个 run 的 EPDMS，CSV 需要 `run`（或 `method`）与 `epdms` 两列。
           也可以直接用 `scripts/opd/analysis/*.py` 那套 `scores.csv` 格式
           （列名 `method,step,epdms`，脚本会用 `method` 当 run 名）。

## 输出

  PNG（散点 + 拟合线 + Spearman/Kendall 标注）与同名 CSV（逐点坐标）。

用法：

    python scripts/opd/analysis/plot_recall_vs_epdms.py \\
        --rank   /exp/analysis/rank_agreement_summary.csv \\
        --scores /exp/analysis/scores.csv \\
        --ks     1 32 256 \\
        --out    /exp/analysis/fig5_recall_vs_epdms.png
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def pick(row: Dict[str, str], names) -> Optional[str]:
    for n in names:
        if n in row and str(row[n]).strip() != '':
            return str(row[n]).strip()
    return None


def spearman(xs: List[float], ys: List[float]) -> float:
    """秩相关。不依赖 scipy。"""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return float('nan')
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float('nan')


def kendall(xs: List[float], ys: List[float]) -> float:
    """Kendall τ-b，含结的修正。"""
    n = len(xs)
    if n < 3:
        return float('nan')
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else float('nan')


def ols(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0, my
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return k, my - k * mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', required=True, help='rank_agreement 的 *_summary.csv')
    ap.add_argument('--scores', required=True, help='含 run/method 与 epdms 两列的 CSV')
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 32, 256])
    ap.add_argument('--out', required=True)
    ap.add_argument('--csv', default=None, help='逐点坐标输出（默认与 --out 同名）')
    ap.add_argument('--title', default='粗筛召回率与最终 EPDMS 的关系')
    args = ap.parse_args()

    rank_rows = read_csv(args.rank)
    score_rows = read_csv(args.scores)

    epdms: Dict[str, float] = {}
    for r in score_rows:
        name = pick(r, ['run', 'method', 'student'])
        val = pick(r, ['epdms', 'EPDMS', 'epdms_mean'])
        if name and val:
            epdms[name] = float(val)

    pts = []          # (k, run, recall, epdms)
    for r in rank_rows:
        run = pick(r, ['run', 'method', 'student'])
        if not run or run not in epdms:
            continue
        for k in args.ks:
            v = pick(r, [f'teacher_top1_in_student@{k}_mean'])
            if v is None:
                continue
            pts.append((k, run, float(v), epdms[run]))

    if not pts:
        print('[ERROR] 没有可用的点。检查 --rank 里是否有 teacher_top1_in_student@K_mean，'
              '以及 --scores 的 run 名是否与之一致。', file=sys.stderr)
        print('  rank 里的 run:', sorted({pick(r, ["run"]) for r in rank_rows}))
        print('  scores 里的 run:', sorted(epdms)[:10])
        raise SystemExit(2)

    xs = [p[2] for p in pts]
    ys = [p[3] for p in pts]
    rho, tau = spearman(xs, ys), kendall(xs, ys)
    k_, b_ = ols(xs, ys)

    print(f'点数 {len(pts)}  Spearman ρ={rho:.4f}  Kendall τ={tau:.4f}  '
          f'拟合 y={k_:.2f}x+{b_:.2f}')

    # 逐点坐标落盘，便于复核
    out_csv = args.csv or str(Path(args.out).with_suffix('.csv'))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['K', 'run', 'teacher_top1_in_student@K', 'epdms'])
        w.writerows(pts)
    print('已写', out_csv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'PingFang SC',
                                       'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(len(args.ks) - 1, 1)) for i in range(len(args.ks))]

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for ci, k in enumerate(args.ks):
        sub = [(p[2], p[3], p[1]) for p in pts if p[0] == k]
        if not sub:
            continue
        ax.scatter([s[0] for s in sub], [s[1] for s in sub],
                   s=52, color=colors[ci], alpha=0.85, linewidths=0.6,
                   edgecolors='white', label=f'K = {k}')
    lx = [min(xs), max(xs)]
    ax.plot(lx, [k_ * x + b_ for x in lx], '--', color='#9E4A3E', lw=1.6,
            label=f'线性拟合 (ρ={rho:.2f})')
    ax.set_xlabel('教师最优候选落入学生 Top-$K$ 的比例')
    ax.set_ylabel('EPDMS')
    ax.set_title(args.title)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print('已写', args.out)
    print(f'\n论文里可报：Spearman ρ = {rho:.3f}，Kendall τ = {tau:.3f}（n = {len(pts)}）')


if __name__ == '__main__':
    main()
