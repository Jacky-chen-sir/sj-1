#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 5 章非 PCA 图表：中心一致性对照、K 步扫描、方差解释率。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from pathlib import Path
    for p in (
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',
    ):
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass
    plt.rcParams['font.sans-serif'] = [
        'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC', 'DejaVu Sans',
    ]
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def plot_center(summary_csv: str, out: str) -> None:
    rows = list(csv.DictReader(open(summary_csv)))
    # 取每臂在 K=20 上的最好 / 各 epoch
    by: Dict[Tuple[str, str], float] = {}
    for r in rows:
        if str(r.get('K')) != '20':
            continue
        by[(r['arm'], r['epoch'])] = float(r['score'])

    arms = ['base', 'center']
    epochs = sorted({e for a, e in by if a in arms})
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(epochs))
    w = 0.36
    for i, arm in enumerate(arms):
        ys = [by.get((arm, e), np.nan) for e in epochs]
        ax.bar(x + (i - 0.5) * w, ys, width=w, label=arm)
        for xi, yi in zip(x + (i - 0.5) * w, ys):
            if np.isfinite(yi):
                ax.text(xi, yi + 0.008, f'{yi:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.set_ylabel('navtest score (zero-noise, K=20)')
    ax.set_title('中心一致性训练对照（Phase1-v2）')
    ax.legend()
    ax.set_ylim(0.5, 0.85)
    ax.grid(axis='y', alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('已写', out)


def plot_phase0(summary_csv: str, out: str) -> None:
    # phase0 csv may differ; also accept hard-coded known numbers as fallback overlay
    rows = []
    p = Path(summary_csv)
    if p.exists():
        rows = list(csv.DictReader(open(p)))
    labels, scores = [], []
    for r in rows:
        # try common schemas
        lab = r.get('tag') or r.get('arm') or r.get('method') or r.get('name')
        sc = r.get('score') or r.get('pdm_score')
        if lab and sc:
            labels.append(str(lab))
            scores.append(float(sc))
    if not labels:
        # 论文已定稿的 Phase0 数字
        labels = ['DDPM first', 'FM first', 'FM medoid', 'FM zero']
        scores = [0.7317, 0.7418, 0.7582, 0.7657]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colors = ['#9E4A3E', '#5B8DC9', '#5FA173', '#37744B']
    bars = ax.bar(labels, scores, color=colors[:len(labels)])
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.003, f'{s:.4f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('navtest score')
    ax.set_title('推断方式对照（Phase 0，零训练）')
    ax.set_ylim(0.70, 0.80)
    ax.grid(axis='y', alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('已写', out)


def plot_kscan(summary_csv: str, out: str, arm: str = 'center') -> None:
    rows = [r for r in csv.DictReader(open(summary_csv)) if r['arm'] == arm]
    # 每个 K 取该臂所有 epoch 的最大分（展示可达前沿）
    best: Dict[int, float] = {}
    for r in rows:
        k = int(r['K'])
        s = float(r['score'])
        best[k] = max(best.get(k, -1.0), s)
    if not best:
        raise SystemExit(f'{summary_csv} 里没有 arm={arm} 的行')
    ks = sorted(best)
    ys = [best[k] for k in ks]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(ks, ys, marker='o', lw=2, color='#37744B')
    for k, y in zip(ks, ys):
        ax.text(k, y + 0.008, f'{y:.3f}', ha='center', fontsize=8)
    ax.set_xlabel('FM inference steps K')
    ax.set_ylabel('navtest score (zero-noise, best epoch)')
    ax.set_title(f'K 步扫描（{arm}）')
    ax.grid(alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('已写', out)


def plot_var_curve(vocab: str, out: str, max_k: int = 20) -> None:
    from decision_pca import PCA, to_absolute, load_traj_file
    traj = to_absolute(load_traj_file(vocab))
    X = traj.reshape(traj.shape[0], -1)
    pca = PCA(min(max_k, X.shape[1])).fit(X)
    cum = pca.cum_var()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.arange(1, len(cum) + 1), cum, marker='o', color='#5B8DC9')
    ax.axhline(0.9, ls='--', color='#9E4A3E', lw=1, label='90%')
    ax.set_xlabel('主成分个数')
    ax.set_ylabel('累计方差解释率')
    ax.set_title('词表 PCA 方差解释率曲线')
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('已写', out, 'cum[:5]=', ' '.join(f'{c:.3f}' for c in cum[:5]))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('center')
    p.add_argument('--summary', required=True)
    p.add_argument('--out', required=True)

    p = sub.add_parser('phase0')
    p.add_argument('--summary', default='')
    p.add_argument('--out', required=True)

    p = sub.add_parser('kscan')
    p.add_argument('--summary', required=True)
    p.add_argument('--arm', default='center')
    p.add_argument('--out', required=True)

    p = sub.add_parser('varcurve')
    p.add_argument('--vocab', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-k', type=int, default=20)

    args = ap.parse_args()
    if args.cmd == 'center':
        plot_center(args.summary, args.out)
    elif args.cmd == 'phase0':
        plot_phase0(args.summary, args.out)
    elif args.cmd == 'kscan':
        plot_kscan(args.summary, args.out, args.arm)
    elif args.cmd == 'varcurve':
        plot_var_curve(args.vocab, args.out, args.max_k)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
