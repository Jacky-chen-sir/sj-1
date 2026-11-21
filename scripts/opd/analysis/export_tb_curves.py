#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export key TensorBoard scalars to CSV + publication PNGs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


RUNS_DEFAULT = {
    'v2_p1_base': 'ch3_df_v2/p1_base/2026.11.10.20.47.31/lightning_logs/version_0',
    'v2_p1_center': 'ch3_df_v2/p1_center/2026.11.13.05.59.35/lightning_logs/version_0',
    'v1_p1_base': 'ch3_df/p1_base/2026.10.23.19.54.55/lightning_logs/version_0',
    'v1_p1_center': 'ch3_df/p1_center/2026.10.26.05.13.51/lightning_logs/version_0',
    'v1_p2_center18': 'ch3_df/p2_center18/2026.10.29.13.52.58/lightning_logs/version_0',
}

# Okabe–Ito
COLORS = {
    'v2_p1_base': '#0072B2',
    'v2_p1_center': '#D55E00',
    'v1_p1_base': '#56B4E9',
    'v1_p1_center': '#E69F00',
    'v1_p2_center18': '#009E73',
}

KEY_TAGS = [
    'train/loss_step',
    'val/loss_epoch',
    'val/fm_x1_ade_epoch',
    'val/fm_x1_fde_epoch',
    'val/fm_zero_ade_epoch',
    'val/fm_zero_fde_epoch',
    'train/fm_center_ade_epoch',
    'train/fm_center_fde_epoch',
    'train/fm_center_loss_epoch',
    'train/fm_center_ade_step',
    'train/fm_center_loss_step',
    'train/fm_x1_ade_step',
    'train/dp_loss_step',
]


def style():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.8,
    })
    return plt


def load_run(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(path), size_guidance={'scalars': 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get('scalars', []):
        ev = ea.Scalars(tag)
        steps = np.array([e.step for e in ev], dtype=np.int64)
        vals = np.array([e.value for e in ev], dtype=np.float64)
        out[tag] = (steps, vals)
    return out


def smooth(y: np.ndarray, w: int = 51) -> np.ndarray:
    if len(y) < w or w < 3:
        return y
    if w % 2 == 0:
        w += 1
    k = np.ones(w) / w
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode='edge')
    return np.convolve(yp, k, mode='valid')


def export_csv(runs: Dict[str, Dict], out_dir: Path):
    import csv
    for name, data in runs.items():
        for tag, (steps, vals) in data.items():
            if tag not in KEY_TAGS and not tag.endswith('_epoch'):
                continue
            safe = tag.replace('/', '__')
            p = out_dir / 'csv' / f'{name}__{safe}.csv'
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['step', 'value'])
                w.writerows(zip(steps.tolist(), vals.tolist()))


def plot_multi(plt, runs, tag: str, out: Path, title: str, ylabel: str,
               names: List[str], smooth_w: int = 1, xlim=None):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for name in names:
        if name not in runs or tag not in runs[name]:
            continue
        steps, vals = runs[name][tag]
        y = smooth(vals, smooth_w) if smooth_w > 1 else vals
        ax.plot(steps, y, color=COLORS.get(name, None), label=name, alpha=0.95)
    ax.set_xlabel('step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend(loc='best')
    ax.grid(alpha=0.25, linewidth=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp-root', default='/home/ws/navsim_workspace/exp')
    ap.add_argument('--out-dir', default='/home/ws/navsim_workspace/exp/ch5_figs/tb_curves')
    args = ap.parse_args()

    plt = style()
    exp = Path(args.exp_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for name, rel in RUNS_DEFAULT.items():
        p = exp / rel
        if not p.exists():
            print('missing', name, p)
            continue
        print('load', name)
        runs[name] = load_run(p)

    export_csv(runs, out_dir)

    # --- publication panels ---
    v2 = ['v2_p1_base', 'v2_p1_center']
    v1 = ['v1_p1_base', 'v1_p1_center', 'v1_p2_center18']

    plot_multi(plt, runs, 'train/loss_step', out_dir / 'fig_tb_v2_train_loss.png',
               'Phase1-v2 train loss', 'loss', v2, smooth_w=101)
    plot_multi(plt, runs, 'val/loss_epoch', out_dir / 'fig_tb_v2_val_loss.png',
               'Phase1-v2 val loss (epoch)', 'loss', v2)
    plot_multi(plt, runs, 'val/fm_x1_ade_epoch', out_dir / 'fig_tb_v2_val_ade.png',
               'Phase1-v2 val FM ADE', 'ADE (m)', v2)
    plot_multi(plt, runs, 'val/fm_x1_fde_epoch', out_dir / 'fig_tb_v2_val_fde.png',
               'Phase1-v2 val FM FDE', 'FDE (m)', v2)
    plot_multi(plt, runs, 'val/fm_zero_ade_epoch', out_dir / 'fig_tb_v2_val_zero_ade.png',
               'Phase1-v2 val FM-zero ADE', 'ADE (m)', v2)
    plot_multi(plt, runs, 'val/fm_center_ade_epoch', out_dir / 'fig_tb_v2_val_center_ade.png',
               'Phase1-v2 val center ADE', 'ADE (m)', ['v2_p1_center'])
    plot_multi(plt, runs, 'train/fm_center_ade_step', out_dir / 'fig_tb_v2_train_center_ade.png',
               'Phase1-v2 train center ADE', 'ADE (m)', ['v2_p1_center'], smooth_w=101)

    plot_multi(plt, runs, 'train/loss_step', out_dir / 'fig_tb_v1_train_loss.png',
               'Phase1/2 train loss', 'loss', v1, smooth_w=101)
    plot_multi(plt, runs, 'val/loss_epoch', out_dir / 'fig_tb_v1_val_loss.png',
               'Phase1/2 val loss (epoch)', 'loss', v1)
    plot_multi(plt, runs, 'val/fm_center_ade_epoch', out_dir / 'fig_tb_v1_val_center_ade.png',
               'Phase1/2 val center ADE', 'ADE (m)',
               ['v1_p1_center', 'v1_p2_center18'])
    plot_multi(plt, runs, 'val/fm_x1_ade_epoch', out_dir / 'fig_tb_v1_val_ade.png',
               'Phase1/2 val FM ADE', 'ADE (m)', v1)

    # combined overview 2x2 for thesis
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    panels = [
        (axes[0, 0], 'train/loss_step', v2, 'v2 train loss', True),
        (axes[0, 1], 'val/fm_x1_ade_epoch', v2, 'v2 val ADE', False),
        (axes[1, 0], 'train/loss_step', v1, 'v1 train loss', True),
        (axes[1, 1], 'val/fm_center_ade_epoch',
         ['v1_p1_center', 'v1_p2_center18', 'v2_p1_center'], 'center ADE (val)', False),
    ]
    for ax, tag, names, title, do_smooth in panels:
        for name in names:
            if name not in runs or tag not in runs[name]:
                continue
            steps, vals = runs[name][tag]
            y = smooth(vals, 101) if do_smooth else vals
            ax.plot(steps, y, color=COLORS.get(name), label=name)
        ax.set_title(title)
        ax.set_xlabel('step')
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_tb_overview.png')
    fig.savefig(out_dir / 'fig_tb_overview.pdf')
    plt.close(fig)
    print('wrote', out_dir / 'fig_tb_overview.png')

    # also copy overview into ch5_figs root
    import shutil
    shutil.copy2(out_dir / 'fig_tb_overview.png', out_dir.parent / 'fig5_tb_overview.png')
    shutil.copy2(out_dir / 'fig_tb_overview.pdf', out_dir.parent / 'fig5_tb_overview.pdf')
    print('DONE →', out_dir)


if __name__ == '__main__':
    main()
