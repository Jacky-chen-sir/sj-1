#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export ALL useful TensorBoard scalars for thesis Ch3 (FM-DP) + Ch4 (OPD/GTRS).

Writes:
  - per-run CSV for every scalar tag
  - epoch-only wide CSV per run
  - publication PNG/PDF panels
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

EXP = Path('/home/ws/navsim_workspace/exp')
OUT = Path('/home/ws/navsim_workspace/exp/ch5_figs/tb_curves')

# Okabe–Ito
C = {
    'v2_p1_base': '#0072B2',
    'v2_p1_center': '#D55E00',
    'v1_p1_base': '#56B4E9',
    'v1_p1_center': '#E69F00',
    'v1_p2_center18': '#009E73',
    'ch3_guidance': '#CC79A7',
    'ch3_supp_control': '#000000',
    'ch4_opd_9cd57b3': '#D55E00',
    'ch4_opd_prev': '#E69F00',
    'ch4_drivesuprim': '#0072B2',
    'ch4_gtrs_r34': '#009E73',
    'ch4_ds_r34': '#56B4E9',
}

RUNS: Dict[str, Path] = {
    # --- Ch3 FM-DP ---
    'v2_p1_base': EXP / 'ch3_df_v2/p1_base/2026.11.10.20.47.31/lightning_logs/version_0',
    'v2_p1_center': EXP / 'ch3_df_v2/p1_center/2026.11.13.05.59.35/lightning_logs/version_0',
    'v1_p1_base': EXP / 'ch3_df/p1_base/2026.10.23.19.54.55/lightning_logs/version_0',
    'v1_p1_center': EXP / 'ch3_df/p1_center/2026.10.26.05.13.51/lightning_logs/version_0',
    'v1_p2_center18': EXP / 'ch3_df/p2_center18/2026.10.29.13.52.58/lightning_logs/version_0',
    'ch3_guidance': EXP / 'train_dp_ch3_scorer_guidance/2026.10.14.20.02.57/lightning_logs/version_0',
    'ch3_supp_control': EXP / 'ch3_supplement/control/2026.10.18.19.42.46/lightning_logs/version_0',
    # --- Ch4 scorer / OPD ---
    'ch4_opd_9cd57b3': EXP / 'training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256-9cd57b3/2026.09.29.03.48.37/lightning_logs/version_0',
    'ch4_opd_prev': EXP / 'training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256/2026.09.14.11.14.19/lightning_logs/version_0',
    'ch4_drivesuprim': EXP / 'training/gtrs_aug_drivesuprim_vov/rot_30-p_0.5/stage_layers_3-topks_256/2026.09.11.19.28.36/lightning_logs/version_0',
    'ch4_gtrs_r34': EXP / 'ablation_same_iter_r34/official_gtrs_r34/2026.09.20.09.18.36/lightning_logs/version_0',
    'ch4_ds_r34': EXP / 'ablation_same_iter_r34/official_drivesuprim_r34/2026.09.24.23.20.43/lightning_logs/version_0',
}


def style():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 140,
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
        out[tag] = (
            np.array([e.step for e in ev], dtype=np.int64),
            np.array([e.value for e in ev], dtype=np.float64),
        )
    return out


def smooth(y: np.ndarray, w: int = 51) -> np.ndarray:
    if len(y) < w or w < 3:
        return y
    if w % 2 == 0:
        w += 1
    k = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(y, (pad, pad), mode='edge'), k, mode='valid')


def dump_all_csv(name: str, data: Dict, out_dir: Path):
    csv_dir = out_dir / 'csv' / name
    csv_dir.mkdir(parents=True, exist_ok=True)
    # per-tag
    for tag, (steps, vals) in data.items():
        safe = tag.replace('/', '__').replace(' ', '_')
        with open(csv_dir / f'{safe}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'value'])
            w.writerows(zip(steps.tolist(), vals.tolist()))
    # wide epoch table
    epoch_tags = sorted([t for t in data if t.endswith('_epoch') or t == 'epoch'])
    if not epoch_tags:
        return
    # align by step
    all_steps = sorted(set().union(*[set(data[t][0].tolist()) for t in epoch_tags]))
    maps = {t: dict(zip(data[t][0].tolist(), data[t][1].tolist())) for t in epoch_tags}
    with open(csv_dir / '_epoch_wide.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step'] + epoch_tags)
        for s in all_steps:
            w.writerow([s] + [maps[t].get(s, '') for t in epoch_tags])


def plot_multi(plt, runs, tag, out, title, ylabel, names, smooth_w=1):
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    n_plotted = 0
    for name in names:
        if name not in runs or tag not in runs[name]:
            continue
        steps, vals = runs[name][tag]
        y = smooth(vals, smooth_w) if smooth_w > 1 else vals
        ax.plot(steps, y, color=C.get(name), label=name, alpha=0.95)
        n_plotted += 1
    if n_plotted == 0:
        plt.close(fig)
        print('skip empty', out.name, tag)
        return
    ax.set_xlabel('step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc='best')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print('wrote', out)


def main():
    plt = style()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / 'csv').mkdir(exist_ok=True)

    runs = {}
    meta = {}
    for name, path in RUNS.items():
        if not path.exists():
            print('MISSING', name, path)
            continue
        print('load', name, '...')
        data = load_run(path)
        runs[name] = data
        dump_all_csv(name, data, out)
        meta[name] = {
            'path': str(path),
            'n_tags': len(data),
            'tags': sorted(data.keys()),
        }
        print(f'  tags={len(data)} csv → csv/{name}/')

    with open(out / 'runs_manifest.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # ========== Ch3 panels ==========
    ch3 = out / 'ch3'
    ch3.mkdir(exist_ok=True)
    v2 = ['v2_p1_base', 'v2_p1_center']
    v1 = ['v1_p1_base', 'v1_p1_center', 'v1_p2_center18']
    all_ch3 = v2 + v1 + ['ch3_guidance', 'ch3_supp_control']

    plot_multi(plt, runs, 'train/loss_step', ch3 / 'fig_ch3_train_loss.png',
               'Ch3 train loss', 'loss', all_ch3, smooth_w=101)
    plot_multi(plt, runs, 'val/loss_epoch', ch3 / 'fig_ch3_val_loss.png',
               'Ch3 val loss (epoch)', 'loss', all_ch3)
    plot_multi(plt, runs, 'val/fm_x1_ade_epoch', ch3 / 'fig_ch3_val_ade.png',
               'Ch3 val FM ADE', 'ADE (m)', v2 + v1)
    plot_multi(plt, runs, 'val/fm_x1_fde_epoch', ch3 / 'fig_ch3_val_fde.png',
               'Ch3 val FM FDE', 'FDE (m)', v2 + v1)
    plot_multi(plt, runs, 'val/fm_zero_ade_epoch', ch3 / 'fig_ch3_val_zero_ade.png',
               'Ch3 val FM-zero ADE', 'ADE (m)', v2)
    plot_multi(plt, runs, 'val/fm_zero_fde_epoch', ch3 / 'fig_ch3_val_zero_fde.png',
               'Ch3 val FM-zero FDE', 'FDE (m)', v2)
    plot_multi(plt, runs, 'val/fm_zero_jerk_epoch', ch3 / 'fig_ch3_val_zero_jerk.png',
               'Ch3 val FM-zero jerk', 'jerk', v2)
    plot_multi(plt, runs, 'val/fm_zero_accel_epoch', ch3 / 'fig_ch3_val_zero_accel.png',
               'Ch3 val FM-zero accel', 'accel', v2)
    plot_multi(plt, runs, 'train/fm_center_ade_epoch', ch3 / 'fig_ch3_train_center_ade.png',
               'Ch3 train center ADE (epoch)', 'ADE (m)',
               ['v2_p1_center', 'v1_p1_center', 'v1_p2_center18'])
    plot_multi(plt, runs, 'train/fm_center_loss_epoch', ch3 / 'fig_ch3_train_center_loss.png',
               'Ch3 train center loss (epoch)', 'loss',
               ['v2_p1_center', 'v1_p1_center', 'v1_p2_center18'])
    plot_multi(plt, runs, 'val/bev_semantic_loss_epoch', ch3 / 'fig_ch3_val_bev.png',
               'Ch3 val BEV semantic loss', 'loss', v2 + v1)
    plot_multi(plt, runs, 'train/dp_loss_step', ch3 / 'fig_ch3_train_dp_loss.png',
               'Ch3 train DP loss', 'loss', all_ch3, smooth_w=101)

    # Ch3 overview 2x2
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    panels = [
        (axes[0, 0], 'train/loss_step', v2, 'v2 train loss', True),
        (axes[0, 1], 'val/fm_zero_ade_epoch', v2, 'v2 val FM-zero ADE', False),
        (axes[1, 0], 'train/loss_step', v1, 'v1 train loss', True),
        (axes[1, 1], 'train/fm_center_ade_epoch',
         ['v1_p1_center', 'v1_p2_center18', 'v2_p1_center'], 'train center ADE', False),
    ]
    for ax, tag, names, title, do_s in panels:
        for name in names:
            if name not in runs or tag not in runs[name]:
                continue
            s, v = runs[name][tag]
            ax.plot(s, smooth(v, 101) if do_s else v, color=C.get(name), label=name)
        ax.set_title(title)
        ax.set_xlabel('step')
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ch3 / 'fig_ch3_overview.png')
    fig.savefig(ch3 / 'fig_ch3_overview.pdf')
    plt.close(fig)
    print('wrote', ch3 / 'fig_ch3_overview.png')

    # ========== Ch4 panels ==========
    ch4 = out / 'ch4'
    ch4.mkdir(exist_ok=True)
    opd = ['ch4_opd_9cd57b3', 'ch4_opd_prev']
    scorers = ['ch4_opd_9cd57b3', 'ch4_drivesuprim', 'ch4_gtrs_r34', 'ch4_ds_r34']

    plot_multi(plt, runs, 'train/loss_step', ch4 / 'fig_ch4_train_loss.png',
               'Ch4 train loss', 'loss', scorers, smooth_w=21)
    plot_multi(plt, runs, 'val/loss_epoch', ch4 / 'fig_ch4_val_loss.png',
               'Ch4 val loss (epoch)', 'loss', scorers)
    plot_multi(plt, runs, 'train/loss-ori_epoch', ch4 / 'fig_ch4_train_loss_ori.png',
               'Ch4 train loss-ori', 'loss', scorers)
    plot_multi(plt, runs, 'train/loss-aug_epoch', ch4 / 'fig_ch4_train_loss_aug.png',
               'Ch4 train loss-aug', 'loss', scorers)
    plot_multi(plt, runs, 'train/loss-soft_epoch', ch4 / 'fig_ch4_train_loss_soft.png',
               'Ch4 train loss-soft (teacher)', 'loss', scorers)
    plot_multi(plt, runs, 'val/imi_loss-ori_epoch', ch4 / 'fig_ch4_val_imi_ori.png',
               'Ch4 val imi-ori', 'loss', scorers)
    plot_multi(plt, runs, 'val/imi_loss-soft_epoch', ch4 / 'fig_ch4_val_imi_soft.png',
               'Ch4 val imi-soft', 'loss', scorers)
    plot_multi(plt, runs, 'train/loss_opd-opd_epoch', ch4 / 'fig_ch4_train_opd_distill.png',
               'Ch4 train OPD distill loss', 'loss', opd)
    plot_multi(plt, runs, 'val/loss_opd-opd_epoch', ch4 / 'fig_ch4_val_opd_distill.png',
               'Ch4 val OPD distill loss', 'loss', opd)
    plot_multi(plt, runs, 'train/distill_imi-opd_epoch', ch4 / 'fig_ch4_train_distill_imi.png',
               'Ch4 train distill imi', 'loss', opd)
    plot_multi(plt, runs, 'val/distill_imi-opd_epoch', ch4 / 'fig_ch4_val_distill_imi.png',
               'Ch4 val distill imi', 'loss', opd)
    plot_multi(plt, runs, 'train/distill_recall-opd_epoch', ch4 / 'fig_ch4_train_distill_recall.png',
               'Ch4 train distill recall', 'loss', opd)
    plot_multi(plt, runs, 'val/distill_recall-opd_epoch', ch4 / 'fig_ch4_val_distill_recall.png',
               'Ch4 val distill recall', 'loss', opd)
    plot_multi(plt, runs, 'train/distill_head-opd_epoch', ch4 / 'fig_ch4_train_distill_head.png',
               'Ch4 train distill head (sum)', 'loss', opd)
    plot_multi(plt, runs, 'val/distill_head-opd_epoch', ch4 / 'fig_ch4_val_distill_head.png',
               'Ch4 val distill head (sum)', 'loss', opd)
    plot_multi(plt, runs, 'val/loss-refinement_ori_epoch', ch4 / 'fig_ch4_val_refine_ori.png',
               'Ch4 val refinement-ori', 'loss', scorers)
    plot_multi(plt, runs, 'val/pdm_comfort_loss-ori_epoch', ch4 / 'fig_ch4_val_comfort.png',
               'Ch4 val comfort-ori', 'loss', scorers)

    # Ch4 overview
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    panels = [
        (axes[0, 0], 'train/loss_step', scorers, 'train loss', True),
        (axes[0, 1], 'val/loss_epoch', scorers, 'val loss', False),
        (axes[1, 0], 'val/imi_loss-ori_epoch', scorers, 'val imi-ori', False),
        (axes[1, 1], 'val/loss_opd-opd_epoch', opd, 'val OPD distill', False),
    ]
    for ax, tag, names, title, do_s in panels:
        for name in names:
            if name not in runs or tag not in runs[name]:
                continue
            s, v = runs[name][tag]
            ax.plot(s, smooth(v, 21) if do_s else v, color=C.get(name), label=name)
        ax.set_title(title)
        ax.set_xlabel('step')
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ch4 / 'fig_ch4_overview.png')
    fig.savefig(ch4 / 'fig_ch4_overview.pdf')
    plt.close(fig)
    print('wrote', ch4 / 'fig_ch4_overview.png')

    # copy overviews to ch5_figs root
    import shutil
    shutil.copy2(ch3 / 'fig_ch3_overview.png', OUT.parent / 'fig3_tb_overview.png')
    shutil.copy2(ch3 / 'fig_ch3_overview.pdf', OUT.parent / 'fig3_tb_overview.pdf')
    shutil.copy2(ch4 / 'fig_ch4_overview.png', OUT.parent / 'fig4_tb_overview.png')
    shutil.copy2(ch4 / 'fig_ch4_overview.pdf', OUT.parent / 'fig4_tb_overview.pdf')

    # index
    lines = ['# TensorBoard curve export (Ch3 + Ch4)', '',
             f'Runs: {len(runs)}', f'Output: `{out}`', '',
             '## Runs', '']
    for name, m in meta.items():
        lines.append(f'- **{name}**: {m["n_tags"]} tags → `csv/{name}/`')
    lines += ['', '## Figures', '',
              '- `ch3/fig_ch3_overview.png` / `fig3_tb_overview.png`',
              '- `ch4/fig_ch4_overview.png` / `fig4_tb_overview.png`',
              '- per-metric png under `ch3/` and `ch4/`',
              '- raw csv: every scalar tag under `csv/<run>/`',
              ]
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print('DONE →', out)


if __name__ == '__main__':
    main()
