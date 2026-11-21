#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-quality TensorBoard export for thesis Ch3 + Ch4.

Design (matches good tb_key_curves.png style):
  - x-axis = epoch (not noisy step)
  - markers on epoch points, thin connecting lines
  - comparable runs only on the same axes
  - clean display names
  - dump ALL epoch scalars to wide CSV
  - one PNG per key metric + multi-panel overviews
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

EXP = Path('/home/ws/navsim_workspace/exp')
OUT = Path('/home/ws/navsim_workspace/exp/ch5_figs/tb_curves_pub')

# path, display name, color
RUNS = {
    # Ch3
    'p1_base': (
        EXP / 'ch3_df/p1_base/2026.10.23.19.54.55/lightning_logs/version_0',
        'P1-base', '#7F7F7F'),
    'p1_center': (
        EXP / 'ch3_df/p1_center/2026.10.26.05.13.51/lightning_logs/version_0',
        'P1-center', '#0072B2'),
    'p2_center18': (
        EXP / 'ch3_df/p2_center18/2026.10.29.13.52.58/lightning_logs/version_0',
        'P2-center', '#D55E00'),
    'v2_base': (
        EXP / 'ch3_df_v2/p1_base/2026.11.10.20.47.31/lightning_logs/version_0',
        'P1-v2 base', '#56B4E9'),
    'v2_center': (
        EXP / 'ch3_df_v2/p1_center/2026.11.13.05.59.35/lightning_logs/version_0',
        'P1-v2 center', '#E69F00'),
    'ch3_guidance': (
        EXP / 'train_dp_ch3_scorer_guidance/2026.10.14.20.02.57/lightning_logs/version_0',
        'Scorer-guidance', '#009E73'),
    'ch3_supp': (
        EXP / 'ch3_supplement/control/2026.10.18.19.42.46/lightning_logs/version_0',
        'Supp-control', '#CC79A7'),
    'fm_joint': (
        EXP / 'train_dp_official_fm_joint/2026.08.27.15.46.57/lightning_logs/version_0',
        'FM-joint', '#000000'),
    # Ch4 — prefer tb_compare (same-iter) when available
    'gtrs': (
        EXP / 'ablation_same_iter_r34/tb_compare/official_gtrs_r34',
        'GTRS-r34', '#009E73'),
    'drivesuprim': (
        EXP / 'ablation_same_iter_r34/tb_compare/official_drivesuprim_r34',
        'DriveSuprim', '#0072B2'),
    'opd': (
        EXP / 'ablation_same_iter_r34/tb_compare/opd',
        'OPD', '#D55E00'),
    'opd_train': (
        EXP / 'training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256-9cd57b3/2026.09.29.03.48.37/lightning_logs/version_0',
        'OPD-9cd57b3', '#E69F00'),
    'opd_prev': (
        EXP / 'training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256/2026.09.14.11.14.19/lightning_logs/version_0',
        'OPD-prev', '#CC79A7'),
    'ds_vov': (
        EXP / 'training/gtrs_aug_drivesuprim_vov/rot_30-p_0.5/stage_layers_3-topks_256/2026.09.11.19.28.36/lightning_logs/version_0',
        'DS-VoV', '#56B4E9'),
}


def style():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'legend.frameon': True,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '#CCCCCC',
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.28,
        'grid.linewidth': 0.6,
        'lines.linewidth': 1.7,
        'lines.markersize': 5.5,
    })
    return plt


def load(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return tag -> (steps, values, wall_times)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(path), size_guidance={'scalars': 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get('scalars', []):
        ev = ea.Scalars(tag)
        out[tag] = (
            np.array([e.step for e in ev], np.int64),
            np.array([e.value for e in ev], np.float64),
            np.array([e.wall_time for e in ev], np.float64),
        )
    return out


def step_to_epoch(data: Dict) -> Dict[int, float]:
    """Map step -> epoch float using 'epoch' tag if present."""
    if 'epoch' not in data:
        return {}
    steps, vals, _ = data['epoch']
    return {int(s): float(v) for s, v in zip(steps, vals)}


def as_epoch_series(data: Dict, tag: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Convert a *_epoch (or any) tag to (epoch_x, values), one point per epoch.

    Prefer unique integer epochs; take last value in each epoch bin.
    """
    if tag not in data:
        return None
    steps, vals, _ = data[tag]
    epmap = step_to_epoch(data)
    if epmap:
        epochs = np.array([epmap.get(int(s), np.nan) for s in steps], float)
        # if many NaN, fall back to step-normalized
        if np.isnan(epochs).mean() > 0.5:
            epochs = steps.astype(float)
    else:
        epochs = steps.astype(float)

    # bin by rounded epoch
    order = np.argsort(epochs)
    epochs, vals = epochs[order], vals[order]
    # keep last sample per integer epoch
    bins = {}
    for e, v in zip(epochs, vals):
        if np.isnan(e):
            continue
        bins[int(round(e))] = float(v)
    if not bins:
        return None
    xs = np.array(sorted(bins.keys()), float)
    ys = np.array([bins[int(x)] for x in xs], float)
    return xs, ys


def dump_wide_epoch_csv(name: str, data: Dict, out: Path):
    tags = sorted([t for t in data if t.endswith('_epoch')])
    if not tags:
        return
    # collect all integer epochs
    series = {t: as_epoch_series(data, t) for t in tags}
    series = {t: s for t, s in series.items() if s is not None}
    all_ep = sorted(set().union(*[set(s[0].astype(int).tolist()) for s in series.values()]))
    maps = {t: dict(zip(s[0].astype(int).tolist(), s[1].tolist())) for t, s in series.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch'] + list(series.keys()))
        for e in all_ep:
            w.writerow([e] + [maps[t].get(e, '') for t in series])


def dump_all_tag_csv(name: str, data: Dict, out_dir: Path):
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    for tag, (steps, vals, _) in data.items():
        safe = tag.replace('/', '__')
        with open(d / f'{safe}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'value'])
            w.writerows(zip(steps.tolist(), vals.tolist()))


def plot_epoch_panel(ax, runs_data, run_keys: List[str], tag: str, title: str,
                     ylabel: str = '', ylim=None):
    for key in run_keys:
        if key not in runs_data:
            continue
        path, label, color = RUNS[key]
        ser = as_epoch_series(runs_data[key], tag)
        if ser is None:
            continue
        x, y = ser
        ax.plot(x, y, '-o', color=color, label=label, markerfacecolor='white',
                markeredgewidth=1.2, markersize=5)
    ax.set_title(title, pad=4)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel('epoch')
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc='best', fontsize=7)


def save_grid(plt, panels, out: Path, ncols=3, figsize=None, legend_note=None):
    n = len(panels)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (3.6 * ncols, 2.85 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, (run_keys, tag, title, ylabel, ylim) in enumerate(panels):
        r, c = divmod(i, ncols)
        plot_epoch_panel(axes[r][c], RUN_DATA, run_keys, tag, title, ylabel, ylim)
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].axis('off')
    if legend_note:
        fig.suptitle(legend_note, fontsize=11, y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print('wrote', out)


def save_single(plt, run_keys, tag, title, out, ylabel=''):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    plot_epoch_panel(ax, RUN_DATA, run_keys, tag, title, ylabel)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print('wrote', out)


RUN_DATA: Dict[str, Dict] = {}


def main():
    global RUN_DATA
    plt = style()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'csv').mkdir(exist_ok=True)
    (OUT / 'ch3').mkdir(exist_ok=True)
    (OUT / 'ch4').mkdir(exist_ok=True)
    (OUT / 'ch3/metrics').mkdir(exist_ok=True)
    (OUT / 'ch4/metrics').mkdir(exist_ok=True)

    meta = {}
    for key, (path, label, color) in RUNS.items():
        if not path.exists():
            print('MISSING', key, path)
            continue
        print('load', key, label)
        data = load(path)
        RUN_DATA[key] = data
        dump_all_tag_csv(key, data, OUT / 'csv')
        dump_wide_epoch_csv(key, data, OUT / 'csv' / key / '_epoch_wide.csv')
        meta[key] = {'label': label, 'path': str(path), 'n_tags': len(data),
                     'color': color, 'epoch_tags': sorted(t for t in data if t.endswith('_epoch'))}
        print(f'  tags={len(data)} epoch_tags={len(meta[key]["epoch_tags"])}')

    with open(OUT / 'manifest.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # ===================== Ch3 =====================
    p1p2 = ['p1_base', 'p1_center', 'p2_center18']
    v2 = ['v2_base', 'v2_center']
    centers = ['p1_center', 'p2_center18', 'v2_center']

    # Main overview like existing good figure
    save_grid(plt, [
        (p1p2, 'val/loss_epoch', 'val loss', 'loss', None),
        (p1p2, 'val/fm_x1_ade_epoch', 'val FM ADE', 'ADE (m)', None),
        (centers, 'train/fm_center_ade_epoch', 'train center ADE', 'ADE (m)', None),
        (centers, 'train/fm_center_loss_epoch', 'train center loss', 'loss', None),
        (p1p2, 'val/bev_semantic_loss_epoch', 'val BEV semantic', 'loss', None),
        (p1p2, 'val/fm_x1_fde_epoch', 'val FM FDE', 'FDE (m)', None),
    ], OUT / 'ch3/fig_ch3_p1p2_key.png', ncols=3,
       legend_note='Ch3 FM-DP: P1 vs P2 (epoch curves)')

    save_grid(plt, [
        (v2, 'val/loss_epoch', 'val loss', 'loss', None),
        (v2, 'val/fm_x1_ade_epoch', 'val FM ADE', 'ADE (m)', None),
        (v2, 'val/fm_x1_fde_epoch', 'val FM FDE', 'FDE (m)', None),
        (v2, 'val/fm_zero_ade_epoch', 'val FM-zero ADE', 'ADE (m)', None),
        (v2, 'val/fm_zero_fde_epoch', 'val FM-zero FDE', 'FDE (m)', None),
        (v2, 'val/fm_zero_jerk_epoch', 'val FM-zero jerk', 'jerk', None),
        (v2, 'val/fm_zero_accel_epoch', 'val FM-zero accel', 'accel', None),
        (['v2_center'], 'train/fm_center_ade_epoch', 'train center ADE', 'ADE (m)', None),
        (v2, 'val/bev_semantic_loss_epoch', 'val BEV semantic', 'loss', None),
    ], OUT / 'ch3/fig_ch3_v2_key.png', ncols=3,
       legend_note='Ch3 Phase1-v2: base vs center')

    # Extra ch3 singles
    for tag, title, ylab in [
        ('val/loss_epoch', 'val loss', 'loss'),
        ('val/fm_x1_ade_epoch', 'val FM ADE', 'ADE (m)'),
        ('val/fm_x1_fde_epoch', 'val FM FDE', 'FDE (m)'),
        ('val/bev_semantic_loss_epoch', 'val BEV semantic', 'loss'),
        ('train/fm_center_ade_epoch', 'train center ADE', 'ADE (m)'),
        ('train/fm_center_loss_epoch', 'train center loss', 'loss'),
        ('train/fm_center_fde_epoch', 'train center FDE', 'FDE (m)'),
        ('val/fm_zero_ade_epoch', 'val FM-zero ADE', 'ADE (m)'),
        ('val/fm_zero_fde_epoch', 'val FM-zero FDE', 'FDE (m)'),
        ('val/fm_zero_jerk_epoch', 'val FM-zero jerk', 'jerk'),
        ('val/fm_zero_accel_epoch', 'val FM-zero accel', 'accel'),
        ('train/loss_epoch', 'train loss (epoch)', 'loss'),
        ('train/dp_loss_epoch', 'train DP loss', 'loss'),
        ('train/fm_x1_ade_epoch', 'train FM ADE', 'ADE (m)'),
        ('val/fm_aux_pos_loss_epoch', 'val FM aux pos', 'loss'),
        ('val/fm_kin_loss_epoch', 'val FM kin', 'loss'),
    ]:
        keys = [k for k in ['p1_base', 'p1_center', 'p2_center18', 'v2_base', 'v2_center',
                            'ch3_guidance', 'ch3_supp', 'fm_joint']
                if k in RUN_DATA and tag in RUN_DATA[k]]
        if len(keys) < 1:
            continue
        safe = tag.replace('/', '__')
        save_single(plt, keys, tag, f'Ch3 | {title}', OUT / 'ch3/metrics' / f'{safe}.png', ylab)

    # ===================== Ch4 =====================
    # Fair same-iter comparison
    same = ['gtrs', 'drivesuprim', 'opd']
    opd_only = ['opd', 'opd_train', 'opd_prev']

    save_grid(plt, [
        (same, 'val/loss_epoch', 'val loss', 'loss', None),
        (same, 'val/loss-ori_epoch', 'val loss-ori', 'loss', None),
        (same, 'val/loss-aug_epoch', 'val loss-aug', 'loss', None),
        (same, 'val/loss-soft_epoch', 'val loss-soft', 'loss', None),
        (same, 'val/imi_loss-ori_epoch', 'val imi-ori', 'loss', None),
        (same, 'val/imi_loss-soft_epoch', 'val imi-soft', 'loss', None),
        (same, 'val/loss-refinement_ori_epoch', 'val refinement-ori', 'loss', None),
        (same, 'val/pdm_comfort_loss-ori_epoch', 'val comfort-ori', 'loss', None),
        (['opd'], 'val/loss_opd-opd_epoch', 'val OPD distill', 'loss', None),
    ], OUT / 'ch4/fig_ch4_sameiter_key.png', ncols=3,
       legend_note='Ch4 same-iter ablation: GTRS vs DriveSuprim vs OPD')

    save_grid(plt, [
        (opd_only, 'val/loss_opd-opd_epoch', 'val OPD distill total', 'loss', None),
        (opd_only, 'val/distill_imi-opd_epoch', 'val distill imi', 'loss', None),
        (opd_only, 'val/distill_head-opd_epoch', 'val distill head', 'loss', None),
        (opd_only, 'val/distill_recall-opd_epoch', 'val distill recall', 'loss', None),
        (opd_only, 'val/distill_prod-opd_epoch', 'val distill prod', 'loss', None),
        (opd_only, 'val/distill_refine-opd_epoch', 'val distill refine', 'loss', None),
        (opd_only, 'train/loss_opd-opd_epoch', 'train OPD distill', 'loss', None),
        (opd_only, 'train/distill_imi-opd_epoch', 'train distill imi', 'loss', None),
        (opd_only, 'train/distill_recall-opd_epoch', 'train distill recall', 'loss', None),
    ], OUT / 'ch4/fig_ch4_opd_distill.png', ncols=3,
       legend_note='Ch4 OPD distillation breakdown')

    # PDM heads for same-iter
    pdm_tags = [
        ('val/pdm_noc_loss-ori_epoch', 'NOC'),
        ('val/pdm_da_loss-ori_epoch', 'DA'),
        ('val/pdm_ttc_loss-ori_epoch', 'TTC'),
        ('val/pdm_progress_loss-ori_epoch', 'Progress'),
        ('val/pdm_ddc_loss-ori_epoch', 'DDC'),
        ('val/pdm_lk_loss-ori_epoch', 'Lane-keep'),
        ('val/pdm_tl_loss-ori_epoch', 'Traffic-light'),
        ('val/pdm_comfort_loss-ori_epoch', 'Comfort'),
        ('val/imi_loss-ori_epoch', 'IMI'),
    ]
    save_grid(plt, [
        (same, tag, f'val {lab}-ori', 'loss', None) for tag, lab in pdm_tags
    ], OUT / 'ch4/fig_ch4_pdm_heads.png', ncols=3,
       legend_note='Ch4 PDM head losses (ori branch)')

    # soft vs ori for OPD
    save_grid(plt, [
        (['opd', 'opd_train'], 'val/imi_loss-ori_epoch', 'imi-ori', 'loss', None),
        (['opd', 'opd_train'], 'val/imi_loss-soft_epoch', 'imi-soft', 'loss', None),
        (['opd', 'opd_train'], 'val/imi_loss-aug_epoch', 'imi-aug', 'loss', None),
        (['opd', 'opd_train'], 'val/loss-ori_epoch', 'loss-ori', 'loss', None),
        (['opd', 'opd_train'], 'val/loss-soft_epoch', 'loss-soft', 'loss', None),
        (['opd', 'opd_train'], 'val/loss-aug_epoch', 'loss-aug', 'loss', None),
    ], OUT / 'ch4/fig_ch4_ori_soft_aug.png', ncols=3,
       legend_note='Ch4 OPD: ori / soft / aug branches')

    # distill head detail
    head_tags = [
        'val/distill_head_no_at_fault_collisions-opd_epoch',
        'val/distill_head_drivable_area_compliance-opd_epoch',
        'val/distill_head_time_to_collision_within_bound-opd_epoch',
        'val/distill_head_ego_progress-opd_epoch',
        'val/distill_head_driving_direction_compliance-opd_epoch',
        'val/distill_head_lane_keeping-opd_epoch',
        'val/distill_head_traffic_light_compliance-opd_epoch',
        'val/distill_head_history_comfort-opd_epoch',
    ]
    short = ['NOC', 'DA', 'TTC', 'Progress', 'DDC', 'LK', 'TL', 'Comfort']
    save_grid(plt, [
        (opd_only, t, f'distill {s}', 'loss', None) for t, s in zip(head_tags, short)
    ], OUT / 'ch4/fig_ch4_distill_heads.png', ncols=4,
       legend_note='Ch4 OPD distill per-head')

    # Every epoch tag for OPD / GTRS / DS as individual metric figs (complete dump)
    for key in ['opd', 'gtrs', 'drivesuprim', 'opd_train', 'v2_center', 'p2_center18']:
        if key not in RUN_DATA:
            continue
        ep_tags = sorted(t for t in RUN_DATA[key] if t.endswith('_epoch'))
        for tag in ep_tags:
            safe = tag.replace('/', '__')
            chapter = 'ch3' if key in ('v2_center', 'p2_center18', 'p1_center', 'p1_base',
                                       'v2_base', 'ch3_guidance', 'ch3_supp', 'fm_joint') else 'ch4'
            # only plot this run alone for complete archive
            fig, ax = plt.subplots(figsize=(4.6, 3.0))
            plot_epoch_panel(ax, RUN_DATA, [key], tag, f'{RUNS[key][1]} | {tag}', '')
            fig.tight_layout()
            dest = OUT / chapter / 'metrics' / key
            dest.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest / f'{safe}.png')
            fig.savefig(dest / f'{safe}.pdf')
            plt.close(fig)
        print(f'per-tag figs for {key}: {len(ep_tags)}')

    # Copy main overviews to ch5_figs root
    import shutil
    root = OUT.parent
    for src, dst in [
        (OUT / 'ch3/fig_ch3_p1p2_key.png', root / 'fig3_tb_overview.png'),
        (OUT / 'ch3/fig_ch3_v2_key.png', root / 'fig3_tb_v2.png'),
        (OUT / 'ch4/fig_ch4_sameiter_key.png', root / 'fig4_tb_overview.png'),
        (OUT / 'ch4/fig_ch4_opd_distill.png', root / 'fig4_tb_opd_distill.png'),
        (OUT / 'ch4/fig_ch4_pdm_heads.png', root / 'fig4_tb_pdm_heads.png'),
    ]:
        if src.exists():
            shutil.copy2(src, dst)
            pdf = src.with_suffix('.pdf')
            if pdf.exists():
                shutil.copy2(pdf, Path(dst).with_suffix('.pdf'))

    # README
    n_png = len(list(OUT.rglob('*.png')))
    n_csv = len(list((OUT / 'csv').rglob('*.csv')))
    (OUT / 'README.md').write_text(
        f"""# Publication TB curves (Ch3 + Ch4)

- Runs loaded: {len(RUN_DATA)}
- PNG figures: {n_png}
- CSV files: {n_csv}
- Style: **epoch x-axis**, markers, Okabe–Ito, comparable groups only

## Main figures
- `ch3/fig_ch3_p1p2_key.png` — P1/P2 key metrics
- `ch3/fig_ch3_v2_key.png` — Phase1-v2 base vs center (+ zero ADE/jerk)
- `ch4/fig_ch4_sameiter_key.png` — GTRS / DriveSuprim / OPD
- `ch4/fig_ch4_opd_distill.png` — OPD distill breakdown
- `ch4/fig_ch4_pdm_heads.png` — PDM heads
- `ch4/fig_ch4_distill_heads.png` — per-head distill
- `ch3/metrics/`, `ch4/metrics/` — every epoch tag

## CSV
`csv/<run>/` — all scalar tags + `_epoch_wide.csv`
"""
    )
    print('DONE', OUT, 'png=', n_png, 'csv=', n_csv)


if __name__ == '__main__':
    main()
