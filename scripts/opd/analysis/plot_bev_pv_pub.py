#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-quality BEV + PV decision / inference visualization (CVPR-style).

Draws physical-space decision figures — NOT embedding scatter plots:
  - BEV: map + agents + trajectory overlays (forward-up)
  - PV: front camera with the same trajectories projected into the image

Palette: Okabe–Ito (colorblind-safe). Layout: BEV | PV, plus method strips.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Okabe–Ito + paper hierarchy
C = {
    'gt':      '#000000',
    'zero':    '#0072B2',   # blue  — deterministic FM
    'medoid':  '#009E73',   # bluish-green
    'first':   '#E69F00',   # amber
    'ddpm':    '#D55E00',   # vermillion
    'samples': '#56B4E9',   # sky   — FM sample cloud
}


def style():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'lines.antialiased': True,
        'path.simplify': False,
    })
    return plt


def load_props(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def as_poses(obj) -> np.ndarray:
    if hasattr(obj, 'poses'):
        return np.asarray(obj.poses, dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


def make_traj(poses: np.ndarray):
    from navsim.common.dataclasses import Trajectory, NAVSIM_INTERVAL_LENGTH
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2:
        raise ValueError(f'expected (T,3) poses, got {poses.shape}')
    samp = TrajectorySampling(num_poses=int(poses.shape[0]),
                              interval_length=NAVSIM_INTERVAL_LENGTH)
    return Trajectory(poses, samp)


def pick_prop(rec, key='trajectory', idx=0) -> np.ndarray:
    if isinstance(rec, dict):
        if key in rec and rec[key] is not None:
            a = as_poses(rec[key])
            return a if a.ndim == 2 else a[idx]
        if 'proposals' in rec:
            a = as_poses(rec['proposals'])
            return a if a.ndim == 2 else a[idx]
    return as_poses(rec)


def densify_xy(poses: np.ndarray, n: int = 40) -> np.ndarray:
    """Cubic-ish polyline densification in ego XY (heading ignored for draw)."""
    xy = np.asarray(poses)[:, :2]
    if len(xy) < 2:
        return np.c_[xy, np.zeros(len(xy))]
    # include origin
    xy = np.vstack([np.zeros((1, 2)), xy])
    t = np.linspace(0, 1, len(xy))
    tt = np.linspace(0, 1, n)
    x = np.interp(tt, t, xy[:, 0])
    y = np.interp(tt, t, xy[:, 1])
    return np.c_[x, y, np.zeros(n)]


# ---------------------------------------------------------------------------
# Geometry: ego/lidar (x-forward, y-left, z-up) → camera pixels
# ---------------------------------------------------------------------------
def ego_traj_to_cam_pixels(traj_xyh: np.ndarray, camera, z: float = 0.4
                           ) -> Tuple[np.ndarray, np.ndarray]:
    from navsim.visualization.camera import _transform_points_to_image

    pts = np.zeros((len(traj_xyh), 3), dtype=np.float64)
    pts[:, 0] = traj_xyh[:, 0]
    pts[:, 1] = traj_xyh[:, 1]
    pts[:, 2] = z

    R_s2l = np.asarray(camera.sensor2lidar_rotation, dtype=np.float64)
    t_s2l = np.asarray(camera.sensor2lidar_translation, dtype=np.float64)
    R_l2c = np.linalg.inv(R_s2l)
    t_l2c = -(R_l2c @ t_s2l)
    pts_cam = (R_l2c @ pts.T).T + t_l2c

    K = np.asarray(camera.intrinsics, dtype=np.float64)
    h, w = camera.image.shape[:2]
    pix, mask = _transform_points_to_image(pts_cam.astype(np.float32), K.astype(np.float32), (h, w))
    return pix, mask


def _plot_pix_polyline(ax, pix, mask, color, lw, zorder=5, halo=True, marker_end=True):
    if mask.sum() < 2:
        return
    idx = np.where(mask)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for seg in np.split(idx, splits):
        if len(seg) < 2:
            continue
        xs, ys = pix[seg, 0], pix[seg, 1]
        if halo:
            ax.plot(xs, ys, color='white', lw=lw + 2.2, solid_capstyle='round',
                    solid_joinstyle='round', zorder=zorder - 0.1, alpha=0.9)
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round',
                solid_joinstyle='round', zorder=zorder)
    if marker_end and mask[-1]:
        ax.scatter(pix[-1, 0], pix[-1, 1], c=color, s=22, zorder=zorder + 1,
                   edgecolors='white', linewidths=0.6)


def draw_traj_on_image(ax, traj, color, lw=2.5, camera=None, z=0.4,
                       halo=True, densify=True, marker_end=True):
    if camera is None or traj is None or len(traj) == 0:
        return
    full = densify_xy(traj, n=48) if densify else np.vstack(
        [np.zeros((1, 3)), traj[:, :3] if traj.shape[1] >= 3 else
         np.c_[traj[:, :2], np.zeros(len(traj))]]
    )
    # drop points behind / under camera (near ego origin often z_cam < 0)
    pix, mask = ego_traj_to_cam_pixels(full[:, :3], camera, z=z)
    _plot_pix_polyline(ax, pix, mask, color, lw, halo=halo, marker_end=marker_end)


# ---------------------------------------------------------------------------
# BEV drawing (clean paper style)
# ---------------------------------------------------------------------------
def finish_bev_ax(ax, half_w: float = 22.0, rear: float = 8.0, forward: float = 40.0):
    """Navsim convention: plot(y, x), x-forward = up, invert x so +y is left."""
    ax.set_aspect('equal')
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(-rear, forward)
    ax.invert_xaxis()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color('#333333')


def draw_bev(ax, scene, trajs: List[Tuple[str, np.ndarray, str, float, float]],
             sample_cloud: Optional[np.ndarray] = None,
             half_w: float = 22.0, rear: float = 8.0, forward: float = 40.0):
    from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
    from navsim.visualization.config import TRAJECTORY_CONFIG

    frame_idx = scene.scene_metadata.num_history_frames - 1
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

    if sample_cloud is not None:
        cloud = np.asarray(sample_cloud)
        if cloud.ndim == 2:
            cloud = cloud[None]
        # subsample for clarity
        n = min(50, len(cloud))
        step = max(1, len(cloud) // n)
        st = dict(TRAJECTORY_CONFIG['agent'])
        st.update(line_color=C['samples'], line_color_alpha=0.22, line_width=0.85,
                  line_style='-', marker='', marker_size=0,
                  marker_edge_color=C['samples'], zorder=2)
        for i in range(0, len(cloud), step):
            add_trajectory_to_bev_ax(ax, make_traj(cloud[i]), st)

    for name, poses, color, lw, alpha in trajs:
        st = dict(TRAJECTORY_CONFIG['agent'])
        st.update(line_color=color, fill_color=color, line_color_alpha=alpha,
                  line_width=lw, line_style='-', marker='o', marker_size=3.2,
                  marker_edge_color='white', zorder=5 if name == 'GT' else 4)
        add_trajectory_to_bev_ax(ax, make_traj(poses), st)

    finish_bev_ax(ax, half_w=half_w, rear=rear, forward=forward)


def build_loader(tokens: List[str], log_path: str, sensor_path: str):
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    data_path = Path(log_path)
    log_names = [p.stem for p in data_path.glob('*.pkl')]
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True,
                     log_names=log_names, tokens=tokens)
    return SceneLoader(
        data_path=data_path,
        original_sensor_path=Path(sensor_path),
        scene_filter=sf,
        sensor_config=SensorConfig.build_all_sensors(),
        synthetic_sensor_path=None,
        synthetic_scenes_path=None,
    )


def legend_proxy(plt, keys: Optional[Sequence[str]] = None):
    from matplotlib.lines import Line2D
    items = [
        ('gt', 'Human', 2.6),
        ('zero', 'FM zero', 2.2),
        ('medoid', 'FM medoid', 2.2),
        ('first', 'FM first', 2.2),
        ('ddpm', 'DDPM', 2.2),
        ('samples', 'FM samples', 1.3),
    ]
    if keys is not None:
        items = [it for it in items if it[0] in keys]
    return [
        Line2D([0], [0], color=C[k], lw=lw, label=lab,
               alpha=0.55 if k == 'samples' else 1.0)
        for k, lab, lw in items
    ]


def panel_label(ax, text, dark=False):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', ha='left',
            fontsize=11, fontweight='bold',
            color='white' if dark else '#111111',
            bbox=dict(boxstyle='square,pad=0.22',
                      fc=('#000000' if dark else 'white'),
                      ec='none', alpha=0.45 if dark else 0.88))


# ---------------------------------------------------------------------------
# Figure types
# ---------------------------------------------------------------------------
def _traj_bundle(scene, zero_rec, medoid_rec, first_rec, ddpm_rec):
    human = as_poses(scene.get_future_trajectory())
    props = as_poses(medoid_rec['proposals'])
    if props.ndim == 2:
        props = props[None]
    trajs = [
        ('GT', human, C['gt'], 2.9, 1.0),
        ('FM zero', pick_prop(zero_rec, key='trajectory'), C['zero'], 2.5, 1.0),
        ('FM medoid', pick_prop(medoid_rec, key='trajectory'), C['medoid'], 2.3, 0.95),
        ('FM first', pick_prop(first_rec, key='trajectory'), C['first'], 2.1, 0.9),
        ('DDPM', pick_prop(ddpm_rec, key='trajectory'), C['ddpm'], 2.1, 0.9),
    ]
    return human, props, trajs


def fig_scene_bev_pv(scene, zero_rec, medoid_rec, first_rec, ddpm_rec, out: Path):
    """One scene: BEV (left) + PV front (right)."""
    plt = style()
    _, props, trajs = _traj_bundle(scene, zero_rec, medoid_rec, first_rec, ddpm_rec)
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    cam = frame.cameras.cam_f0

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7),
                             gridspec_kw={'width_ratios': [1.0, 1.45], 'wspace': 0.05})

    draw_bev(axes[0], scene, trajs, sample_cloud=props)
    panel_label(axes[0], '(a) BEV')

    axes[1].imshow(cam.image)
    axes[1].set_axis_off()
    # samples first (under), then main trajs
    n_s = min(20, len(props))
    for i in np.linspace(0, len(props) - 1, n_s, dtype=int):
        draw_traj_on_image(axes[1], props[i], C['samples'], lw=1.0, camera=cam,
                           halo=False, marker_end=False)
    for name, poses, color, lw, _ in trajs:
        draw_traj_on_image(axes[1], poses, color, lw=lw + 0.5, camera=cam, halo=True)
    panel_label(axes[1], '(b) PV (front camera)', dark=True)

    fig.legend(handles=legend_proxy(plt), loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.03), frameon=False, columnspacing=1.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print('wrote', out)


def fig_method_row(scene, methods: List[Tuple[str, np.ndarray, str]], cloud, out: Path):
    """Multi-column BEV: one method per column."""
    plt = style()
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(2.65 * n, 2.9))
    if n == 1:
        axes = [axes]
    human = as_poses(scene.get_future_trajectory())
    for ax, (name, poses, color) in zip(axes, methods):
        trajs = [
            ('GT', human, C['gt'], 2.6, 1.0),
            (name, poses, color, 2.4, 1.0),
        ]
        show_cloud = cloud if name.startswith('FM') and 'zero' not in name.lower() else (
            cloud if 'medoid' in name.lower() or 'first' in name.lower() else None
        )
        # show sample cloud only for FM first / medoid panels
        draw_bev(ax, scene, trajs,
                 sample_cloud=cloud if ('medoid' in name.lower() or 'first' in name.lower()) else None)
        ax.set_title(name, fontsize=10, pad=3)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C['gt'], lw=2.4, label='Human')]
    seen = set()
    for name, _, color in methods:
        if name in seen:
            continue
        seen.add(name)
        handles.append(Line2D([0], [0], color=color, lw=2.2, label=name))
    handles.append(Line2D([0], [0], color=C['samples'], lw=1.2, alpha=0.55, label='FM samples'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print('wrote', out)


def fig_pv_methods(scene, methods: List[Tuple[str, np.ndarray, str]], out: Path):
    """Row of PV views with different inference methods."""
    plt = style()
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    cam = frame.cameras.cam_f0
    human = as_poses(scene.get_future_trajectory())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(3.15 * n, 2.35))
    if n == 1:
        axes = [axes]
    for ax, (name, poses, color) in zip(axes, methods):
        ax.imshow(cam.image)
        ax.set_axis_off()
        draw_traj_on_image(ax, human, C['gt'], lw=2.8, camera=cam)
        draw_traj_on_image(ax, poses, color, lw=3.0, camera=cam)
        ax.set_title(name, fontsize=10, pad=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print('wrote', out)


def fig_overlay(scene, methods: List[Tuple[str, np.ndarray, str]], cloud, out: Path):
    """Single BEV with all methods overlaid (clearest mode contrast)."""
    plt = style()
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 4.6))
    human = as_poses(scene.get_future_trajectory())
    trajs = [('GT', human, C['gt'], 3.0, 1.0)]
    for name, poses, color in methods:
        trajs.append((name, poses, color, 2.5, 1.0))
    draw_bev(ax, scene, trajs, sample_cloud=cloud, half_w=18, rear=6, forward=28)
    ax.set_title('Inference modes (same scene)', fontsize=11, pad=4)
    fig.legend(handles=legend_proxy(plt), loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print('wrote', out)


def fig_multisource_panel(scenes_data, out: Path):
    """Paper plate: N rows × (BEV | PV)."""
    plt = style()
    n = len(scenes_data)
    fig, axes = plt.subplots(n, 2, figsize=(10.2, 3.55 * n),
                             gridspec_kw={'width_ratios': [1.0, 1.4], 'wspace': 0.04, 'hspace': 0.08})
    if n == 1:
        axes = np.array([axes])
    for r, (scene, zero_rec, medoid_rec, first_rec, ddpm_rec) in enumerate(scenes_data):
        _, props, trajs = _traj_bundle(scene, zero_rec, medoid_rec, first_rec, ddpm_rec)
        frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
        cam = frame.cameras.cam_f0
        draw_bev(axes[r, 0], scene, trajs, sample_cloud=props,
                 half_w=20, rear=6, forward=36)
        if r == 0:
            panel_label(axes[r, 0], 'BEV')
        axes[r, 1].imshow(cam.image)
        axes[r, 1].set_axis_off()
        for i in np.linspace(0, len(props) - 1, min(15, len(props)), dtype=int):
            draw_traj_on_image(axes[r, 1], props[i], C['samples'], lw=0.9, camera=cam,
                               halo=False, marker_end=False)
        for name, poses, color, lw, _ in trajs:
            draw_traj_on_image(axes[r, 1], poses, color, lw=lw + 0.3, camera=cam)
        if r == 0:
            panel_label(axes[r, 1], 'PV', dark=True)
    fig.legend(handles=legend_proxy(plt), loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--fm-zero', required=True)
    ap.add_argument('--fm-first', required=True)
    ap.add_argument('--fm-medoid', required=True)
    ap.add_argument('--ddpm', required=True)
    ap.add_argument('--log-path', required=True)
    ap.add_argument('--sensor-path', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-scenes', type=int, default=4)
    args = ap.parse_args()

    tokens = [ln.strip() for ln in Path(args.tokens).read_text().splitlines() if ln.strip()]
    tokens = tokens[: args.n_scenes]

    zero = load_props(args.fm_zero)
    first = load_props(args.fm_first)
    medoid = load_props(args.fm_medoid)
    ddpm = load_props(args.ddpm)
    tokens = [t for t in tokens if t in zero and t in first and t in medoid and t in ddpm]
    assert tokens, 'no overlapping tokens'

    loader = build_loader(tokens, args.log_path, args.sensor_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plate = []
    first_tag = None
    for i, tok in enumerate(tokens):
        try:
            scene = loader.get_scene_from_token(tok)
        except Exception as e:
            print('skip', tok, e)
            continue
        tag = f'{i+1:02d}_{tok[:10]}'
        if first_tag is None:
            first_tag = tag
        fig_scene_bev_pv(scene, zero[tok], medoid[tok], first[tok], ddpm[tok],
                         out_dir / f'fig_bev_pv_{tag}.png')

        props = as_poses(medoid[tok]['proposals'])
        if props.ndim == 2:
            props = props[None]
        methods = [
            ('FM first', pick_prop(first[tok], key='trajectory'), C['first']),
            ('FM medoid', pick_prop(medoid[tok], key='trajectory'), C['medoid']),
            ('FM zero', pick_prop(zero[tok], key='trajectory'), C['zero']),
            ('DDPM', pick_prop(ddpm[tok], key='trajectory'), C['ddpm']),
        ]
        fig_method_row(scene, methods, props, out_dir / f'fig_bev_methods_{tag}.png')
        fig_pv_methods(scene, methods, out_dir / f'fig_pv_methods_{tag}.png')
        fig_overlay(scene, methods, props, out_dir / f'fig_bev_overlay_{tag}.png')
        plate.append((scene, zero[tok], medoid[tok], first[tok], ddpm[tok]))

    if plate:
        fig_multisource_panel(plate[:4], out_dir / 'fig_bev_pv_plate.png')

    # convenience copies into ch5_figs root for thesis inclusion
    import shutil
    root = out_dir.parent
    copies = [('fig_bev_pv_plate.png', 'fig5_decision_bev_pv_plate.png')]
    if first_tag is not None:
        copies += [
            (f'fig_bev_pv_{first_tag}.png', 'fig5_decision_bev_pv.png'),
            (f'fig_bev_methods_{first_tag}.png', 'fig5_decision_bev_methods.png'),
            (f'fig_pv_methods_{first_tag}.png', 'fig5_decision_pv_methods.png'),
            (f'fig_bev_overlay_{first_tag}.png', 'fig5_decision_bev_overlay.png'),
        ]
    for src_name, dst_name in copies:
        src = out_dir / src_name
        if src.exists():
            shutil.copy2(src, root / dst_name)
            pdf = src.with_suffix('.pdf')
            if pdf.exists():
                shutil.copy2(pdf, (root / dst_name).with_suffix('.pdf'))
            print('copy', src_name, '→', dst_name)

    print('DONE →', out_dir)


if __name__ == '__main__':
    main()
