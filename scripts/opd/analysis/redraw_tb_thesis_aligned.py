#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw thesis TB overview figures with aligned epoch intervals.

Fixes:
  - all panels in one figure share the same x-lim / integer ticks
  - short-budget (5ep) arms are not mixed with 18ep on the same axis
  - include newly trained Official DDPM base (p1_official)
  - one shared legend; Okabe–Ito; 400 dpi PNG+PDF
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

EXP = Path("/home/ws/navsim_workspace/exp")
OUT = Path("/home/ws/navsim_workspace/exp/ch5_figs")
PUB = OUT / "tb_curves_pub" / "aligned"
PUB.mkdir(parents=True, exist_ok=True)

RUNS = {
    "official": (
        EXP / "ch3_df_v2/p1_official/2026.11.17.17.29.49/lightning_logs/version_0",
        "Official DDPM",
        "#7F7F7F",
    ),
    "v2_base": (
        EXP / "ch3_df_v2/p1_base/2026.11.10.20.47.31/lightning_logs/version_0",
        "FM-base",
        "#56B4E9",
    ),
    "v2_center": (
        EXP / "ch3_df_v2/p1_center/2026.11.13.05.59.35/lightning_logs/version_0",
        "FM+center",
        "#D55E00",
    ),
    "p1_base": (
        EXP / "ch3_df/p1_base/2026.10.23.19.54.55/lightning_logs/version_0",
        "P1-base (v1)",
        "#0072B2",
    ),
    "p1_center": (
        EXP / "ch3_df/p1_center/2026.10.26.05.13.51/lightning_logs/version_0",
        "P1-center (v1)",
        "#E69F00",
    ),
    "p2_center18": (
        EXP / "ch3_df/p2_center18/2026.10.29.13.52.58/lightning_logs/version_0",
        "P2-center (18ep)",
        "#009E73",
    ),
    "gtrs": (
        EXP / "ablation_same_iter_r34/tb_compare/official_gtrs_r34",
        "GTRS-r34",
        "#009E73",
    ),
    "drivesuprim": (
        EXP / "ablation_same_iter_r34/tb_compare/official_drivesuprim_r34",
        "DriveSuprim",
        "#0072B2",
    ),
    "opd": (
        EXP / "ablation_same_iter_r34/tb_compare/opd",
        "OPD",
        "#D55E00",
    ),
}


def style():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for p in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    ):
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CCCCCC",
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.55,
            "axes.axisbelow": True,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
        }
    )
    return plt


def load(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        out[tag] = (
            np.array([e.step for e in ev], np.int64),
            np.array([e.value for e in ev], np.float64),
        )
    return out


def as_epoch_series(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]], tag: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if tag not in data:
        return None
    steps, vals = data[tag]
    epmap = {}
    if "epoch" in data:
        es, ev = data["epoch"]
        epmap = {int(s): float(v) for s, v in zip(es, ev)}
    if epmap:
        epochs = np.array([epmap.get(int(s), np.nan) for s in steps], float)
        if np.isnan(epochs).mean() > 0.5:
            epochs = steps.astype(float)
    else:
        epochs = steps.astype(float)
    bins: Dict[int, float] = {}
    for e, v in zip(epochs, vals):
        if np.isnan(e):
            continue
        bins[int(round(float(e)))] = float(v)
    if not bins:
        return None
    xs = np.array(sorted(bins.keys()), float)
    ys = np.array([bins[int(x)] for x in xs], float)
    return xs, ys


def clip_xy(x: np.ndarray, y: np.ndarray, xmax: Optional[float]):
    if xmax is None:
        return x, y
    m = x <= xmax + 1e-9
    return x[m], y[m]


def nice_ylim(ys_list: List[np.ndarray], pad_frac: float = 0.08):
    vals = np.concatenate([y for y in ys_list if len(y)])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return lo - 0.05, hi + 0.05
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def plot_panel(
    ax,
    runs_data,
    keys: List[str],
    tag: str,
    title: str,
    ylabel: str,
    xlim: Tuple[float, float],
    mark_early_stop: Optional[float] = None,
):
    from matplotlib.ticker import MaxNLocator

    collected = []
    for key in keys:
        if key not in runs_data:
            continue
        _, label, color = RUNS[key]
        ser = as_epoch_series(runs_data[key], tag)
        if ser is None:
            continue
        x, y = clip_xy(ser[0], ser[1], xlim[1])
        if len(x) == 0:
            continue
        ax.plot(
            x,
            y,
            "-o",
            color=color,
            label=label,
            markerfacecolor="white",
            markeredgewidth=1.25,
            markersize=5.2,
        )
        collected.append(y)
    ax.set_title(title, pad=4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("epoch")
    ax.set_xlim(xlim[0] - 0.15, xlim[1] + 0.15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=int(xlim[1] - xlim[0]) + 1))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    if collected:
        ax.set_ylim(*nice_ylim(collected))
    if mark_early_stop is not None:
        ax.axvline(mark_early_stop, color="#444444", ls="--", lw=1.0, alpha=0.7)
        ymin, ymax = ax.get_ylim()
        ax.text(
            mark_early_stop + 0.15,
            ymax - 0.04 * (ymax - ymin),
            f"early-stop ep{int(mark_early_stop)}",
            fontsize=7,
            color="#333333",
            va="top",
        )


def save_aligned_grid(
    plt,
    runs_data,
    panels,
    out: Path,
    xlim: Tuple[float, float],
    ncols: int = 3,
    title: str = "",
    mark_early_stop: Optional[float] = None,
):
    n = len(panels)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.55 * ncols, 2.75 * nrows), sharex=True, squeeze=False
    )
    handles, labels = [], []
    for i, (keys, tag, ptitle, ylabel) in enumerate(panels):
        r, c = divmod(i, ncols)
        plot_panel(
            axes[r][c],
            runs_data,
            keys,
            tag,
            ptitle,
            ylabel,
            xlim,
            mark_early_stop=mark_early_stop,
        )
        h, lab = axes[r][c].get_legend_handles_labels()
        for hh, ll in zip(h, lab):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
        # remove per-axis legend
        leg = axes[r][c].get_legend()
        if leg is not None:
            leg.remove()
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].axis("off")
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 5),
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
        )
    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)


def dump_epoch_csv(name: str, data, out: Path):
    tags = sorted(t for t in data if t.endswith("_epoch"))
    series = {t: as_epoch_series(data, t) for t in tags}
    series = {t: s for t, s in series.items() if s is not None}
    if not series:
        return
    all_ep = sorted(set().union(*[set(s[0].astype(int).tolist()) for s in series.values()]))
    maps = {t: dict(zip(s[0].astype(int).tolist(), s[1].tolist())) for t, s in series.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + list(series.keys()))
        for e in all_ep:
            w.writerow([e] + [maps[t].get(e, "") for t in series])


def main():
    plt = style()
    runs_data = {}
    for key, (path, label, _) in RUNS.items():
        if not path.exists():
            print("MISSING", key, path)
            continue
        print("load", key, label)
        data = load(path)
        runs_data[key] = data
        dump_epoch_csv(key, data, PUB / "csv" / f"{key}_epoch_wide.csv")
        print(f"  tags={len(data)}")

    short = ["official", "v2_base", "v2_center"]
    fm = ["v2_base", "v2_center"]
    xlim5 = (0.0, 4.0)

    # 1) Fair short-budget overview: Official + FM-base + FM+center
    #    Only shared metrics (official has no FM ADE etc.)
    save_aligned_grid(
        plt,
        runs_data,
        [
            (short, "train/loss_epoch", "train loss", "loss"),
            (short, "val/loss_epoch", "val loss", "loss"),
            (short, "train/dp_loss_epoch", "train DP loss", "loss"),
            (short, "val/dp_loss_epoch", "val DP loss", "loss"),
            (short, "train/bev_semantic_loss_epoch", "train BEV semantic", "loss"),
            (short, "val/bev_semantic_loss_epoch", "val BEV semantic", "loss"),
        ],
        PUB / "fig3_tb_overview.png",
        xlim=xlim5,
        ncols=3,
        title="Ch3 Phase1-v2 (5ep, aligned): Official DDPM vs FM-base vs FM+center",
    )

    # 2) FM-only metrics, still aligned 0..4
    save_aligned_grid(
        plt,
        runs_data,
        [
            (fm, "val/loss_epoch", "val loss", "loss"),
            (fm, "val/fm_x1_ade_epoch", "val FM ADE", "ADE (m)"),
            (fm, "val/fm_x1_fde_epoch", "val FM FDE", "FDE (m)"),
            (fm, "val/fm_zero_ade_epoch", "val FM-zero ADE", "ADE (m)"),
            (fm, "val/fm_zero_fde_epoch", "val FM-zero FDE", "FDE (m)"),
            (fm, "val/fm_zero_jerk_epoch", "val FM-zero jerk", "jerk"),
            (fm, "val/fm_zero_accel_epoch", "val FM-zero accel", "accel"),
            (["v2_center"], "train/fm_center_ade_epoch", "train center ADE", "ADE (m)"),
            (fm, "val/bev_semantic_loss_epoch", "val BEV semantic", "loss"),
        ],
        PUB / "fig3_tb_v2.png",
        xlim=xlim5,
        ncols=3,
        title="Ch3 Phase1-v2 FM metrics (5ep, aligned): base vs center",
    )

    # 3) Long-run P2 alone (do NOT mix with 5ep arms)
    save_aligned_grid(
        plt,
        runs_data,
        [
            (["p2_center18"], "val/loss_epoch", "val loss", "loss"),
            (["p2_center18"], "val/fm_x1_ade_epoch", "val FM ADE", "ADE (m)"),
            (["p2_center18"], "val/fm_x1_fde_epoch", "val FM FDE", "FDE (m)"),
            (["p2_center18"], "train/fm_center_ade_epoch", "train center ADE", "ADE (m)"),
            (["p2_center18"], "train/fm_center_loss_epoch", "train center loss", "loss"),
            (["p2_center18"], "val/bev_semantic_loss_epoch", "val BEV semantic", "loss"),
        ],
        PUB / "fig3_tb_p2_long.png",
        xlim=(0.0, 17.0),
        ncols=3,
        title="Ch3 P2-center long run (18ep); dashed = early-stop at ep4",
        mark_early_stop=4.0,
    )

    # 4) Short-budget v1 P1 vs clipped P2 (fair 0..4 only)
    save_aligned_grid(
        plt,
        runs_data,
        [
            (["p1_base", "p1_center", "p2_center18"], "val/loss_epoch", "val loss", "loss"),
            (["p1_base", "p1_center", "p2_center18"], "val/fm_x1_ade_epoch", "val FM ADE", "ADE (m)"),
            (["p1_center", "p2_center18"], "train/fm_center_ade_epoch", "train center ADE", "ADE (m)"),
            (["p1_center", "p2_center18"], "train/fm_center_loss_epoch", "train center loss", "loss"),
            (["p1_base", "p1_center", "p2_center18"], "val/bev_semantic_loss_epoch", "val BEV semantic", "loss"),
            (["p1_base", "p1_center", "p2_center18"], "val/fm_x1_fde_epoch", "val FM FDE", "FDE (m)"),
        ],
        PUB / "fig3_tb_p1p2_short.png",
        xlim=xlim5,
        ncols=3,
        title="Ch3 P1 vs P2 (first 5 epochs only, aligned)",
    )

    # 5) Ch4 same-iter — force shared x
    ch4_keys = [k for k in ("gtrs", "drivesuprim", "opd") if k in runs_data]
    if ch4_keys:
        # discover common max epoch among present series for first tag
        xmax = 0.0
        probe_tags = [
            "val/loss_epoch",
            "val/loss-ori_epoch",
            "val/loss-aug_epoch",
            "val/imi_loss-ori_epoch",
            "val/pdm_comfort_loss-ori_epoch",
        ]
        for k in ch4_keys:
            for t in probe_tags:
                ser = as_epoch_series(runs_data[k], t)
                if ser is not None and len(ser[0]):
                    xmax = max(xmax, float(ser[0].max()))
        xlim4 = (0.0, max(4.0, xmax))
        save_aligned_grid(
            plt,
            runs_data,
            [
                (ch4_keys, "val/loss_epoch", "val loss", "loss"),
                (ch4_keys, "val/loss-ori_epoch", "val loss-ori", "loss"),
                (ch4_keys, "val/loss-aug_epoch", "val loss-aug", "loss"),
                (ch4_keys, "val/loss-soft_epoch", "val loss-soft", "loss"),
                (ch4_keys, "val/imi_loss-ori_epoch", "val imi-ori", "loss"),
                (ch4_keys, "val/imi_loss-soft_epoch", "val imi-soft", "loss"),
                (ch4_keys, "val/loss-refinement_ori_epoch", "val refinement-ori", "loss"),
                (ch4_keys, "val/pdm_comfort_loss-ori_epoch", "val comfort-ori", "loss"),
                (["opd"], "val/loss_opd-opd_epoch", "val OPD distill", "loss"),
            ],
            PUB / "fig4_tb_overview.png",
            xlim=xlim4,
            ncols=3,
            title="Ch4 same-iter (aligned epoch axis): GTRS vs DriveSuprim vs OPD",
        )

    # copy main ones to ch5_figs root (overwrite old misaligned)
    import shutil

    mapping = {
        "fig3_tb_overview.png": OUT / "fig3_tb_overview.png",
        "fig3_tb_v2.png": OUT / "fig3_tb_v2.png",
        "fig4_tb_overview.png": OUT / "fig4_tb_overview.png",
        "fig3_tb_p2_long.png": OUT / "fig3_tb_p2_long.png",
        "fig3_tb_p1p2_short.png": OUT / "fig3_tb_p1p2_short.png",
    }
    for src_name, dst in mapping.items():
        src = PUB / src_name
        if src.exists():
            shutil.copy2(src, dst)
            pdf = src.with_suffix(".pdf")
            if pdf.exists():
                shutil.copy2(pdf, dst.with_suffix(".pdf"))
            print("copied →", dst)

    print("DONE →", PUB)


if __name__ == "__main__":
    main()
