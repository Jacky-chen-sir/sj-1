#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-quality experimental RESULT figures for thesis Ch3 / Ch4 / Ch5.

CVPR-ish style: Okabe–Ito, no top/right spines, 400 dpi, PDF+PNG.
Only uses locked numbers from summaries / score CSVs (no live TB).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path("/home/ws/navsim_workspace/exp/ch5_figs/results_pub")
OUT.mkdir(parents=True, exist_ok=True)

# Okabe–Ito
C = {
    "blk": "#000000",
    "org": "#E69F00",
    "sky": "#56B4E9",
    "grn": "#009E73",
    "ylw": "#F0E442",
    "blu": "#0072B2",
    "ver": "#D55E00",
    "red": "#CC79A7",
    "gry": "#7F7F7F",
}


def style():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
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
        }
    )
    return plt


def save(fig, stem: str):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"wrote {png}")
    return png


def annotate_bars(ax, bars, values, fmt="{:.1f}", dy=0.4):
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + dy,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#222222",
        )


# ─── locked numbers (percent for scorer chapters; fraction*100 for DP) ───

# Ch3 P0 (fraction → report as %)
P0 = {
    "Official DDPM\n(first)": 73.17,
    "FM first\n(mean±std)": (74.18, 0.06),  # mean of 3, std~0.06 from 74.12/74.24/74.17
    "FM medoid": 75.82,
    "FM zero": 76.57,
}
P0_FIRST = [74.1226, 74.2420, 74.1747]

P0_SUB = {
    # NC, DAC, EP, TTC, comfort
    "Official": [97.12, 92.83, 86.94, 96.83, 83.09],
    "FM first": [97.70, 92.59, 86.35, 97.21, 85.63],
    "FM medoid": [97.92, 93.89, 86.47, 97.58, 88.48],
    "FM zero": [98.24, 94.39, 85.52, 97.96, 88.56],
}
P0_SUB_KEYS = ["NC", "DAC", "EP", "TTC", "Comfort"]

# Ch3 main generation (self-traj navtest %)
CH3_MAIN = [
    ("Official DDPM\n(paper ckpt)", 73.05, C["gry"]),
    ("FM-DP ep18\n(first, K=20)", 73.80, C["blu"]),
    ("FM-DP ep18\n(zero)", 76.57, C["grn"]),
    ("P1-v2 center\nep04 K=20", 73.52, C["ver"]),
]

# Ch3 P1 5ep zero (v1)
P1 = {"FM base\n(5ep, zero)": 63.99, "FM + center\n(5ep, zero)": 71.25}
P1_SUB = {
    "FM base": [95.06, 86.18, 88.47, 94.48, 62.47],
    "FM + center": [97.25, 90.45, 87.61, 96.18, 87.65],
}

# Ch3 P1-v2 best per arm at K=20
P1V2_K20 = {
    "ep02": {"base": 65.02, "center": 70.53},
    "ep03": {"base": 60.50, "center": 67.25},
    "ep04": {"base": 56.15, "center": 73.52},
}
P1V2_K4 = {
    "ep02": {"base": 56.73, "center": 66.45},
    "ep03": {"base": 60.54, "center": 68.39},
    "ep04": {"base": 64.24, "center": 72.66},
}

# FM joint navtest curve (first sampling)
FM_EPOCH = {
    11: 70.51,
    12: 70.81,
    13: 72.51,
    14: 71.88,
    15: 72.66,
    16: 71.56,
    17: 72.44,
    18: 73.80,
    19: 72.13,
}

# Ch4 same-iter EPDMS
SAME_ITER = {
    "OPD": {5000: 78.46, 9000: 81.21, 13000: 82.28},
    "GTRS-r34": {5000: 73.84, 9000: 78.99, 13000: 80.88},
    "DriveSuprim": {5000: 71.24, 9000: 78.58, 13000: 80.69},
}
SAME_ITER_13K_SUB = {
    # NC DAC EP TTC EC zero%
    "OPD": [97.55, 95.78, 87.91, 96.76, 78.00, 9.54],
    "GTRS-r34": [97.63, 96.56, 82.95, 96.68, 71.21, 8.67],
    "DriveSuprim": [97.51, 95.35, 84.91, 96.42, 74.14, 9.89],
}
SUB13 = ["NC", "DAC", "EP", "TTC", "EC", "zero%↓"]

# Safety gate (OPD 9cd57b3)
GATE = {
    5000: (78.38, 78.46),
    9000: (79.49, 79.67),
    13000: (82.37, 82.57),
}


def fig_ch3_p0_inference(plt):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.05, 1.35]})

    ax = axes[0]
    labels = ["Official\nDDPM", "FM first", "FM medoid", "FM zero"]
    vals = [73.17, float(np.mean(P0_FIRST)), 75.82, 76.57]
    errs = [0.0, float(np.std(P0_FIRST, ddof=1)), 0.0, 0.0]
    colors = [C["gry"], C["sky"], C["org"], C["grn"]]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=errs, color=colors, width=0.72, capsize=3, error_kw={"linewidth": 1})
    ax.axhline(73.05, color=C["blk"], ls="--", lw=0.9, alpha=0.7, label="Official paper 73.05")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("navtest score (%)")
    ax.set_ylim(70, 78.5)
    ax.set_title("(a) Inference mode (same FM ep18 ckpt)")
    annotate_bars(ax, bars, vals, "{:.2f}", dy=0.15)
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    keys = P0_SUB_KEYS
    methods = list(P0_SUB.keys())
    x = np.arange(len(keys))
    w = 0.2
    cols = [C["gry"], C["sky"], C["org"], C["grn"]]
    for i, (m, col) in enumerate(zip(methods, cols)):
        ax.bar(x + (i - 1.5) * w, P0_SUB[m], width=w, label=m, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("subscore (%)")
    ax.set_ylim(80, 100)
    ax.set_title("(b) Key subscores")
    ax.legend(ncol=2, loc="lower right", fontsize=6.5)

    fig.suptitle("Ch3 — Single-sample inference degradation", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "fig_ch3_p0_inference")
    plt.close(fig)


def fig_ch3_main_bars(plt):
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    labels = [t[0] for t in CH3_MAIN]
    vals = [t[1] for t in CH3_MAIN]
    colors = [t[2] for t in CH3_MAIN]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.7)
    ax.axhline(73.05, color=C["blk"], ls="--", lw=0.9, alpha=0.65)
    ax.text(len(labels) - 0.55, 73.2, "official 73.05", fontsize=7, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("navtest score (%)")
    ax.set_ylim(70, 78)
    ax.set_title("Ch3 — Generative planner main results")
    annotate_bars(ax, bars, vals, "{:.2f}", dy=0.12)
    fig.tight_layout()
    save(fig, "fig_ch3_main_results")
    plt.close(fig)


def fig_ch3_center_ablation(plt):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [0.85, 1.4]})

    ax = axes[0]
    labels = list(P1.keys())
    vals = list(P1.values())
    colors = [C["sky"], C["ver"]]
    bars = ax.bar(np.arange(2), vals, color=colors, width=0.65)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("navtest score (%), zero-noise")
    ax.set_ylim(55, 78)
    ax.set_title("(a) Center consistency (5ep)")
    annotate_bars(ax, bars, vals, "{:.2f}", dy=0.5)
    ax.annotate(
        f"+{vals[1] - vals[0]:.2f}",
        xy=(1, vals[1]),
        xytext=(0.55, vals[1] + 2.5),
        fontsize=8,
        color=C["ver"],
        arrowprops=dict(arrowstyle="->", color=C["ver"], lw=1),
    )

    ax = axes[1]
    keys = P0_SUB_KEYS
    x = np.arange(len(keys))
    w = 0.35
    b0 = ax.bar(x - w / 2, P1_SUB["FM base"], width=w, color=C["sky"], label="FM base")
    b1 = ax.bar(x + w / 2, P1_SUB["FM + center"], width=w, color=C["ver"], label="FM + center")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("subscore (%)")
    ax.set_ylim(55, 100)
    ax.set_title("(b) Subscores (comfort drives the gain)")
    ax.legend(loc="lower right")

    fig.suptitle("Ch3 — Center consistency training", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "fig_ch3_center_ablation")
    plt.close(fig)


def fig_ch3_p1v2_epochs(plt):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=True)
    epochs = ["ep02", "ep03", "ep04"]
    x = np.arange(len(epochs))
    w = 0.35

    for ax, data, title in [
        (axes[0], P1V2_K20, "(a) K=20 (paper)"),
        (axes[1], P1V2_K4, "(b) K=4 (train center grid)"),
    ]:
        base = [data[e]["base"] for e in epochs]
        cen = [data[e]["center"] for e in epochs]
        ax.bar(x - w / 2, base, width=w, color=C["sky"], label="base")
        ax.bar(x + w / 2, cen, width=w, color=C["ver"], label="center")
        ax.set_xticks(x)
        ax.set_xticklabels(["epoch 2", "epoch 3", "epoch 4"])
        ax.set_title(title)
        ax.set_ylim(50, 78)
        if ax is axes[0]:
            ax.set_ylabel("navtest score (%), zero")
        ax.legend(loc="lower left")

    fig.suptitle("Ch3 — Phase1-v2 base vs center across checkpoints", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "fig_ch3_p1v2_epochs")
    plt.close(fig)


def fig_ch3_fm_epoch_curve(plt):
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    xs = sorted(FM_EPOCH)
    ys = [FM_EPOCH[e] for e in xs]
    ax.plot(xs, ys, "-o", color=C["blu"], lw=1.8, ms=6, label="FM-DP (first, K=20)")
    ax.axhline(73.05, color=C["gry"], ls="--", lw=1, label="Official DDPM 73.05")
    ax.scatter([18], [73.80], s=55, zorder=5, color=C["ver"], label="peak ep18 = 73.80")
    ax.set_xlabel("epoch")
    ax.set_ylabel("navtest score (%)")
    ax.set_ylim(69.5, 75)
    ax.set_xticks(xs)
    ax.set_title("Ch3 — FM-DP training: epochs to match official score")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_ch3_fm_epoch_curve")
    plt.close(fig)


def fig_ch4_sameiter(plt):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.95), gridspec_kw={"width_ratios": [1.15, 1.25]})

    ax = axes[0]
    steps = [5000, 9000, 13000]
    x = np.arange(len(steps))
    w = 0.25
    series = [
        ("OPD", C["ver"]),
        ("GTRS-r34", C["grn"]),
        ("DriveSuprim", C["blu"]),
    ]
    for i, (name, col) in enumerate(series):
        ys = [SAME_ITER[name][s] for s in steps]
        ax.bar(x + (i - 1) * w, ys, width=w, color=col, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["5k", "9k", "13k"])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("EPDMS (%)")
    ax.set_ylim(68, 86)
    ax.set_title("(a) Same-iter navtest EPDMS")
    ax.legend(loc="lower right")

    ax = axes[1]
    keys = SUB13
    x = np.arange(len(keys))
    w = 0.25
    for i, (name, col) in enumerate(series):
        ax.bar(x + (i - 1) * w, SAME_ITER_13K_SUB[name], width=w, color=col, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("metric (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Step 13k subscores")
    ax.legend(loc="lower left", fontsize=7)

    fig.suptitle("Ch4 — Same-iteration comparison (R34)", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "fig_ch4_sameiter_epdms")
    plt.close(fig)


def fig_ch4_gate(plt):
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    steps = [5000, 9000, 13000]
    x = np.arange(len(steps))
    w = 0.35
    off = [GATE[s][0] for s in steps]
    on = [GATE[s][1] for s in steps]
    b0 = ax.bar(x - w / 2, off, width=w, color=C["gry"], label="gate off (0.0)")
    b1 = ax.bar(x + w / 2, on, width=w, color=C["org"], label="gate on (0.8)")
    for i, s in enumerate(steps):
        d = on[i] - off[i]
        ax.text(x[i], max(on[i], off[i]) + 0.25, f"+{d:.2f}", ha="center", fontsize=7.5, color=C["ver"])
    ax.set_xticks(x)
    ax.set_xticklabels(["5k", "9k", "13k"])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("EPDMS (%)")
    ax.set_ylim(76, 85)
    ax.set_title("Ch4/5 — OPD safety gate (ratio 0 vs 0.8)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_ch4_safety_gate")
    plt.close(fig)


def fig_overview_plate(plt):
    """One plate summarizing the key result bars for thesis overview."""
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    # P0
    ax = fig.add_subplot(gs[0, 0])
    labels = ["DDPM", "first", "medoid", "zero"]
    vals = [73.17, float(np.mean(P0_FIRST)), 75.82, 76.57]
    cols = [C["gry"], C["sky"], C["org"], C["grn"]]
    bars = ax.bar(labels, vals, color=cols)
    ax.set_ylim(70, 78)
    ax.set_ylabel("score (%)")
    ax.set_title("Ch3 P0: inference")
    annotate_bars(ax, bars, vals, "{:.1f}", dy=0.12)

    # center
    ax = fig.add_subplot(gs[0, 1])
    vals = [63.99, 71.25]
    bars = ax.bar(["FM base", "FM+center"], vals, color=[C["sky"], C["ver"]])
    ax.set_ylim(55, 78)
    ax.set_ylabel("score (%)")
    ax.set_title("Ch3 P1: center (5ep, zero)")
    annotate_bars(ax, bars, vals, "{:.1f}", dy=0.4)

    # same-iter final
    ax = fig.add_subplot(gs[1, 0])
    vals = [82.28, 80.88, 80.69]
    bars = ax.bar(["OPD", "GTRS", "DriveSuprim"], vals, color=[C["ver"], C["grn"], C["blu"]])
    ax.set_ylim(78, 84)
    ax.set_ylabel("EPDMS (%)")
    ax.set_title("Ch4: same-iter @13k")
    annotate_bars(ax, bars, vals, "{:.1f}", dy=0.08)

    # gate
    ax = fig.add_subplot(gs[1, 1])
    vals = [82.37, 82.57]
    bars = ax.bar(["gate 0", "gate 0.8"], vals, color=[C["gry"], C["org"]])
    ax.set_ylim(81.5, 83.2)
    ax.set_ylabel("EPDMS (%)")
    ax.set_title("Ch4/5: safety gate @13k")
    annotate_bars(ax, bars, vals, "{:.2f}", dy=0.02)

    fig.suptitle("Experimental results overview", fontsize=12, y=0.98)
    save(fig, "fig_results_overview")
    plt.close(fig)


def main():
    plt = style()
    fig_ch3_p0_inference(plt)
    fig_ch3_main_bars(plt)
    fig_ch3_center_ablation(plt)
    fig_ch3_p1v2_epochs(plt)
    fig_ch3_fm_epoch_curve(plt)
    fig_ch4_sameiter(plt)
    fig_ch4_gate(plt)
    fig_overview_plate(plt)

    # manifest
    files = sorted(p.name for p in OUT.glob("fig_*.png"))
    (OUT / "README.md").write_text(
        "# Publication experimental result figures\n\n"
        "Okabe–Ito, 400 dpi PNG + PDF.\n\n"
        + "\n".join(f"- `{f}`" for f in files)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nDone → {OUT}  ({len(files)} png)")


if __name__ == "__main__":
    main()
