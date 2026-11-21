#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""硕士论文第五章图表美化重绘（中文标注 + Noto CJK + Okabe–Ito + 400dpi）。

覆盖 results_pub 全部结果图、Pareto、词表 PCA、训练曲线总览、效率对比图，
并按图注把新图写回 docx 内嵌图片。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

EXP = Path("/home/ws/navsim_workspace/exp")
FIG = EXP / "ch5_figs"
OUT = FIG / "results_pub"
OUT.mkdir(parents=True, exist_ok=True)
DOC = Path("/home/ws/navsim_workspace/GTRS_official/硕士论文1112 修订.docx")

C = {
    "blk": "#222222",
    "org": "#E69F00",
    "sky": "#56B4E9",
    "grn": "#009E73",
    "ylw": "#F0E442",
    "blu": "#0072B2",
    "ver": "#D55E00",
    "red": "#CC79A7",
    "gry": "#8A8A8A",
    "lite": "#F7F7F7",
}


def style():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_name = "DejaVu Sans"
    for p in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
                # TTC: pick SC if possible
                prop = font_manager.FontProperties(fname=p)
                font_name = prop.get_name()
            except Exception:
                pass
            break
    # Prefer explicit SC family names known on this machine
    for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            path = font_manager.findfont(cand, fallback_to_default=False)
            if path and "DejaVu" not in path:
                font_name = cand
                break
        except Exception:
            continue

    mpl.rcParams.update(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name, "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#DDDDDD",
            "legend.fancybox": False,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt, font_name


def save(fig, stem: str, also: Optional[Path] = None) -> Path:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    if also is not None:
        also.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(also)
        fig.savefig(also.with_suffix(".pdf"))
    print(f"wrote {png}")
    return png


def annotate_bars(ax, bars, values, fmt="{:.1f}", dy=0.35, fontsize=8):
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + dy,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
            fontweight="medium",
        )


# ── locked numbers ─────────────────────────────────────────────
P0_FIRST = [74.1226, 74.2420, 74.1747]
P0_SUB = {
    "官方 DDPM": [97.12, 92.83, 86.94, 96.83, 83.09],
    "FM first": [97.70, 92.59, 86.35, 97.21, 85.63],
    "FM medoid": [97.92, 93.89, 86.47, 97.58, 88.48],
    "FM zero": [98.24, 94.39, 85.52, 97.96, 88.56],
}
P0_SUB_KEYS = ["NC", "DAC", "EP", "TTC", "舒适"]

P1 = {"FM-base\n(5ep)": 63.99, "FM+center\n(5ep)": 71.25}
P1_SUB = {
    "FM-base": [95.06, 86.18, 88.47, 94.48, 62.47],
    "FM+center": [97.25, 90.45, 87.61, 96.18, 87.65],
}

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

FM_EPOCH = {11: 70.51, 12: 70.81, 13: 72.51, 14: 71.88, 15: 72.66, 16: 71.56, 17: 72.44, 18: 73.80, 19: 72.13}

SAME_ITER = {
    "OPD（本文）": {5000: 78.46, 9000: 81.21, 13000: 82.28},
    "GTRS-R34": {5000: 73.84, 9000: 78.99, 13000: 80.88},
    "DriveSuprim": {5000: 71.24, 9000: 78.58, 13000: 80.69},
}
SAME_ITER_13K_SUB = {
    "OPD（本文）": [97.55, 95.78, 87.91, 96.76, 78.00, 9.54],
    "GTRS-R34": [97.63, 96.56, 82.95, 96.68, 71.21, 8.67],
    "DriveSuprim": [97.51, 95.35, 84.91, 96.42, 74.14, 9.89],
}
SUB13 = ["NC", "DAC", "EP", "TTC", "EC", "zero%↓"]

GATE_EPDMS = {5000: (78.38, 78.46), 9000: (79.49, 79.67), 13000: (82.37, 82.57)}
GATE_ZERO = {5000: (8.62, 8.22), 9000: (8.70, 8.05), 13000: (7.61, 7.16)}

# GPU0 efficiency
EFF = {
    "教师 ViT-L": {"params": 321.3, "flops": 812.0, "lat": 154.6, "mem": 2564},
    "学生 R34": {"params": 37.9, "flops": 364.2, "lat": 101.0, "mem": 477},
}


def fig_p0(plt):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    labels = ["官方\nDDPM", "FM\nfirst", "FM\nmedoid", "FM\nzero"]
    vals = [73.17, float(np.mean(P0_FIRST)), 75.82, 76.57]
    errs = [0.0, float(np.std(P0_FIRST, ddof=1)), 0.0, 0.0]
    colors = [C["gry"], C["sky"], C["org"], C["grn"]]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=errs, color=colors, width=0.68, capsize=3,
                  error_kw={"linewidth": 1.1, "ecolor": "#444"}, zorder=3,
                  edgecolor="white", linewidth=0.6)
    ax.axhline(73.05, color=C["blk"], ls="--", lw=1.0, alpha=0.55, label="文献官方 73.05")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("navtest 总分 (%)")
    ax.set_ylim(70.5, 78.2)
    ax.set_title("(a) 推断方式对照（同一 FM ep18）", pad=8)
    annotate_bars(ax, bars, vals, "{:.2f}", dy=0.12, fontsize=8)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    keys = P0_SUB_KEYS
    methods = list(P0_SUB.keys())
    x = np.arange(len(keys))
    w = 0.2
    cols = [C["gry"], C["sky"], C["org"], C["grn"]]
    for i, (m, col) in enumerate(zip(methods, cols)):
        ax.bar(x + (i - 1.5) * w, P0_SUB[m], width=w, label=m, color=col,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("子指标 (%)")
    ax.set_ylim(80, 100)
    ax.set_title("(b) 关键子指标", pad=8)
    ax.legend(ncol=2, loc="lower right", fontsize=7.5)

    fig.suptitle("生成器推断方式：官方扩散 vs 流匹配", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_ch3_p0_inference")
    plt.close(fig)


def fig_main(plt):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    items = [
        ("官方 DDPM\n(论文 ckpt)", 73.05, C["gry"]),
        ("FM-DP ep18\n(first, K=20)", 73.80, C["blu"]),
        ("FM-DP ep18\n(零噪声)", 76.57, C["grn"]),
        ("P1-v2 center\nep04 · K=20", 73.52, C["ver"]),
    ]
    labels = [t[0] for t in items]
    vals = [t[1] for t in items]
    colors = [t[2] for t in items]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.66, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(73.05, color=C["blk"], ls="--", lw=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("navtest 总分 (%)")
    ax.set_ylim(70, 78.5)
    ax.set_title("生成器主结果汇总", pad=10, fontweight="bold")
    annotate_bars(ax, bars, vals, "{:.2f}", dy=0.15)
    fig.tight_layout()
    save(fig, "fig_ch3_main_results")
    plt.close(fig)


def fig_center(plt):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15), gridspec_kw={"width_ratios": [0.9, 1.35]})
    ax = axes[0]
    labels = list(P1.keys())
    vals = list(P1.values())
    colors = [C["sky"], C["ver"]]
    bars = ax.bar(np.arange(2), vals, color=colors, width=0.58, edgecolor="white", zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("navtest 总分 (%) · 零噪声")
    ax.set_ylim(55, 78)
    ax.set_title("(a) 中心一致性（同预算 5 epoch）", pad=8)
    annotate_bars(ax, bars, vals, "{:.2f}", dy=0.55)
    ax.annotate(
        f"+{vals[1] - vals[0]:.2f}",
        xy=(1, vals[1]),
        xytext=(0.45, vals[1] + 3.2),
        fontsize=9,
        color=C["ver"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["ver"], lw=1.2),
    )

    ax = axes[1]
    keys = P0_SUB_KEYS
    x = np.arange(len(keys))
    w = 0.34
    ax.bar(x - w / 2, P1_SUB["FM-base"], width=w, color=C["sky"], label="FM-base", edgecolor="white", zorder=3)
    ax.bar(x + w / 2, P1_SUB["FM+center"], width=w, color=C["ver"], label="FM+center", edgecolor="white", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("子指标 (%)")
    ax.set_ylim(55, 102)
    ax.set_title("(b) 分项：舒适度贡献最大", pad=8)
    ax.legend(loc="lower right")

    fig.suptitle("中心一致性短训对照", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_ch3_center_ablation")
    plt.close(fig)


def fig_p1v2(plt):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.1), sharey=True)
    epochs = ["ep02", "ep03", "ep04"]
    x = np.arange(len(epochs))
    w = 0.34
    for ax, data, title in [
        (axes[0], P1V2_K20, "(a) K=20（论文口径）"),
        (axes[1], P1V2_K4, "(b) K=4（训练中心网格）"),
    ]:
        base = [data[e]["base"] for e in epochs]
        cen = [data[e]["center"] for e in epochs]
        ax.bar(x - w / 2, base, width=w, color=C["sky"], label="FM-base", edgecolor="white", zorder=3)
        ax.bar(x + w / 2, cen, width=w, color=C["ver"], label="FM+center", edgecolor="white", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["epoch 2", "epoch 3", "epoch 4"])
        ax.set_title(title, pad=8)
        ax.set_ylim(50, 78)
        if ax is axes[0]:
            ax.set_ylabel("navtest 总分 (%) · 零噪声")
        ax.legend(loc="lower left")
    fig.suptitle("Phase1-v2：base / center 随 checkpoint", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_ch3_p1v2_epochs")
    plt.close(fig)


def fig_fm_epoch(plt):
    fig, ax = plt.subplots(figsize=(6.0, 3.15))
    xs = sorted(FM_EPOCH)
    ys = [FM_EPOCH[e] for e in xs]
    ax.plot(xs, ys, "-o", color=C["blu"], lw=2.2, ms=7, label="FM-DP（first, K=20）", zorder=3)
    ax.axhline(73.05, color=C["gry"], ls="--", lw=1.2, label="官方 DDPM 73.05")
    ax.scatter([18], [73.80], s=70, zorder=5, color=C["ver"], edgecolors="white", linewidths=0.8, label="峰值 ep18 = 73.80")
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("navtest 总分 (%)")
    ax.set_ylim(69.5, 75.2)
    ax.set_xticks(xs)
    ax.set_title("FM-DP 训练曲线：逼近官方分数", pad=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_ch3_fm_epoch_curve")
    plt.close(fig)


def fig_sameiter(plt):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), gridspec_kw={"width_ratios": [1.1, 1.3]})
    ax = axes[0]
    steps = [5000, 9000, 13000]
    x = np.arange(len(steps))
    w = 0.24
    series = [("OPD（本文）", C["ver"]), ("GTRS-R34", C["grn"]), ("DriveSuprim", C["blu"])]
    for i, (name, col) in enumerate(series):
        ys = [SAME_ITER[name][s] for s in steps]
        ax.bar(x + (i - 1) * w, ys, width=w, color=col, label=name, edgecolor="white", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["5k", "9k", "13k"])
    ax.set_xlabel("优化步数")
    ax.set_ylabel("EPDMS (%)")
    ax.set_ylim(68, 86)
    ax.set_title("(a) 同迭代预算 EPDMS", pad=8)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    keys = SUB13
    x = np.arange(len(keys))
    w = 0.24
    for i, (name, col) in enumerate(series):
        ax.bar(x + (i - 1) * w, SAME_ITER_13K_SUB[name], width=w, color=col, label=name,
               edgecolor="white", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("指标 (%)")
    ax.set_ylim(0, 108)
    ax.set_title("(b) 13k 步关键子指标", pad=8)
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("评价器同迭代对照（ResNet34）", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_ch4_sameiter_epdms")
    plt.close(fig)


def fig_gate(plt):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.15))
    steps = [5000, 9000, 13000]
    x = np.arange(len(steps))
    w = 0.34

    ax = axes[0]
    off = [GATE_EPDMS[s][0] for s in steps]
    on = [GATE_EPDMS[s][1] for s in steps]
    ax.bar(x - w / 2, off, width=w, color=C["gry"], label="关闭 (0)", edgecolor="white", zorder=3)
    ax.bar(x + w / 2, on, width=w, color=C["org"], label="开启 (0.8)", edgecolor="white", zorder=3)
    for i in range(len(steps)):
        d = on[i] - off[i]
        ax.text(x[i], max(on[i], off[i]) + 0.28, f"+{d:.2f}", ha="center", fontsize=8, color=C["ver"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["5k", "9k", "13k"])
    ax.set_xlabel("优化步数")
    ax.set_ylabel("EPDMS (%)")
    ax.set_ylim(76.5, 84.5)
    ax.set_title("(a) EPDMS：安全门开关", pad=8)
    ax.legend(loc="lower right")

    ax = axes[1]
    offz = [GATE_ZERO[s][0] for s in steps]
    onz = [GATE_ZERO[s][1] for s in steps]
    ax.bar(x - w / 2, offz, width=w, color=C["gry"], label="关闭 (0)", edgecolor="white", zorder=3)
    ax.bar(x + w / 2, onz, width=w, color=C["org"], label="开启 (0.8)", edgecolor="white", zorder=3)
    for i in range(len(steps)):
        d = onz[i] - offz[i]
        ax.text(x[i], min(onz[i], offz[i]) - 0.35, f"{d:+.2f}", ha="center", fontsize=8, color=C["blu"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["5k", "9k", "13k"])
    ax.set_xlabel("优化步数")
    ax.set_ylabel("zero% ↓")
    ax.set_ylim(6.2, 9.4)
    ax.set_title("(b) 零分场景占比（越低越好）", pad=8)
    ax.legend(loc="upper right")

    fig.suptitle("安全门后处理消融（ratio = 0 / 0.8）", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_ch4_safety_gate")
    plt.close(fig)


def fig_overview(plt):
    fig = plt.figure(figsize=(8.0, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    vals = [73.17, float(np.mean(P0_FIRST)), 75.82, 76.57]
    bars = ax.bar(["DDPM", "first", "medoid", "zero"], vals, color=[C["gry"], C["sky"], C["org"], C["grn"]],
                  edgecolor="white", zorder=3)
    ax.set_ylim(70, 78)
    ax.set_ylabel("总分 (%)")
    ax.set_title("生成 · 推断方式")
    annotate_bars(ax, bars, vals, "{:.1f}", dy=0.12)

    ax = fig.add_subplot(gs[0, 1])
    vals = [63.99, 71.25]
    bars = ax.bar(["FM-base", "FM+center"], vals, color=[C["sky"], C["ver"]], edgecolor="white", zorder=3)
    ax.set_ylim(55, 78)
    ax.set_ylabel("总分 (%)")
    ax.set_title("生成 · 中心一致性 5ep")
    annotate_bars(ax, bars, vals, "{:.1f}", dy=0.4)

    ax = fig.add_subplot(gs[1, 0])
    vals = [82.28, 80.88, 80.69]
    bars = ax.bar(["OPD", "GTRS", "DriveSuprim"], vals, color=[C["ver"], C["grn"], C["blu"]], edgecolor="white", zorder=3)
    ax.set_ylim(78, 84)
    ax.set_ylabel("EPDMS (%)")
    ax.set_title("评价 · 同迭代 @13k")
    annotate_bars(ax, bars, vals, "{:.1f}", dy=0.08)

    ax = fig.add_subplot(gs[1, 1])
    vals = [82.37, 82.57]
    bars = ax.bar(["门关闭", "门 0.8"], vals, color=[C["gry"], C["org"]], edgecolor="white", zorder=3)
    ax.set_ylim(81.5, 83.3)
    ax.set_ylabel("EPDMS (%)")
    ax.set_title("评价 · 安全门 @13k")
    annotate_bars(ax, bars, vals, "{:.2f}", dy=0.02)

    fig.suptitle("第三、四章关键实验结果总览", fontsize=13, fontweight="bold", y=0.98)
    save(fig, "fig_results_overview")
    plt.close(fig)


def fig_pareto(plt):
    """精度—时延 Pareto（步数×8ms 标定）。"""
    # points: (K, score%, latency_ms, label, color, marker)
    pts = [
        (20, 76.57, 160, "FM zero · K=20", C["grn"], "o"),
        (20, 75.82, 160, "FM medoid · K=20", C["org"], "s"),
        (20, 74.18, 160, "FM first · K=20", C["sky"], "D"),
        (100, 73.17, 800, "DDPM first · K=100", C["ver"], "^"),
        (4, 72.66, 32, "center v2 · K=4", C["blu"], "P"),
        (20, 73.52, 160, "center v2 · K=20", C["blu"], "X"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.35))

    ax = axes[0]
    # K scan schematic for center v2
    ax.plot([4, 20], [72.66, 73.52], "-o", color=C["blu"], lw=2.0, ms=7, label="FM+center (v2)", zorder=3)
    ax.scatter([20], [76.57], s=80, color=C["grn"], zorder=4, label="FM zero (ep18)")
    ax.scatter([20], [75.82], s=70, color=C["org"], zorder=4, label="FM medoid")
    ax.scatter([20], [74.18], s=70, color=C["sky"], zorder=4, label="FM first")
    ax.scatter([100], [73.17], s=80, color=C["ver"], marker="^", zorder=4, label="官方 DDPM")
    ax.set_xlabel("推理步数 K")
    ax.set_ylabel("navtest 总分 (%)")
    ax.set_xlim(-5, 110)
    ax.set_ylim(71.5, 77.5)
    ax.set_title("(a) 步数扫描", pad=8)
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[1]
    ax.axvspan(0, 100, color=C["grn"], alpha=0.10, label="约 10 Hz 预算", zorder=0)
    for k, sc, lat, lab, col, mk in pts:
        ax.scatter([lat], [sc], s=90, color=col, marker=mk, zorder=4, edgecolors="white", linewidths=0.7)
        # smart offset
        ox, oy = 12, 0.12
        if "DDPM" in lab:
            ox, oy = -130, 0.15
        elif "K=4" in lab:
            ox, oy = 8, -0.35
        elif "medoid" in lab:
            ox, oy = 12, -0.28
        elif "first" in lab:
            ox, oy = 12, -0.45
        ax.annotate(lab, (lat, sc), xytext=(lat + ox, sc + oy), fontsize=7.5, color="#333")
    ax.set_xlabel("估计单次决策时延 (ms)  ·  约 8 ms/步")
    ax.set_ylabel("navtest 总分 (%)")
    ax.set_xlim(-20, 880)
    ax.set_ylim(71.5, 77.5)
    ax.set_title("(b) 精度—时延 Pareto（示意）", pad=8)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("生成器少步优势：精度与时延", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = save(fig, "fig5_k_pareto")
    # also write canonical path used by thesis
    fig.savefig(FIG / "_old_scatter" / "fig5_k_pareto.png")
    fig.savefig(FIG / "_old_scatter" / "fig5_k_pareto.pdf")
    plt.close(fig)
    return p


def fig_vocab_pca(plt):
    """Redraw vocab PCA if csv exists; else polish existing via restyle note."""
    csv_path = FIG / "_old_scatter" / "fig5_2_vocab_pca.csv"
    npy = EXP / "GTRS_official".replace("GTRS_official", "")  # noop
    # try common locations
    candidates = [
        FIG / "_old_scatter" / "fig5_2_vocab_pca.csv",
        FIG / "fig5_2_vocab_pca.csv",
        Path("/home/ws/navsim_workspace/GTRS_official/traj_final/test_8192_kmeans.npy"),
    ]
    vocab = Path("/home/ws/navsim_workspace/GTRS_official/traj_final/test_8192_kmeans.npy")
    if not vocab.exists():
        print("WARN: vocab npy missing, skip PCA redraw")
        return None

    from sklearn.decomposition import PCA

    X = np.load(vocab).astype(np.float64)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    # intent heuristic from endpoint / progress (same spirit as decision_pca)
    # Use final (x,y) if flattened 40*4 or similar
    # Fallback: cluster by endpoint angle of first 2 dims of last pose block
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    # Intention labels: reuse if csv exists
    labels = None
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "intent" in df.columns:
            labels = df["intent"].astype(str).tolist()
    if labels is None or len(labels) != len(Z):
        # simple geometric intents from PC / endpoint
        # assume traj [T,4] flattened
        T = 40
        D = X.shape[1] // T
        end = X.reshape(len(X), T, D)[:, -1, :2]
        ang = np.arctan2(end[:, 1], end[:, 0])
        spd = np.linalg.norm(end, axis=1)
        labels = []
        for a, s in zip(ang, spd):
            if s < np.percentile(spd, 22):
                labels.append("brake")
            elif a > 0.45:
                labels.append("left")
            elif a < -0.45:
                labels.append("right")
            else:
                labels.append("straight")

    intent_zh = {"straight": "直行", "left": "左转", "right": "右转", "brake": "制动"}
    intent_c = {"straight": C["sky"], "left": C["grn"], "right": C["org"], "brake": C["red"]}
    order = ["straight", "left", "right", "brake"]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for key in order:
        m = np.array([l == key for l in labels])
        if not m.any():
            continue
        ax.scatter(
            Z[m, 0], Z[m, 1],
            s=8, alpha=0.35, c=intent_c[key],
            label=f"{intent_zh[key]} (n={m.sum()})",
            rasterized=True, linewidths=0,
        )
    ax.set_xlabel(f"PC1（{pca.explained_variance_ratio_[0]*100:.1f}%）")
    ax.set_ylabel(f"PC2（{pca.explained_variance_ratio_[1]*100:.1f}%）")
    ax.set_title("候选词表 PCA 语义聚类（N=8192）", fontweight="bold", pad=10)
    leg = ax.legend(title="驾驶意图", loc="best", markerscale=2.2, fontsize=9, title_fontsize=9)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
        lh.set_sizes([40])
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    save(fig, "fig5_2_vocab_pca")
    fig.savefig(FIG / "_old_scatter" / "fig5_2_vocab_pca.png")
    fig.savefig(FIG / "_old_scatter" / "fig5_2_vocab_pca.pdf")
    plt.close(fig)
    print(f"PCA var={pca.explained_variance_ratio_[:2]}")
    return True


def fig_efficiency(plt):
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 3.0))
    names = list(EFF.keys())
    cols = [C["ver"], C["blu"]]
    metrics = [
        ("params", "参数量 (M)", "{:.1f}"),
        ("flops", "FLOPs (G)", "{:.1f}"),
        ("lat", "延迟 p50 (ms)", "{:.1f}"),
        ("mem", "峰值显存 (MB)", "{:.0f}"),
    ]
    for ax, (key, ylab, fmt) in zip(axes, metrics):
        vals = [EFF[n][key] for n in names]
        bars = ax.bar(names, vals, color=cols, width=0.62, edgecolor="white", zorder=3)
        ax.set_ylabel(ylab)
        ax.set_title(ylab.split()[0], fontsize=10)
        ymax = max(vals) * 1.18
        ax.set_ylim(0, ymax)
        annotate_bars(ax, bars, vals, fmt, dy=ymax * 0.02, fontsize=8)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("推理效率对比（GPU0 · RTX 3090 · batch=1）", fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "fig5_efficiency")
    plt.close(fig)


def redraw_tb_chinese(plt):
    """Redraw TB overview with Chinese titles using aligned loader."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "redraw_tb_thesis_aligned",
        Path(__file__).with_name("redraw_tb_thesis_aligned.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    load, RUNS = mod.load, mod.RUNS

    name_zh = {
        "official": "官方 DDPM",
        "v2_base": "FM-base",
        "v2_center": "FM+center",
    }
    series = {}
    for key in ("official", "v2_base", "v2_center"):
        path, _name, color = RUNS[key]
        try:
            series[key] = (load(Path(str(path))), name_zh[key], color)
        except Exception as e:
            print("TB load fail", key, e)
    if len(series) < 2:
        print("WARN: skip TB chinese redraw")
        return

    metrics = [
        ("train_loss_epoch", "训练损失"),
        ("val_loss_epoch", "验证损失"),
        ("train_dp_loss_epoch", "训练 DP 损失"),
        ("val_dp_loss_epoch", "验证 DP 损失"),
        ("train_bev_semantic_loss_epoch", "训练 BEV 语义"),
        ("val_bev_semantic_loss_epoch", "验证 BEV 语义"),
    ]

    def find_tag(d, want):
        if want in d:
            return want
        for k in d:
            if k.replace("/", "_").endswith(want) or want in k.replace("/", "_"):
                return k
        return None

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4), sharex=True)
    axes = axes.ravel()
    xmax = 0
    for ax, (tag, title) in zip(axes, metrics):
        for key, (data, name, color) in series.items():
            t = find_tag(data, tag)
            if t is None:
                continue
            xs, ys = data[t]
            ax.plot(xs, ys, "-o", color=color, label=name, ms=5, lw=1.9)
            if len(xs):
                xmax = max(xmax, float(np.max(xs)))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
    for ax in axes:
        ax.set_xlim(-0.2, max(4.2, xmax + 0.2))
        ax.set_xticks(range(0, int(max(4, xmax)) + 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Phase1-v2 训练曲线（epoch 轴对齐）", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = FIG / "fig3_tb_overview.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    save(fig, "fig3_tb_overview")
    plt.close(fig)
    print("wrote", out)


def replace_docx_images():
    """Replace embedded images whose following/prev caption matches known stems."""
    from docx import Document
    from docx.oxml.ns import qn

    if not DOC.exists():
        print("DOC missing")
        return

    # longest-prefix first so 图5-1 does not steal 图5-10/11
    mapping = [
        ("图5-12", OUT / "fig5_efficiency.png"),
        ("图5-11", FIG / "fig5_decision_bev_pv_plate.png"),
        ("图5-10", FIG / "fig5_decision_bev_pv.png"),
        ("图5-5c", OUT / "fig_ch3_main_results.png"),
        ("图5-5b", OUT / "fig_ch3_fm_epoch_curve.png"),
        ("图5-9", FIG / "_old_scatter" / "fig5_2_vocab_pca.png"),
        ("图5-8", OUT / "fig_results_overview.png"),
        ("图5-7", OUT / "fig_ch4_safety_gate.png"),
        ("图5-6", OUT / "fig_ch4_sameiter_epdms.png"),
        ("图5-5", FIG / "fig3_tb_overview.png"),
        ("图5-4", OUT / "fig_ch3_p1v2_epochs.png"),
        ("图5-3", OUT / "fig_ch3_center_ablation.png"),
        ("图5-2", FIG / "_old_scatter" / "fig5_k_pareto.png"),
        ("图5-1", OUT / "fig_ch3_p0_inference.png"),
    ]

    doc = Document(str(DOC))

    def para_text(p):
        return "".join(r.text for r in p.runs).strip()

    def match_img(t: str):
        for key, img in mapping:
            if key == "图5-5":
                if t.startswith("图5-5") and not t.startswith("图5-5b") and not t.startswith("图5-5c"):
                    return img
            elif t.startswith(key):
                return img
        return None

    replaced = 0
    paras = list(doc.paragraphs)
    for i, p in enumerate(paras):
        t = para_text(p)
        matched_img = match_img(t)
        if matched_img is None or not matched_img.exists():
            continue
        # image is typically the previous paragraph (possibly empty text with drawing)
        done = False
        for k in (i - 1, i - 2):
            if k < 0:
                continue
            cp = paras[k]
            blips = cp._element.xpath(".//*[local-name()='blip']")
            for blip in blips:
                rId = blip.get(qn("r:embed"))
                if not rId:
                    continue
                part = cp.part.related_parts.get(rId)
                if part is None:
                    continue
                part._blob = matched_img.read_bytes()
                replaced += 1
                done = True
                print(f"replaced image for {t[:48]} ← {matched_img.name}")
                break
            if done:
                break

    # Also insert efficiency figure after 表 5-2 caption if not present
    from docx.oxml import OxmlElement
    from docx.text.paragraph import Paragraph
    from docx.shared import Cm, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    has_eff = any(para_text(p).startswith("图5-12") for p in doc.paragraphs)
    if not has_eff and (OUT / "fig5_efficiency.png").exists():
        for p in doc.paragraphs:
            if para_text(p).startswith("表 5-2  推理效率") or para_text(p) == "表 5-2  推理效率对比":
                # find note paragraph after table is hard; insert after analysis para that mentions GPU0
                pass
        for p in doc.paragraphs:
            if "蒸馏把教师的安全偏好写入学生权重" in para_text(p):
                img_el = OxmlElement("w:p")
                p._p.addnext(img_el)
                img_p = Paragraph(img_el, p._parent)
                img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                img_p.add_run().add_picture(str(OUT / "fig5_efficiency.png"), width=Cm(14.5))
                cap_el = OxmlElement("w:p")
                img_el.addnext(cap_el)
                cap_p = Paragraph(cap_el, p._parent)
                cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = cap_p.add_run("图5-12  师生推理效率对比（参数量 / FLOPs / 延迟 / 显存）")
                r.bold = True
                r.font.size = Pt(10.5)
                print("inserted 图5-12 efficiency")
                break

    doc.save(str(DOC))
    print(f"docx saved, replaced≈{replaced}")


def main():
    plt, font = style()
    print("font:", font)
    fig_p0(plt)
    fig_main(plt)
    fig_center(plt)
    fig_p1v2(plt)
    fig_fm_epoch(plt)
    fig_sameiter(plt)
    fig_gate(plt)
    fig_overview(plt)
    fig_pareto(plt)
    fig_vocab_pca(plt)
    fig_efficiency(plt)
    redraw_tb_chinese(plt)
    replace_docx_images()
    print("DONE")


if __name__ == "__main__":
    main()
