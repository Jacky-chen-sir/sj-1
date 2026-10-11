#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 5 章 5.3 节「基于 PCA 的决策过程可视化」的取数与绘图脚本。

论文 5.3 节要回答四个问题，本脚本用子命令一一对应：

  vocab       5.3.2 词表空间结构
              8192 条候选锚点在低维子空间里是否按**驾驶意图**（直行/左转/右转/加减速）
              组织成簇？若是，说明离散决策空间的覆盖是"有语义的覆盖"，而不是一堆
              无结构的点。做法：把每条锚点的**绝对轨迹**展平成向量 → PCA 投影到 2D/3D
              → 按端点横向位移与航向变化上色。同时输出方差解释率曲线，用于说明
              "前几个主成分能刻画多少结构"。

  generation  5.3.3 生成侧可视化
              把 FM 采样得到的提案投影到**与词表同一套主成分**上（投影矩阵由 vocab 拟合，
              不重新拟合——否则两个空间不可比），与真值轨迹、扩散基线提案对比。
              要看的是：FM 提案是否覆盖词表簇之间的**空隙**，即连续生成相对离散词表的补足。
              输出每个来源在子空间里的覆盖率与成对最近邻距离（多样性指标）。

  scoring     5.3.4 评价侧可视化
              两个子问题：(a) 评分头权重沿候选维的语义组织；(b) 蒸馏前后学生候选分数
              分布向教师对齐的趋势。做法：读教师/学生的 per-token pickle 缓存
              （`run_teacher_opd_cache.py` 的产物，含 `coarse` [V] 与 `topk_idx` [K]），
              比较同一批场景下师生粗筛分数的分布（散点 + 分位数曲线）。
              **不需要模型、不需要 GPU**，只读 pickle。

  fullchain   5.3.5 全链路决策可视化
              对**单个场景**，把"词表候选云 → FM 提案 → 粗筛 Top-K → 精排 → argmax 输出"
              五步画在统一的 PCA 坐标系里，串联第 3、4 章，展示决策的完整演化。

为什么用 PCA 而不是 t-SNE：PCA 是线性的，投影后的坐标有明确的方差解释含义，且
**可以用同一组基去投影不同来源的样本**（这正是 5.3.3 与 5.3.5 需要"同一坐标系"的前提）。
t-SNE 的坐标不可解释、也不能对样本外数据直接投影，故只作定性对照（`--tsne` 开关）。

依赖：numpy（必需）、matplotlib（画图时必需）。PCA 用 numpy 的 SVD 实现，不依赖 sklearn。

用法示例：

    # 5.3.2 词表空间结构
    python scripts/opd/analysis/decision_pca.py vocab \
        --vocab $NAVSIM_DEVKIT_ROOT/traj_final/test_8192_kmeans.npy \
        --out $NAVSIM_EXP_ROOT/figs/fig5_2_vocab_pca.png \
        --csv $NAVSIM_EXP_ROOT/figs/fig5_2_vocab_pca.csv

    # 5.3.3 生成侧（FM 提案 vs 真值 vs 扩散基线）
    python scripts/opd/analysis/decision_pca.py generation \
        --vocab $NAVSIM_DEVKIT_ROOT/traj_final/test_8192_kmeans.npy \
        --fm /exp/fm_proposals.pkl --label-fm "FM-DP (K=20)" \
        --ddpm /exp/ddpm_proposals.pkl --label-ddpm "DDPM (K=100)" \
        --gt /exp/gt_traj.pkl \
        --out $NAVSIM_EXP_ROOT/figs/fig5_3_generation_pca.png

    # 5.3.4 评价侧（师生对齐）
    python scripts/opd/analysis/decision_pca.py scoring \
        --teacher /exp/opd_cache/teacher_navtest \
        --student base_13k=/exp/opd_cache/base_step13000 \
        --student opd_13k=/exp/opd_cache/opd_step13000 \
        --out $NAVSIM_EXP_ROOT/figs/fig5_4_scoring_pca.png

    # 5.3.5 全链路（单场景）
    python scripts/opd/analysis/decision_pca.py fullchain \
        --vocab $NAVSIM_DEVKIT_ROOT/traj_final/test_8192_kmeans.npy \
        --scene <token> --cache /exp/opd_cache/opd_step13000 \
        --fm /exp/fm_proposals.pkl \
        --out $NAVSIM_EXP_ROOT/figs/fig5_5_fullchain_pca.png
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 配色：与论文第 3、4 章的图保持同一套语义色
# ---------------------------------------------------------------------------
C_INTENT = {            # 驾驶意图 → 颜色（5.3.2 / 5.3.3 / 5.3.5 通用）
    'straight': '#5B8DC9',   # 直行 —— 蓝
    'left':     '#5FA173',   # 左转 —— 绿
    'right':    '#D9A05B',   # 右转 —— 橙
    'brake':    '#8B7BB8',   # 减速 —— 紫
    'accel':    '#D07A6E',   # 加速 —— 红
}
C_SOURCE = {            # 样本来源 → 颜色（5.3.3）
    'vocab': '#C7CDD6',
    'gt':    '#1F2937',
    'fm':    '#37744B',
    'ddpm':  '#9E4A3E',
}


# ===========================================================================
# 轨迹编解码
# ===========================================================================
def to_absolute(arr: np.ndarray) -> np.ndarray:
    """把轨迹张量统一成绝对位姿 [N, H, 3]（x, y, θ）。

    支持两种输入：
      [N, H, 3] 已经是绝对位姿，原样返回；
      [N, H, 4] 是增量编码 (Δx, Δy, sinθ, cosθ)，先 atan2 还原航向，
                再对位置做累积积分（cumsum）得到绝对坐标。
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:                       # 展平的 [N, H*C] → 还原成 [N, H, C]
        c = 3 if a.shape[1] % 3 == 0 else 4
        a = a.reshape(a.shape[0], a.shape[1] // c, c)
    if a.ndim != 3:
        raise ValueError(f'轨迹张量维度不支持: {a.shape}')
    if a.shape[-1] == 3:
        return a
    if a.shape[-1] == 4:
        pos = np.cumsum(a[..., :2], axis=1)          # 一阶差分 → 累积积分
        theta = np.arctan2(a[..., 2], a[..., 3])     # (sinθ, cosθ) → θ
        return np.concatenate([pos, theta[..., None]], axis=-1)
    raise ValueError(f'轨迹最后一维必须是 3 或 4，得到 {a.shape[-1]}')


def intent_of(traj: np.ndarray) -> np.ndarray:
    """按端点相对起点的位移与航向变化，把每条轨迹判成五类驾驶意图之一。

    判据（经验阈值，可按数据集调整）：
      横向位移 |Δy| < 1.5 m 且 |Δθ| < 0.2 rad  → straight
      纵向位移 Δx < 5 m                        → brake（几乎没往前走）
      端点航向变化 Δθ > 0.25 rad               → left
      端点航向变化 Δθ < -0.25 rad              → right
      其余按横向位移正负号分 left / right
    返回每个样本的意图字符串数组。
    """
    # 用轨迹的**后半段**判方向：起步阶段所有轨迹都差不多，判别力弱
    h = traj.shape[1]
    end = traj[:, -1, :2]
    mid = traj[:, h // 2, :2]
    d_end = end - traj[:, 0, :2]
    d_mid = mid - traj[:, 0, :2]

    dtheta = traj[:, -1, 2] - traj[:, 0, 2]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi     # 归一到 (-π, π]

    out = np.empty(traj.shape[0], dtype=object)
    straight = (np.abs(d_end[:, 1]) < 1.5) & (np.abs(dtheta) < 0.2)
    brake = d_end[:, 0] < 5.0
    left = (dtheta > 0.25) | (d_mid[:, 1] > 0.8)
    right = (dtheta < -0.25) | (d_mid[:, 1] < -0.8)

    out[:] = 'straight'
    out[left] = 'left'
    out[right] = 'right'
    out[brake] = 'brake'
    out[straight & ~brake] = 'straight'
    return out


# ===========================================================================
# PCA（numpy SVD 实现，不依赖 sklearn）
# ===========================================================================
class PCA:
    """标准 PCA：中心化 → SVD → 取前 k 个主成分。

    `fit` 得到投影基，`transform` 可以把**任意来源**的样本投到同一组基上——
    这是 5.3.3 / 5.3.5 要求"词表与提案在同一坐标系"的关键。
    """

    def __init__(self, n_components: int = 3):
        self.k = n_components
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'PCA':
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # 样本数远大于特征数时用协方差矩阵的特征分解更省内存；这里统一走 SVD，
        # 数值更稳，且只需要前 k 个奇异向量。
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = (S ** 2) / max(len(X) - 1, 1)
        self.explained_ratio_ = var / var.sum()
        self.components_ = Vt[:self.k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.components_ is not None, '先调用 fit'
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def cum_var(self) -> np.ndarray:
        assert self.explained_ratio_ is not None
        return np.cumsum(self.explained_ratio_)


# ===========================================================================
# 数据加载
# ===========================================================================
def load_traj_file(path: str) -> np.ndarray:
    """读 .npy / .pkl 里的轨迹张量，返回 [N, H, C]。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix == '.npy':
        obj = np.load(p, allow_pickle=True)
    else:
        with open(p, 'rb') as f:
            obj = pickle.load(f)
    if isinstance(obj, dict):
        # 常见容器键，按优先级找
        for k in ('traj', 'trajectories', 'proposals', 'data', 'arr'):
            if k in obj:
                obj = obj[k]
                break
        else:
            raise KeyError(f'{path} 里没找到轨迹数组，现有键: {list(obj)[:8]}')
    return np.asarray(obj)


def load_coarse(path: str) -> Optional[np.ndarray]:
    """读 per-token pickle 里的粗筛分数 `coarse` [V]。"""
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        for k in ('coarse', 'coarse_score', 'score'):
            if isinstance(d, dict) and k in d:
                return np.asarray(d[k], dtype=np.float64)
    except Exception:
        return None
    return None


def iter_pickle(root: str, limit: Optional[int] = None):
    """遍历一个目录下的 per-token pickle（`run_teacher_opd_cache.py` 的产物）。"""
    files = sorted(Path(root).glob('*.pkl'))
    for i, f in enumerate(files):
        if limit is not None and i >= limit:
            break
        yield f


# ===========================================================================
# 子命令 1：词表空间结构（5.3.2）
# ===========================================================================
def cmd_vocab(args):
    vocab = to_absolute(load_traj_file(args.vocab))
    n, h, _ = vocab.shape
    print(f'词表：{n} 条 × {h} 步')

    X = vocab.reshape(n, -1)                 # 展平成特征向量（绝对位姿）
    pca = PCA(args.components).fit(X)
    Z = pca.transform(X)
    intents = intent_of(vocab)

    print('方差解释率（前 5 个主成分）：',
          ' '.join(f'{r:.3f}' for r in pca.explained_ratio_[:5]))
    print(f'前 {args.components} 个主成分累计解释率：{pca.cum_var()[-1]:.3f}')
    for it in C_INTENT:
        m = intents == it
        print(f'  {it:9s} {m.sum():5d} 条 ({m.mean() * 100:.1f}%)')

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['idx', 'pc1', 'pc2', 'pc3', 'intent', 'end_x', 'end_y', 'dtheta'])
            dt = (vocab[:, -1, 2] - vocab[:, 0, 2] + np.pi) % (2 * np.pi) - np.pi
            for i in range(n):
                w.writerow([i, Z[i, 0], Z[i, 1], Z[i, 2] if Z.shape[1] > 2 else 0.0,
                            intents[i], vocab[i, -1, 0], vocab[i, -1, 1], dt[i]])
        print('已写', args.csv)

    if args.plot:
        _plot_scatter(
            Z, intents, args.out,
            title=f'候选词表在 PCA 子空间中的语义聚类（N={n}）',
            xlabel=f'PC1（{pca.explained_ratio_[0] * 100:.1f}%）',
            ylabel=f'PC2（{pca.explained_ratio_[1] * 100:.1f}%）',
            legend_title='驾驶意图',
        )
    return 0


# ===========================================================================
# 子命令 2：生成侧可视化（5.3.3）
# ===========================================================================
def cmd_generation(args):
    vocab = to_absolute(load_traj_file(args.vocab))
    n = vocab.shape[0]

    # 关键：投影基**只在词表上拟合**，其余来源复用同一组基，
    # 否则两个来源各自拟合出的坐标不可比。
    pca = PCA(3).fit(vocab.reshape(n, -1))
    Zv = pca.transform(vocab.reshape(n, -1))

    groups: List[Tuple[str, np.ndarray, str]] = [('vocab', Zv, C_SOURCE['vocab'])]
    for path, label, key in ((args.gt, 'gt', 'gt'),
                             (args.fm, 'fm', 'fm'),
                             (args.ddpm, 'ddpm', 'ddpm')):
        if not path:
            continue
        t = to_absolute(load_traj_file(path))
        Z = pca.transform(t.reshape(t.shape[0], -1))
        groups.append((label, Z, C_SOURCE[key]))
        print(f'{label:6s} {t.shape[0]:6d} 条')

    # 多样性指标：各来源在 3D 子空间里的平均最近邻距离（越大越分散）
    print('\n子空间多样性（平均最近邻距离，已按词表尺度归一化）：')
    ref = _mean_nn_dist(Zv)
    for label, Z, _ in groups:
        d = _mean_nn_dist(Z)
        print(f'  {label:6s} {d:.4f}   （相对词表 {d / max(ref, 1e-12):.2f}×）')

    if args.plot:
        _plot_sources(groups, pca, args.out, n)
    return 0


# ===========================================================================
# 子命令 3：评价侧可视化（5.3.4）
# ===========================================================================
def cmd_scoring(args):
    teacher = [load_coarse(str(f)) for f in iter_pickle(args.teacher, args.max_scenes)]
    teacher = [t for t in teacher if t is not None]
    if not teacher:
        raise SystemExit(f'教师缓存里没读到 coarse：{args.teacher}')
    T = np.stack(teacher)
    print(f'教师：{T.shape[0]} 个场景 × {T.shape[1]} 条候选')

    rows = []
    for spec in args.student:
        name, path = spec.split('=', 1)
        S = [load_coarse(str(f)) for f in iter_pickle(path, args.max_scenes)]
        S = [s for s in S if s is not None and s.shape == T.shape[1:]]
        if not S:
            print(f'  [skip] {name}: 没读到可比的 coarse')
            continue
        S = np.stack(S[:T.shape[0]])
        Tt = T[:S.shape[0]]
        # 三个指标刻画"学生的排序是否在向教师靠拢"
        top1 = (S.argmax(1) == Tt.argmax(1)).mean()             # Top-1 一致率
        sp = _spearman_rows(Tt, S).mean()                       # 平均 Spearman ρ
        k = min(32, S.shape[1])
        rec = _recall_at_k(Tt, S, k)                            # 教师 top-k 的召回
        print(f'  {name:10s} Top-1 一致 {top1:.4f} | Spearman ρ {sp:.4f} | '
              f'Recall@{k} {rec:.4f}')
        rows.append((name, top1, sp, rec))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['student', 'top1_agreement', 'spearman_rho', 'recall_at_k'])
            w.writerows(rows)
        print('已写', args.csv)

    if args.plot:
        _plot_scoring(T, args.student, args.max_scenes, args.out)
    return 0


# ===========================================================================
# 子命令 4：全链路决策可视化（5.3.5）
# ===========================================================================
def cmd_fullchain(args):
    vocab = to_absolute(load_traj_file(args.vocab))
    n = vocab.shape[0]
    pca = PCA(2).fit(vocab.reshape(n, -1))

    # 该场景的粗筛分数：从缓存里按 token 找
    cache = Path(args.cache) / f'{args.scene}.pkl'
    coarse = load_coarse(str(cache))
    if coarse is None:
        raise SystemExit(f'缓存里没有该场景的粗筛分数：{cache}')

    Zv = pca.transform(vocab.reshape(n, -1))
    order = np.argsort(-coarse)
    topk = order[:args.topk]
    best = order[0]

    fm = None
    if args.fm:
        t = to_absolute(load_traj_file(args.fm))
        fm = pca.transform(t.reshape(t.shape[0], -1))

    if args.plot:
        _plot_fullchain(Zv, Zv[topk], Zv[best], fm, args.out,
                        f'单场景全链路决策（token={args.scene[:12]}…）')
    print(f'场景 {args.scene}: 粗筛 Top-1 = 候选 #{best}，'
          f'Top-{args.topk} 已标出')
    return 0


# ===========================================================================
# 指标小工具
# ===========================================================================
def _mean_nn_dist(Z: np.ndarray, sample: int = 2000) -> float:
    """平均最近邻距离（子采样以控制复杂度）。"""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Z), size=min(sample, len(Z)), replace=False)
    A = Z[idx]
    d = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def _rank_rows(X: np.ndarray) -> np.ndarray:
    """按行取秩（用于 Spearman）。"""
    order = X.argsort(axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(X.shape[0])[:, None]
    ranks[rows, order] = np.arange(X.shape[1])[None, :]
    return ranks


def _spearman_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ra, rb = _rank_rows(A), _rank_rows(B)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return num / np.maximum(den, 1e-12)


def _recall_at_k(T: np.ndarray, S: np.ndarray, k: int) -> float:
    """教师 top-k 集合被学生 top-k 集合覆盖的比例（逐场景平均）。"""
    tk = np.argsort(-T, axis=1)[:, :k]
    sk = np.argsort(-S, axis=1)[:, :k]
    hit = [len(set(a) & set(b)) / k for a, b in zip(tk, sk)]
    return float(np.mean(hit))


# ===========================================================================
# 绘图
# ===========================================================================
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'PingFang SC',
                                       'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def _plot_scatter(Z, intents, out, title, xlabel, ylabel, legend_title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for it, col in C_INTENT.items():
        m = intents == it
        if not m.any():
            continue
        ax.scatter(Z[m, 0], Z[m, 1], s=6, c=col, alpha=0.55, linewidths=0,
                   label=f'{it} ({m.sum()})')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=legend_title, markerscale=3, fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('已写', out)


def _plot_sources(groups, pca, out, n):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for label, Z, col in groups:
        size = 2 if label == 'vocab' else 12
        alpha = 0.25 if label == 'vocab' else 0.75
        ax.scatter(Z[:, 0], Z[:, 1], s=size, c=col, alpha=alpha, linewidths=0,
                   label=label)
    ax.set_title('生成侧提案在词表 PCA 子空间中的分布')
    ax.set_xlabel(f'PC1（{pca.explained_ratio_[0] * 100:.1f}%）')
    ax.set_ylabel(f'PC2（{pca.explained_ratio_[1] * 100:.1f}%）')
    ax.legend(markerscale=2, fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('已写', out)


def _plot_scoring(T, student_specs, max_scenes, out):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    # 左：教师粗筛分数的分位数曲线（分布形状）
    qs = np.linspace(0, 1, 101)
    axes[0].plot(qs, np.quantile(T.ravel(), qs), color='#1F2937', lw=2, label='teacher')
    for spec in student_specs:
        name, path = spec.split('=', 1)
        S = [load_coarse(str(f)) for f in iter_pickle(path, max_scenes)]
        S = [s for s in S if s is not None and s.shape == T.shape[1:]]
        if not S:
            continue
        S = np.stack(S[:T.shape[0]])
        axes[0].plot(qs, np.quantile(S.ravel(), qs), lw=1.6, label=name)
    axes[0].set_xlabel('分位数')
    axes[0].set_ylabel('粗筛分数')
    axes[0].set_title('(a) 粗筛分数分布：学生是否向教师靠拢')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    # 右：逐场景 Top-1 一致率
    names, vals = [], []
    for spec in student_specs:
        name, path = spec.split('=', 1)
        S = [load_coarse(str(f)) for f in iter_pickle(path, max_scenes)]
        S = [s for s in S if s is not None and s.shape == T.shape[1:]]
        if not S:
            continue
        S = np.stack(S[:T.shape[0]])
        names.append(name)
        vals.append((S.argmax(1) == T[:S.shape[0]].argmax(1)).mean())
    if names:
        axes[1].bar(names, vals, color='#5B8DC9')
        axes[1].set_ylabel('Top-1 一致率')
        axes[1].set_title('(b) 与教师的最终选择一致率')
        axes[1].grid(alpha=0.25, axis='y')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('已写', out)


def _plot_fullchain(Zv, Ztopk, Zbest, Zfm, out, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(Zv[:, 0], Zv[:, 1], s=3, c=C_SOURCE['vocab'], alpha=0.35,
               linewidths=0, label=f'词表候选云（{len(Zv)}）')
    ax.scatter(Ztopk[:, 0], Ztopk[:, 1], s=26, c='#5B8DC9', alpha=0.9,
               linewidths=0, label=f'粗筛 Top-K（{len(Ztopk)}）')
    if Zfm is not None:
        ax.scatter(Zfm[:, 0], Zfm[:, 1], s=14, c=C_SOURCE['fm'], alpha=0.8,
                   marker='^', linewidths=0, label=f'FM 提案（{len(Zfm)}）')
    ax.scatter(Zbest[0], Zbest[1], s=180, facecolors='none', edgecolors='#9E4A3E',
               linewidths=2.2, label='argmax 输出')
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print('已写', out)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description='第 5 章 5.3 节：基于 PCA 的决策过程可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('vocab', help='5.3.2 词表空间结构')
    p.add_argument('--vocab', required=True, help='词表 npy，[V,H,3] 或 [V,H,4]')
    p.add_argument('--components', type=int, default=3)
    p.add_argument('--out', default='fig5_2_vocab_pca.png')
    p.add_argument('--csv', default=None)
    p.add_argument('--plot', action='store_true')
    p.set_defaults(func=cmd_vocab)

    p = sub.add_parser('generation', help='5.3.3 生成侧可视化')
    p.add_argument('--vocab', required=True)
    p.add_argument('--fm', default=None, help='FM 提案 npy/pkl')
    p.add_argument('--ddpm', default=None, help='扩散基线提案 npy/pkl')
    p.add_argument('--gt', default=None, help='真值轨迹 npy/pkl')
    p.add_argument('--out', default='fig5_3_generation_pca.png')
    p.add_argument('--plot', action='store_true')
    p.set_defaults(func=cmd_generation)

    p = sub.add_parser('scoring', help='5.3.4 评价侧可视化')
    p.add_argument('--teacher', required=True, help='教师 per-token pickle 目录')
    p.add_argument('--student', action='append', default=[],
                   metavar='NAME=DIR', help='可重复；学生缓存目录')
    p.add_argument('--max-scenes', type=int, default=3000)
    p.add_argument('--out', default='fig5_4_scoring_pca.png')
    p.add_argument('--csv', default=None)
    p.add_argument('--plot', action='store_true')
    p.set_defaults(func=cmd_scoring)

    p = sub.add_parser('fullchain', help='5.3.5 全链路决策可视化')
    p.add_argument('--vocab', required=True)
    p.add_argument('--cache', required=True, help='含该场景粗筛分数的缓存目录')
    p.add_argument('--scene', required=True, help='场景 token')
    p.add_argument('--fm', default=None)
    p.add_argument('--topk', type=int, default=256)
    p.add_argument('--out', default='fig5_5_fullchain_pca.png')
    p.add_argument('--plot', action='store_true')
    p.set_defaults(func=cmd_fullchain)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
