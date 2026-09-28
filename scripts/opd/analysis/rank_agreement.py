#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""师生排序一致性分析（#8）。

蒸馏能不能起作用，取决于学生是否真的学到了教师的**排序**，而不只是学到了"选哪条"。
本脚本在**同一批 navtest 场景、同一个 8192 条词表**上，逐场景比较教师与学生的粗筛分数：

  Spearman ρ / Kendall τ   全词表排序相关性（分布形状是否被学到）
  Recall@K                 学生 top-K 是否覆盖教师 top-K（精排阶段的可达上界）
  Top-1 一致率              最终选出的轨迹是否相同（端到端行为一致性）
  Teacher-Top1@K           教师的第一名落在学生前 K 名里的比例（比集合召回更贴近实际）

输入是 `run_teacher_opd_cache.py` 落盘的 per-token pickle 目录（教师和学生用同一个脚本产出，
格式一致：`coarse` [V] fp32 + `topk_idx` [K] int32）。因此**不需要模型、不需要 GPU**，
在任意机器上都能重跑。

用法：

    python scripts/opd/analysis/rank_agreement.py \
        --teacher /exp/opd_cache/teacher_navtest \
        --student base_5k=/exp/opd_cache/base_step5000 \
        --student base_13k=/exp/opd_cache/base_step13000 \
        --student opd_5k=/exp/opd_cache/opd_step5000 \
        --student opd_13k=/exp/opd_cache/opd_step13000 \
        --out /exp/analysis/rank_agreement.csv \
        --max-scenes 3000 --workers 8
"""

import argparse
import csv
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_KS = (1, 8, 32, 256)
EPS = 1e-12


def _load_coarse(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return np.asarray(payload["coarse"], dtype=np.float64)
    except Exception:
        return None


def _ranks(x: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # 并列取平均
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > EPS else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_ranks(a), _ranks(b))


def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    """scipy 的 kendalltau 是 O(n log n)；全量失败（无 scipy / 内存）时退化到等距采样。"""
    try:
        from scipy.stats import kendalltau
    except ImportError:
        return float("nan")
    try:
        return float(kendalltau(a, b).statistic)
    except Exception:
        n = min(len(a), 4096)
        idx = np.linspace(0, len(a) - 1, n).astype(int)
        try:
            return float(kendalltau(a[idx], b[idx]).statistic)
        except Exception:
            return float("nan")


def _entropy(x: np.ndarray) -> float:
    z = x - x.max()
    p = np.exp(z)
    p /= p.sum()
    return float(-(p * np.log(p + EPS)).sum())


def _pair_metrics(args: Tuple[str, str, Tuple[int, ...]]) -> Optional[Dict]:
    teacher_path, student_path, ks = args
    t = _load_coarse(teacher_path)
    s = _load_coarse(student_path)
    if t is None or s is None or t.shape != s.shape:
        return None

    token = Path(student_path).stem
    row: Dict = {"token": token, "vocab_size": int(t.shape[0]),
                 "spearman": _spearman(t, s),
                 "kendall": _kendall(t, s),
                 "teacher_entropy": _entropy(t),
                 "student_entropy": _entropy(s)}

    t_order = np.argsort(-t, kind="mergesort")
    s_order = np.argsort(-s, kind="mergesort")
    row["top1_agree"] = float(t_order[0] == s_order[0])
    for k in ks:
        inter = np.intersect1d(t_order[:k], s_order[:k], assume_unique=False)
        row[f"recall@{k}"] = float(len(inter)) / k
    # 教师第一名落在学生前 K 名
    t_top1 = int(t_order[0])
    s_prefix = set()
    for k in ks:
        s_prefix.update(s_order[:k].tolist())
        row[f"teacher_top1_in_student@{k}"] = float(t_top1 in s_prefix)
    return row


def _collect_pairs(teacher_dir: str, student_dir: str, max_scenes: Optional[int]):
    t_dir, s_dir = Path(teacher_dir), Path(student_dir)
    if not t_dir.is_dir() or not s_dir.is_dir():
        raise SystemExit(f"目录不存在: {t_dir} / {s_dir}")
    tokens = sorted({p.stem for p in t_dir.glob("*.pkl")} & {p.stem for p in s_dir.glob("*.pkl")})
    if max_scenes and len(tokens) > max_scenes:
        step = len(tokens) / float(max_scenes)
        tokens = [tokens[int(i * step)] for i in range(max_scenes)]
    return [(str(t_dir / f"{t}.pkl"), str(s_dir / f"{t}.pkl")) for t in tokens]


def _summarize(rows: List[Dict], ks) -> Dict:
    out = {"n_scenes": len(rows)}
    for key in ["spearman", "kendall", "top1_agree", "teacher_entropy", "student_entropy"] + \
               [f"recall@{k}" for k in ks] + [f"teacher_top1_in_student@{k}" for k in ks]:
        vals = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])],
                        dtype=np.float64)
        out[key + "_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[key + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description="师生排序一致性分析")
    ap.add_argument("--teacher", required=True, help="教师 per-token pickle 目录")
    ap.add_argument("--student", action="append", required=True, metavar="NAME=DIR",
                    help="学生目录，可重复；NAME 用于结果表分组（如 opd_13k=/path）")
    ap.add_argument("--out", required=True, help="逐场景结果 CSV 路径")
    ap.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    ap.add_argument("--max-scenes", type=int, default=None, help="最多采样多少个场景（默认全量）")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    runs = []
    for spec in args.student:
        if "=" not in spec:
            raise SystemExit(f"--student 需要 NAME=DIR 形式，收到: {spec}")
        name, path = spec.split("=", 1)
        runs.append((name, path))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_rows, summary_rows = [], []

    for name, sdir in runs:
        pairs = _collect_pairs(args.teacher, sdir, args.max_scenes)
        if not pairs:
            print(f"[warn] {name}: 与教师目录没有交集 token，跳过", file=sys.stderr)
            continue
        print(f"[{name}] {len(pairs)} 个场景 ...", flush=True)
        tasks = [(tp, sp, ks) for tp, sp in pairs]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_pair_metrics, tasks, chunksize=16):
                if r is not None:
                    rows.append(r)
        if not rows:
            print(f"[warn] {name}: 全部场景读取失败", file=sys.stderr)
            continue

        summary = _summarize(rows, ks)
        summary["run"] = name
        summary["student_dir"] = sdir
        summary_rows.append(summary)
        for r in rows:
            r["run"] = name
            detail_rows.append(r)

        print(f"[{name}] ρ={summary['spearman_mean']:.4f}  τ={summary['kendall_mean']:.4f}  "
              f"top1={summary['top1_agree_mean']:.4f}  "
              + "  ".join(f"R@{k}={summary[f'recall@{k}_mean']:.4f}" for k in ks), flush=True)

    if not detail_rows:
        raise SystemExit("没有任何可用结果")

    detail_fields = ["run", "token", "vocab_size", "spearman", "kendall", "top1_agree",
                     "teacher_entropy", "student_entropy"] + \
                    [f"recall@{k}" for k in ks] + [f"teacher_top1_in_student@{k}" for k in ks]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for r in detail_rows:
            w.writerow({k: r.get(k, "") for k in detail_fields})

    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    sum_fields = ["run", "student_dir", "n_scenes"] + \
                 [f"{m}_{s}" for m in ["spearman", "kendall", "top1_agree", "teacher_entropy",
                                       "student_entropy"] + [f"recall@{k}" for k in ks] +
                  [f"teacher_top1_in_student@{k}" for k in ks] for s in ("mean", "std")]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction="ignore")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"\n逐场景结果: {out_path}\n汇总结果  : {summary_path}")


if __name__ == "__main__":
    main()
