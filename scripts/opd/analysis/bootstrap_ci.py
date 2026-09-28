#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ΔEPDMS 的 bootstrap 置信区间与显著性（#10）。

单种子 + 单点数字在顶会上说服力不够。本脚本把两个方法的逐场景分数按 token 对齐，
用 bootstrap 给出 ΔEPDMS 的 95% 置信区间、获胜场景比例与双侧 p 值。

**默认按 log 做 cluster bootstrap，而不是按场景独立重采样。** navtest 的场景来自有限条
log，同一条 log 内相邻帧高度相关（同一段路、同一批交通参与者），按场景独立重采样会把
有效样本量高估好几倍、把置信区间压得过窄。cluster bootstrap 是这里统计上正确的做法，
也是审稿人会追问的点；`--cluster none` 可以退回按场景重采样做对照。

用法：

    python scripts/opd/analysis/bootstrap_ci.py \
        --baseline /exp/base/per_token.csv \
        --ours     /exp/opd/per_token.csv \
        --attrs    /exp/analysis/scene_attributes.csv \
        --n-boot 10000 --out /exp/analysis/bootstrap_ci.csv

注意：bootstrap 衡量的是**测试集采样**带来的不确定性，不包含训练随机性。多种子方差要
分别跑每个种子的 per_token.csv，再对 Δ 的种子间标准差单独报告——两者不要混为一谈。
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCORE_COLS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
              "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
              "lane_keeping", "history_comfort"]
SUMMARY_TOKENS = {"extended_pdm_score_combined", "extended_pdm_score_stage_one",
                  "extended_pdm_score_stage_two", "average_all_frames"}


def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "token" not in df.columns or "score" not in df.columns:
        raise SystemExit(f"{path} 需要同时有 token 与 score 列（用新版打分脚本产出的 per_token.csv）")
    df = df[~df["token"].isin(SUMMARY_TOKENS)]
    if "valid" in df.columns:
        df = df[df["valid"].fillna(True).astype(bool)]
    return df


def _boot_scene(base: np.ndarray, ours: np.ndarray, n_boot: int, rng, chunk: int = 200) -> np.ndarray:
    n = len(base)
    out = np.empty(n_boot, dtype=np.float64)
    i = 0
    while i < n_boot:
        m = min(chunk, n_boot - i)
        idx = rng.integers(0, n, size=(m, n))
        out[i:i + m] = ours[idx].mean(axis=1) - base[idx].mean(axis=1)
        i += m
    return out


def _boot_cluster(base: np.ndarray, ours: np.ndarray, cluster: np.ndarray,
                  n_boot: int, rng, chunk: int = 500) -> np.ndarray:
    uniq, inv = np.unique(cluster, return_inverse=True)
    g = len(uniq)
    sum_b = np.bincount(inv, weights=base, minlength=g)
    sum_o = np.bincount(inv, weights=ours, minlength=g)
    cnt = np.bincount(inv, minlength=g).astype(np.float64)
    out = np.empty(n_boot, dtype=np.float64)
    i = 0
    while i < n_boot:
        m = min(chunk, n_boot - i)
        pick = rng.integers(0, g, size=(m, g))
        c = cnt[pick].sum(axis=1)
        out[i:i + m] = sum_o[pick].sum(axis=1) / c - sum_b[pick].sum(axis=1) / c
        i += m
    return out


def _sign_test_p(diff: np.ndarray) -> float:
    """双侧符号检验（正态近似，含连续性校正）。平局场景不计入样本量。"""
    d = diff[np.isfinite(diff)]
    pos = int((d > 0).sum())
    m = pos + int((d < 0).sum())
    if m == 0:
        return float("nan")
    z = max(abs(pos - m / 2) - 0.5, 0.0) / math.sqrt(m / 4.0)
    return float(math.erfc(z / math.sqrt(2.0)))


def _analyze(base: np.ndarray, ours: np.ndarray, cluster: np.ndarray, n_boot: int, rng,
             use_cluster: bool) -> Dict[str, float]:
    point = float(ours.mean() - base.mean())
    boot = _boot_cluster(base, ours, cluster, n_boot, rng) if use_cluster \
        else _boot_scene(base, ours, n_boot, rng)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_boot = 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    diff = ours - base
    return {
        "n_scenes": int(len(base)),
        "n_clusters": int(len(np.unique(cluster))) if use_cluster else int(len(base)),
        "mean_baseline": float(base.mean()),
        "mean_ours": float(ours.mean()),
        "delta": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "boot_p": float(min(p_boot, 1.0)),
        "sign_test_p": _sign_test_p(diff),
        "win_rate": float((diff > 0).mean()),
        "tie_rate": float((diff == 0).mean()),
        "delta_std_over_scenes": float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan"),
    }


def _bucket_masks(attrs: pd.DataFrame, tokens: pd.Index) -> Dict[Tuple[str, str], np.ndarray]:
    """返回 {(维度, 桶): 布尔掩码}。属性缺失的场景**不落入任何桶**，避免被静默算成负类。"""
    a = attrs.drop_duplicates(subset=["token"]).set_index("token").reindex(tokens)
    masks: Dict[Tuple[str, str], np.ndarray] = {}
    for key, names in (("is_intersection", ("非路口", "路口")),
                       ("has_vru", ("无 VRU", "有 VRU")),
                       ("is_lane_change", ("非换道", "换道"))):
        if key not in a.columns:
            continue
        s = a[key].astype("boolean")
        known = s.notna().to_numpy()
        v = s.fillna(False).to_numpy(dtype=bool)
        masks[(key, names[0])] = known & ~v
        masks[(key, names[1])] = known & v
    for key, names in (("n_objects", ("稀疏", "中等", "密集")),
                       ("ego_speed", ("低速", "中速", "高速"))):
        if key not in a.columns:
            continue
        vals = pd.to_numeric(a[key], errors="coerce")
        known = vals.notna().to_numpy()
        q1, q2 = vals.quantile([1 / 3, 2 / 3])
        arr = vals.to_numpy(dtype=float)
        masks[(key, names[0])] = known & (arr <= q1)
        masks[(key, names[1])] = known & (arr > q1) & (arr <= q2)
        masks[(key, names[2])] = known & (arr > q2)
    return masks


def main():
    ap = argparse.ArgumentParser(description="ΔEPDMS bootstrap 置信区间")
    ap.add_argument("--baseline", required=True, help="基线 per_token.csv")
    ap.add_argument("--ours", required=True, help="本方法 per_token.csv")
    ap.add_argument("--attrs", default=None, help="scene_attributes.csv（给了就额外出分桶 CI）")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cluster", default="log_name", choices=["log_name", "none"],
                    help="重采样单元：log_name（默认，推荐）或 none（按场景独立）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base_df, ours_df = _read(args.baseline), _read(args.ours)

    merged = base_df.merge(ours_df, on="token", suffixes=("_base", "_ours"))
    if merged.empty:
        raise SystemExit("两个文件没有共同的 token")
    n_drop_b = len(base_df) - len(merged)
    n_drop_o = len(ours_df) - len(merged)
    if n_drop_b or n_drop_o:
        print(f"[info] 交集 {len(merged)} 个场景（基线独有 {n_drop_b}，本方法独有 {n_drop_o}，已丢弃）",
              file=sys.stderr)

    use_cluster = args.cluster == "log_name"
    if use_cluster and "log_name_base" not in merged.columns and "log_name" not in merged.columns:
        print("[warn] per_token.csv 里没有 log_name，退回按场景重采样", file=sys.stderr)
        use_cluster = False
    cluster = (merged["log_name_base"].to_numpy() if "log_name_base" in merged.columns
               else merged.get("log_name", pd.Series(["all"] * len(merged))).to_numpy())

    base = merged["score_base"].to_numpy(dtype=np.float64)
    ours = merged["score_ours"].to_numpy(dtype=np.float64)
    tokens = merged["token"]

    rows: List[Dict] = []
    overall = _analyze(base, ours, cluster, args.n_boot, rng, use_cluster)
    overall.update({"dimension": "ALL", "bucket": "全部", "metric": "EPDMS"})
    rows.append(overall)

    for col in SCORE_COLS:
        cb, co = f"{col}_base", f"{col}_ours"
        if cb in merged.columns and co in merged.columns:
            r = _analyze(merged[cb].to_numpy(dtype=np.float64), merged[co].to_numpy(dtype=np.float64),
                         cluster, args.n_boot, rng, use_cluster)
            r.update({"dimension": "ALL", "bucket": "全部", "metric": col})
            rows.append(r)

    if args.attrs:
        attrs = pd.read_csv(args.attrs)
        for (dim, bucket), mask in _bucket_masks(attrs, tokens).items():
            if mask.sum() < 30:
                continue
            r = _analyze(base[mask], ours[mask], cluster[mask] if use_cluster else None,
                         args.n_boot, rng, use_cluster)
            r.update({"dimension": dim, "bucket": bucket, "metric": "EPDMS"})
            rows.append(r)

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    mode = f"cluster(log_name, {overall['n_clusters']} 条 log)" if use_cluster else "按场景独立"
    print(f"\n重采样方式: {mode}    B={args.n_boot}\n")
    print(f"  基线 EPDMS      : {overall['mean_baseline']:.4f}")
    print(f"  本方法 EPDMS    : {overall['mean_ours']:.4f}")
    print(f"  ΔEPDMS          : {overall['delta']:+.4f}   "
          f"95% CI [{overall['ci_lo']:+.4f}, {overall['ci_hi']:+.4f}]")
    print(f"  bootstrap p     : {overall['boot_p']:.4g}")
    print(f"  符号检验 p      : {overall['sign_test_p']:.4g}")
    print(f"  获胜场景比例    : {overall['win_rate'] * 100:.1f}%  "
          f"(平局 {overall['tie_rate'] * 100:.1f}%)")

    if args.attrs:
        sub = out[(out["dimension"] != "ALL") & (out["metric"] == "EPDMS")].copy()
        sub = sub.sort_values("delta", ascending=False)
        print("\n分桶 ΔEPDMS（按提升排序）")
        for _, r in sub.iterrows():
            sig = "*" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else " "
            print(f"  {r['dimension']:<18}{r['bucket']:<8}n={int(r['n_scenes']):<6}"
                  f"Δ={r['delta']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]{sig}")

    print(f"\n已写出: {out_path}")


if __name__ == "__main__":
    main()
