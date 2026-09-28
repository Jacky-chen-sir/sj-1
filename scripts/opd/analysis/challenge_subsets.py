#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挑战子集分解（#9）。

只报一个 navtest 总分会掩盖改进发生在哪里。本脚本把每个方法的逐场景分数按场景属性分桶，
回答"提升是不是集中在困难场景"——这正是审稿人会追问的问题。

依赖两张表：
  · `per_token.csv`       —— 打分脚本产出（run_pdm_score_gpu_v2_aug.py 现在会写），每场景一行，
                             含最终 EPDMS 与 8 项子分数；
  · `scene_attributes.csv` —— run_scene_attributes.py 产出。

分桶维度（每个维度都独立成对，样本互不重叠）：

  is_intersection  路口 / 非路口（route 含 lane connector）
  has_vru          有行人或自行车 / 无
  is_lane_change   人类轨迹横向偏移 > 3.5 m（换道）/ 否则
  density          n_objects 三分位：稀疏 / 中等 / 密集
  speed            ego_speed 三分位：低速 / 中速 / 高速

用法：

    python scripts/opd/analysis/challenge_subsets.py \
        --attrs /exp/analysis/scene_attributes.csv \
        --run base_13k=/exp/base/per_token.csv \
        --run opd_13k=/exp/opd/per_token.csv \
        --out /exp/analysis/challenge_subsets.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCORE_COLS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
              "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
              "lane_keeping", "history_comfort"]

SHORT = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC",
         "driving_direction_compliance": "DDC", "traffic_light_compliance": "TL",
         "ego_progress": "EP", "time_to_collision_within_bound": "TTC",
         "lane_keeping": "LK", "history_comfort": "HC"}

SUMMARY_TOKENS = {"extended_pdm_score_combined", "extended_pdm_score_stage_one",
                  "extended_pdm_score_stage_two", "average_all_frames"}


def _read_per_token(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "token" not in df.columns:
        raise SystemExit(f"{path} 缺少 token 列——确认用的是新版打分脚本产出的 per_token.csv")
    df = df[~df["token"].isin(SUMMARY_TOKENS)]
    df = df[df["valid"].fillna(True).astype(bool)] if "valid" in df.columns else df
    if "score" not in df.columns:
        raise SystemExit(f"{path} 缺少 score 列")
    return df


def _tertile_labels(values: pd.Series, names: Tuple[str, str, str]) -> pd.Series:
    q1, q2 = values.quantile([1 / 3, 2 / 3])
    return pd.cut(values, bins=[-np.inf, q1, q2, np.inf], labels=list(names)).astype(object)


def _bucket_columns(attrs: pd.DataFrame) -> Dict[str, pd.Series]:
    """返回 {维度名: 该场景所属桶标签}，值为 NaN 的场景在对应维度上被排除。"""
    cols: Dict[str, pd.Series] = {}
    for key, names in (("is_intersection", ("非路口", "路口")),
                       ("has_vru", ("无 VRU", "有 VRU")),
                       ("is_lane_change", ("非换道", "换道"))):
        if key in attrs.columns:
            s = attrs[key]
            cols[key] = pd.Series(np.where(s.astype("boolean").fillna(False), names[1], names[0]),
                                  index=attrs.index)
    if "n_objects" in attrs.columns:
        cols["density"] = _tertile_labels(attrs["n_objects"], ("稀疏", "中等", "密集"))
    if "ego_speed" in attrs.columns:
        cols["speed"] = _tertile_labels(attrs["ego_speed"], ("低速", "中速", "高速"))
    return cols


def _aggregate(df: pd.DataFrame, dim: str, buckets: pd.Series, run: str) -> List[Dict]:
    rows = []
    present = [c for c in SCORE_COLS if c in df.columns]
    for bucket in pd.unique(buckets.dropna()):
        sub = df[buckets == bucket]
        if len(sub) == 0:
            continue
        row = {"run": run, "dimension": dim, "bucket": str(bucket), "n": int(len(sub)),
               "EPDMS": float(sub["score"].mean())}
        for c in present:
            row[SHORT[c]] = float(sub[c].mean())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="挑战子集分解")
    ap.add_argument("--attrs", required=True, help="scene_attributes.csv")
    ap.add_argument("--run", action="append", required=True, metavar="NAME=per_token.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    attrs = pd.read_csv(args.attrs)
    attrs = attrs.drop_duplicates(subset=["token"])
    bucket_cols = _bucket_columns(attrs)
    if not bucket_cols:
        raise SystemExit("scene_attributes.csv 里没有任何可用的分桶列")

    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run 需要 NAME=CSV 形式，收到: {spec}")
        name, path = spec.split("=", 1)
        runs.append((name, path))

    all_rows: List[Dict] = []
    for name, path in runs:
        df = _read_per_token(path)
        merged = df.merge(attrs, on="token", how="left", suffixes=("", "_attr"))
        n_missing = int(merged["is_intersection"].isna().sum()) if "is_intersection" in merged else 0
        if n_missing:
            print(f"[warn] {name}: {n_missing}/{len(merged)} 个场景在属性表里缺失，"
                  f"这些场景不参与分桶（但仍计入总表）", file=sys.stderr)

        all_rows.append({"run": name, "dimension": "ALL", "bucket": "全部", "n": int(len(merged)),
                         "EPDMS": float(merged["score"].mean()),
                         **{SHORT[c]: float(merged[c].mean()) for c in SCORE_COLS if c in merged}})
        for dim, labels in bucket_cols.items():
            all_rows.extend(_aggregate(merged, dim, labels.reindex(merged.index), name))

    out = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    pivot = out.pivot_table(index=["dimension", "bucket"], columns="run", values="EPDMS")
    print(f"\nEPDMS 分桶（行=桶，列=方法）\n{pivot.round(2).to_string()}\n")
    if len(runs) == 2:
        a, b = runs[0][0], runs[1][0]
        if a in pivot.columns and b in pivot.columns:
            delta = (pivot[b] - pivot[a]).dropna().sort_values(ascending=False)
            print(f"Δ = {b} − {a}（按提升排序）\n{delta.round(2).to_string()}\n")
    print(f"已写出: {out_path}")


if __name__ == "__main__":
    main()
