#!/usr/bin/env python3
"""Fill a scores.csv row from a navtest eval CSV.
Usage: update_scores.py METHOD STEP CSV_PATH
"""
import sys
from pathlib import Path
import pandas as pd

method, step, csv_path = sys.argv[1], sys.argv[2], Path(sys.argv[3]).expanduser()
root = Path("/home/ws/navsim_workspace/exp/ablation_same_iter_r34")
if not csv_path.is_absolute():
    csv_path = (root / csv_path).resolve()
else:
    csv_path = csv_path.resolve()
scores = root / "scores.csv"
df_eval = pd.read_csv(csv_path)
v = df_eval[df_eval.valid == True] if "valid" in df_eval.columns else df_eval
cols = {
    "epdms": "score",
    "nc": "no_at_fault_collisions",
    "dac": "drivable_area_compliance",
    "ddc": "driving_direction_compliance",
    "tl": "traffic_light_compliance",
    "ep": "ego_progress",
    "ttc": "time_to_collision_within_bound",
    "lk": "lane_keeping",
    "hc": "history_comfort",
    "ec": "two_frame_extended_comfort",
}
present = {k: c for k, c in cols.items() if c in v.columns}
m = v[[c for c in present.values()]].mean() * 100
tbl = pd.read_csv(scores, dtype=str)
mask = (tbl["method"] == method) & (tbl["step"] == str(int(step)))
if not mask.any():
    row = {c: "" for c in tbl.columns}
    row.update({"method": method, "step": str(int(step)), "split": "same_iter",
                "csv": str(csv_path.relative_to(root))})
    tbl = pd.concat([tbl, pd.DataFrame([row])], ignore_index=True)
    mask = (tbl["method"] == method) & (tbl["step"] == str(int(step)))
tbl.loc[mask, "n_valid"] = str(int(len(v)))
tbl.loc[mask, "epdms"] = f"{m['score']:.2f}" if "epdms" in present else ""
for k, c in cols.items():
    if k == "epdms":
        continue
    tbl.loc[mask, k] = f"{m[c]:.2f}" if k in present else ""
tbl.loc[mask, "zero_pct"] = f"{(v['score']==0).mean()*100:.2f}" if "score" in v.columns else ""
tbl.loc[mask, "csv"] = str(csv_path.relative_to(root))
tbl.to_csv(scores, index=False)
missing = [k for k in cols if k not in present]
extra = f" (missing cols: {missing})" if missing else ""
print(f"updated scores.csv {method} step={step} EPDMS={m.get('score', float('nan')):.2f} n_valid={len(v)}{extra}")
