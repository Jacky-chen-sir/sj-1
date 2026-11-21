#!/usr/bin/env bash
# 第 5 章插图一键出图（不训练）。依赖已有 proposals.pkl / 词表 / 教师缓存。
set -euo pipefail
ROOT="${NAVSIM_DEVKIT_ROOT:-/home/ws/navsim_workspace/GTRS_official}"
EXP="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
FIG="$EXP/ch5_figs"
VOCAB="$ROOT/traj_final/test_8192_kmeans.npy"
TEACH="${TEACHER_CACHE:-/home/ws/navsim_workspace/dataset/traj_pdm_v2/opd/teacher/vit_l/8192_cache_ori}"
PY="${PYTHON:-python}"
ANA="$ROOT/scripts/opd/analysis"
mkdir -p "$FIG/npy"

echo "[1/6] export FM / DDPM proposals → npy"
$PY "$ANA/export_proposals_npy.py" \
  --pkl "$EXP/ch3_df/p0_medoid/proposals.pkl" \
  --out "$FIG/npy/fm_medoid_interp.npy" \
  --max-scenes 200 --max-props-per-scene 20

$PY "$ANA/export_proposals_npy.py" \
  --pkl "$EXP/ch3_df/p0_official_first/proposals.pkl" \
  --out "$FIG/npy/ddpm_first_interp.npy" \
  --max-scenes 200 --max-props-per-scene 20

$PY "$ANA/export_proposals_npy.py" \
  --pkl "$EXP/ch3_df/p0_zero/proposals.pkl" \
  --out "$FIG/npy/fm_zero_interp.npy" \
  --max-scenes 500 --max-props-per-scene 1

# center-v2 最好点（若已有）
CENTER_PKL="$EXP/ch3_df_v2/navtest-p1_center-zero-k20-epoch02/proposals.pkl"
if [[ -f "$CENTER_PKL" ]]; then
  $PY "$ANA/export_proposals_npy.py" \
    --pkl "$CENTER_PKL" \
    --out "$FIG/npy/fm_center_v2_ep02_interp.npy" \
    --max-scenes 500 --max-props-per-scene 1
fi

echo "[2/6] vocab PCA (5.3.2)"
$PY "$ANA/decision_pca.py" vocab \
  --vocab "$VOCAB" \
  --out "$FIG/fig5_2_vocab_pca.png" \
  --csv "$FIG/fig5_2_vocab_pca.csv" --plot

$PY "$ANA/plot_ch5_tables.py" varcurve \
  --vocab "$VOCAB" --out "$FIG/fig5_2_vocab_varcurve.png"

echo "[3/6] generation PCA (5.3.3)"
$PY "$ANA/decision_pca.py" generation \
  --vocab "$VOCAB" \
  --fm "$FIG/npy/fm_medoid_interp.npy" \
  --ddpm "$FIG/npy/ddpm_first_interp.npy" \
  --out "$FIG/fig5_3_generation_pca.png" --plot

echo "[4/6] Phase0 + center tables"
$PY "$ANA/plot_ch5_tables.py" phase0 \
  --summary "$EXP/ch3_df/summary_phase0.csv" \
  --out "$FIG/fig5_0_inference_modes.png"

if [[ -f "$EXP/ch3_df_v2/summary_phase1_v2.csv" ]]; then
  $PY "$ANA/plot_ch5_tables.py" center \
    --summary "$EXP/ch3_df_v2/summary_phase1_v2.csv" \
    --out "$FIG/fig5_1_center_consistency.png" || true
  $PY "$ANA/plot_ch5_tables.py" kscan \
    --summary "$EXP/ch3_df_v2/summary_phase1_v2.csv" \
    --arm center --out "$FIG/fig5_1_kscan_center.png" || true
  $PY "$ANA/plot_ch5_tables.py" kscan \
    --summary "$EXP/ch3_df_v2/summary_phase1_v2.csv" \
    --arm base --out "$FIG/fig5_1_kscan_base.png" || true
fi

echo "[5/6] fullchain (5.3.5) — pick a scene in both teacher cache & FM dump"
SCENE=$(
  $PY - <<'PY'
import pickle
from pathlib import Path
fm = Path("/home/ws/navsim_workspace/exp/ch3_df/p0_medoid/proposals.pkl")
teach = Path("/home/ws/navsim_workspace/dataset/traj_pdm_v2/opd/teacher/vit_l/8192_cache_ori")
with open(fm, "rb") as f:
    props = pickle.load(f)
for tok in props:
    if (teach / f"{tok}.pkl").exists():
        print(tok)
        break
PY
)
echo "  scene=$SCENE"
if [[ -n "$SCENE" ]]; then
  $PY "$ANA/export_proposals_npy.py" \
    --pkl "$EXP/ch3_df/p0_medoid/proposals.pkl" \
    --out "$FIG/npy/fm_one_scene_${SCENE}.npy" \
    --max-scenes 1 --max-props-per-scene 100
  # export only that scene: rewrite small helper inline
  $PY - <<PY
import pickle, numpy as np
from pathlib import Path
tok = "$SCENE"
with open("$EXP/ch3_df/p0_medoid/proposals.pkl","rb") as f:
    d = pickle.load(f)
arr = np.asarray(d[tok]["interpolated_proposal"], dtype=np.float32)
Path("$FIG/npy/fm_scene_${SCENE}.npy").parent.mkdir(parents=True, exist_ok=True)
np.save("$FIG/npy/fm_scene_${SCENE}.npy", arr)
print("scene props", arr.shape)
PY
  $PY "$ANA/decision_pca.py" fullchain \
    --vocab "$VOCAB" \
    --cache "$TEACH" \
    --scene "$SCENE" \
    --fm "$FIG/npy/fm_scene_${SCENE}.npy" \
    --topk 256 \
    --out "$FIG/fig5_5_fullchain_pca.png" --plot
fi

echo "[6/6] scoring (5.3.4) — teacher only quantile if no student cache"
# 若存在学生缓存目录则画对照；否则跳过并提示
STUDENT_BASE="${STUDENT_BASE_CACHE:-}"
STUDENT_OPD="${STUDENT_OPD_CACHE:-}"
if [[ -n "$STUDENT_BASE" && -n "$STUDENT_OPD" && -d "$STUDENT_BASE" && -d "$STUDENT_OPD" ]]; then
  $PY "$ANA/decision_pca.py" scoring \
    --teacher "$TEACH" \
    --student "base_13k=$STUDENT_BASE" \
    --student "opd_13k=$STUDENT_OPD" \
    --max-scenes 2000 \
    --out "$FIG/fig5_4_scoring_pca.png" \
    --csv "$FIG/fig5_4_scoring.csv" --plot
else
  echo "  [skip] 未设置 STUDENT_BASE_CACHE / STUDENT_OPD_CACHE，跳过 5.3.4"
fi

echo "DONE → $FIG"
ls -lh "$FIG"/*.png 2>/dev/null || true
