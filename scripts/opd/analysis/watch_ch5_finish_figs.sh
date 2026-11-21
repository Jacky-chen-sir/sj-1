#!/usr/bin/env bash
# 等 score-cache 落盘后补画 fig5_4 / fig5_5(+FM)
set -euo pipefail
ROOT="${NAVSIM_DEVKIT_ROOT:-/home/ws/navsim_workspace/GTRS_official}"
EXP="${NAVSIM_EXP_ROOT:-/home/ws/navsim_workspace/exp}"
FIG="$EXP/ch5_figs"
VOCAB="$ROOT/traj_final/test_8192_kmeans.npy"
TEACH_NT="$FIG/cache_teacher_navtest30"
BASE="$FIG/cache_base_r34_2k"
OPD="$FIG/cache_opd_r34_2k"
LOG="$FIG/watch_ch5_finish.log"
exec > >(tee -a "$LOG") 2>&1

wait_n() {
  local dir="$1" n="$2" name="$3"
  echo "[wait] $name ≥$n in $dir"
  while true; do
    c=$(find "$dir" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l | tr -d ' ')
    c=${c:-0}
    echo "  $(date '+%H:%M:%S') $name=$c"
    if [ "$c" -ge "$n" ]; then break; fi
    sleep 60
  done
}

wait_n "$TEACH_NT" 20 teacher_navtest30

# fullchain + FM：取教师缓存与 FM 提案的交集
SCENE=$(python - <<'PY'
import pickle
from pathlib import Path
teach = Path("/home/ws/navsim_workspace/exp/ch5_figs/cache_teacher_navtest30")
with open("/home/ws/navsim_workspace/exp/ch3_df/p0_medoid/proposals.pkl","rb") as f:
    props = pickle.load(f)
for tok in props:
    if (teach / f"{tok}.pkl").exists():
        print(tok); break
PY
)
echo "fullchain scene=$SCENE"
if [[ -n "$SCENE" ]]; then
  python - <<PY
import pickle, numpy as np
from pathlib import Path
tok = "$SCENE"
with open("/home/ws/navsim_workspace/exp/ch3_df/p0_medoid/proposals.pkl","rb") as f:
    d = pickle.load(f)
arr = np.asarray(d[tok]["interpolated_proposal"], dtype=np.float32)
Path("$FIG/npy").mkdir(parents=True, exist_ok=True)
np.save("$FIG/npy/fm_scene_${SCENE}.npy", arr)
print(arr.shape)
PY
  python "$ROOT/scripts/opd/analysis/decision_pca.py" fullchain \
    --vocab "$VOCAB" --cache "$TEACH_NT" --scene "$SCENE" \
    --fm "$FIG/npy/fm_scene_${SCENE}.npy" --topk 256 \
    --out "$FIG/fig5_5_fullchain_pca.png" --plot
fi

wait_n "$BASE" 500 base_2k
wait_n "$OPD" 500 opd_2k

python "$ROOT/scripts/opd/analysis/decision_pca.py" scoring \
  --teacher /home/ws/navsim_workspace/dataset/traj_pdm_v2/opd/teacher/vit_l/8192_cache_ori \
  --student "base_r34=$BASE" \
  --student "opd_r34=$OPD" \
  --max-scenes 2000 \
  --out "$FIG/fig5_4_scoring_pca.png" \
  --csv "$FIG/fig5_4_scoring.csv" --plot

# 刷新 center 表（评测可能已更新）
if [[ -f "$EXP/ch3_df_v2/summary_phase1_v2.csv" ]]; then
  python "$ROOT/scripts/opd/analysis/plot_ch5_tables.py" center \
    --summary "$EXP/ch3_df_v2/summary_phase1_v2.csv" \
    --out "$FIG/fig5_1_center_consistency.png" || true
fi

echo "ALL FIGS DONE"
ls -lh "$FIG"/*.png
