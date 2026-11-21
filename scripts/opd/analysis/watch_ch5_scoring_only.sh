#!/usr/bin/env bash
# 只等学生缓存 → 画 fig5_4 评价侧决策可视化
set -euo pipefail
ROOT=/home/ws/navsim_workspace/GTRS_official
FIG=/home/ws/navsim_workspace/exp/ch5_figs
BASE=$FIG/cache_base_r34_2k
OPD=$FIG/cache_opd_r34_2k
TEACH=/home/ws/navsim_workspace/dataset/traj_pdm_v2/opd/teacher/vit_l/8192_cache_ori
LOG=$FIG/watch_scoring.log
exec >"$LOG" 2>&1

count_pkl() { find "$1" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l | awk '{print $1+0}'; }

echo "[start] wait scoring caches"
while true; do
  b=$(count_pkl "$BASE"); o=$(count_pkl "$OPD")
  echo "$(date '+%F %T') base=$b opd=$o"
  if [ "$b" -ge 500 ] && [ "$o" -ge 500 ]; then break; fi
  # 若父缓存脚本已死且长期 0，报错退出
  if ! pgrep -f 'run_ch5_score_caches.sh' >/dev/null && [ "$b" -eq 0 ]; then
    echo "cache script dead and base empty"; exit 1
  fi
  sleep 60
done

source /home/ws/anaconda3/etc/profile.d/conda.sh
conda activate conda_gtrs
python "$ROOT/scripts/opd/analysis/decision_pca.py" scoring \
  --teacher "$TEACH" \
  --student "base_r34=$BASE" \
  --student "opd_r34=$OPD" \
  --max-scenes 2000 \
  --out "$FIG/fig5_4_scoring_pca.png" \
  --csv "$FIG/fig5_4_scoring.csv" --plot
echo "DONE fig5_4"
ls -lh "$FIG/fig5_4_scoring_pca.png"
