#!/bin/bash
# 用同一个打分器、指定的第一阶段交通参与者策略，给"已存好的预测 pickle"重新打 EPDMS（纯 CPU，不跑模型）。
#
# 为什么需要：run_pdm_score_gpu_v2_aug.py（GTRS/OPD 评测）第一阶段固定用 IDM reactive 交通流，
# DriveSuprim 的 run_pdm_score_one_stage_gpu_ssl.py 用 cfg.traffic_agents=non_reactive（log replay）。
# 两套数字不能直接比。两边的 pickle 都是 {token: {'trajectory': Trajectory, ...}}，打分只读 trajectory，可互换。
#
# 用法：
#   PKL=<预测 pickle> TAG=<名字> [POLICY=non_reactive|reactive] bash scripts/opd/evaluation/rescore_pickle.sh
# 例（同一步数 4 格交叉，就能把"模型差距"和"打分口径差距"拆开）：
#   PKL=$ABL/evals/opd/step13000_subscores.pkl                       TAG=opd_13k POLICY=non_reactive
#   PKL=$ABL/official_drivesuprim_r34/step-step=013000.pkl             TAG=ds_13k  POLICY=reactive
#   （ABL=/home/ws/navsim_workspace/exp/ablation_same_iter_r34；DriveSuprim 评测 save_pickle=true 存在 ckpt 旁）
#   （另两格 = 已有数字：opd reactive 76.5 / ds non_reactive 80.69）
set -euo pipefail

: "${PKL:?PKL=<prediction pickle> required}"
: "${TAG:?TAG=<name> required}"
policy=${POLICY:-non_reactive}
metric_cache=${METRIC_CACHE:-${NAVSIM_EXP_ROOT}/navtest_two_stage_metric_cache}
sensor=${SENSOR:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test/test}
threads=${WORKER_THREADS:-8}
exp_name="rescore/${TAG}-${policy}"

if [ ! -f "$PKL" ]; then echo "Error: PKL not found: $PKL" >&2; exit 1; fi

echo "=== rescore ==="
echo "  pkl    : $PKL"
echo "  policy : $policy (stage one)"
echo "  exp    : $exp_name"

cd "${NAVSIM_DEVKIT_ROOT}"
env SKIP_INFER=1 SUBSCORE_PATH="$PKL" STAGE1_TRAFFIC_AGENTS="$policy" PROGRESS_MODE=eval \
  python navsim/planning/script/run_pdm_score_gpu_v2_aug.py \
    agent=gtrs_aug_opd_r34 train_test_split=navtest \
    agent.checkpoint_path=unused \
    worker.threads_per_node="$threads" \
    experiment_name="$exp_name" +cache_path=null \
    metric_cache_path="$metric_cache" \
    original_sensor_path="$sensor"

csv=$(find "${NAVSIM_EXP_ROOT}/${exp_name}" -name '*.csv' -type f | sort | tail -n 1)
python - "$csv" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
v = df[(df.valid == True) & ~df.token.astype(str).str.contains("average|extended_pdm_score|stage")]
print(f"csv={sys.argv[1]}")
print(f"n_valid={len(v)}  EPDMS={v.score.mean()*100:.2f}  zero_pct={(v.score==0).mean()*100:.2f}")
PY
