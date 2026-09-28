# Same-iter R34 ablation — launch commands

Machine paths below are from the original 3×RTX 3090 run
(`/home/ws/navsim_workspace`). Remap `NAVSIM_*` / conda env as needed.

**Fair pair (protocol):** OPD offline ViT-L ↔ DriveSuprim EMA soft-label  
**Shared lock:** see `ablation_same_iter_r34/hyperparams.lock.txt`  
R34 · LR=`4.219e-05` · BS=3 · NPROC=3 · ACCUM=4 · eff=36 · fp32 · `max_steps=13000` · refine `1/3/256` · rot±30 p=0.5

Scores: `ablation_same_iter_r34/scores.csv`

---

## 0) Env (common)

```bash
source /home/ws/anaconda3/etc/profile.d/conda.sh
conda activate conda_gtrs

export OPENSCENE_DATA_ROOT=/home/ws/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/ws/navsim_workspace/exp
export NAVSIM_TRAJPDM_ROOT=/home/ws/navsim_workspace/dataset/traj_pdm_v2
export NUPLAN_MAPS_ROOT=/home/ws/navsim_workspace/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/mnt/bigdisk/tmp
export DDP_FIND_UNUSED=1
```

---

## 1) OPD train (GTRS_official / `opd/` on sj-opd-bases)

Repo root = `GTRS_official` (branch `sj-opd`).

```bash
export NAVSIM_DEVKIT_ROOT=/home/ws/navsim_workspace/GTRS_official
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT
export OPD_TEACHER_SCORE_DIR=$NAVSIM_TRAJPDM_ROOT/opd/teacher/vit_l/8192_cache_ori

cd $NAVSIM_DEVKIT_ROOT
# teacher cache once (if missing):
# TEACHER_CKPT=... NPROC=3 bash scripts/opd/training/cache_teacher.sh

# train (same-iter lock: BS=3 NPROC=3 ACCUM=4 → LR 4.219e-05)
BS=3 NPROC=3 ACCUM=4 LR=4.219e-05 MAX_EPOCHS=10 \
OPD_TEACHER_SCORE_DIR=$OPD_TEACHER_SCORE_DIR \
bash scripts/opd/training/train_opd.sh gtrs_aug_opd_r34 1 3 256
```

Historical ckpt dir used for scores:  
`exp/training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256/`  
(compare at step 5000 / 9000 / 13000)

Eval (vocab-only, student):

```bash
bash scripts/opd/evaluation/eval_opd.sh 5 \
  training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256 1 3 256
```

Helper copies also in `opd_scripts/` of this folder.

---

## 2) DriveSuprim same-iter train (`drivesuprim/` / DriveSuprim-main)

```bash
bash ablation_same_iter_r34/run_official_drivesuprim_r34.sh
# (script embeds absolute paths; edit NAVSIM_* inside if relocating)
```

Equivalent hydra core (3 GPU):

```bash
export NAVSIM_DEVKIT_ROOT=/home/ws/navsim_workspace/DriveSuprim-main
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT
export RAY_TMPDIR=/mnt/bigdisk/tmp/ray
cd $NAVSIM_DEVKIT_ROOT

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master_port=29513 \
  navsim/planning/script/run_training_ssl.py \
  +debug=false agent=drivesuprim_agent_r34 \
  experiment_name=ablation_same_iter_r34/official_drivesuprim_r34 \
  split=trainval train_test_split=navtrain \
  dataloader.params.batch_size=3 dataloader.params.num_workers=4 \
  '~trainer.params.strategy' \
  trainer.params.precision=32 trainer.params.max_epochs=10 \
  trainer.params.limit_val_batches=0.1 \
  ++trainer.params.accumulate_grad_batches=4 \
  ++trainer.params.sync_batchnorm=true \
  ++trainer.params.max_steps=13000 \
  agent.config.ckpt_path=ablation_same_iter_r34/official_drivesuprim_r34 \
  agent.lr=4.219e-05 \
  agent.config.ego_perturb.n_student_rotation_ensemble=3 \
  agent.config.ego_perturb.offline_aug_angle_boundary=30 \
  agent.config.ego_perturb.offline_aug_file=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_30-trans_0-va_0-p_0.5-ensemble.json \
  agent.config.refinement.use_multi_stage=true \
  agent.config.refinement.num_refinement_stage=1 \
  agent.config.refinement.stage_layers=3 \
  agent.config.refinement.topks=256 \
  agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_navtrain_final/navtrain.pkl \
  agent.config.aug_vocab_pdm_score_dir=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_30-p_0.5-ensemble/vocab_score_8192_navtrain_final/split_pickles \
  cache_path=null
```

Ckpt: `exp/ablation_same_iter_r34/official_drivesuprim_r34/step-step={005000,009000,013000}.ckpt`

### DriveSuprim eval (vocab-only, teacher, 3-GPU parallel)

```bash
# recommended: bs=8 workers=4 ray_threads=2; save_pickle=true
STEPS=5000  EVAL_GPU=0 EVAL_BS=8 EVAL_WORKERS=4 WORKER_THREADS=2 \
  bash ablation_same_iter_r34/eval_official_drivesuprim_r34.sh
STEPS=9000  EVAL_GPU=1 EVAL_BS=8 EVAL_WORKERS=4 WORKER_THREADS=2 \
  bash ablation_same_iter_r34/eval_official_drivesuprim_r34.sh
STEPS=13000 EVAL_GPU=2 EVAL_BS=8 EVAL_WORKERS=4 WORKER_THREADS=2 \
  bash ablation_same_iter_r34/eval_official_drivesuprim_r34.sh
```

Do **not** run three Ray scorers with `threads_per_node=8` at once (SIGBUS / OOM).

---

## 3) official_gtrs_r34 (GTRSori EMA soft-label baseline, same lock)

Repo root on this branch = `gtrsori/` (local `GTRSori`). Set `NAVSIM_DEVKIT_ROOT` to that tree.

```bash
bash ablation_same_iter_r34/run_official_gtrs_r34.sh
```

Eval infer (GTRSori) + optional SKIP_INFER rescore (GTRS_official):

```bash
bash ablation_same_iter_r34/eval_official_gtrs_r34.sh
# then, if needed:
bash ablation_same_iter_r34/score_official_gtrs_r34_skipinfer.sh
```

Note: published gtrs 13k scores used `inference.model=teacher`; OPD uses **student**.

---

## 4) Result snapshot (EPDMS %, n_valid≈11990–11992)

| method | 5k | 9k | 13k |
|---|---|---|---|
| OPD | 74.00 | 75.62 | 76.50 |
| official_gtrs_r34 | 70.48 | 74.93 | 76.61 |
| official_drivesuprim_r34 | 71.24 | 78.58 | **80.69** |

See `ablation_same_iter_r34/scores.csv` for full subscores.
