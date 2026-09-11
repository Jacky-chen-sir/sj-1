# GTRS 论文工作区（FM-DP + DriveSuprim AUG）

基于 [NVlabs/GTRS](https://github.com/NVlabs/GTRS) 的硕士论文代码库。上游原始说明见 [README_GTRS.md](README_GTRS.md)。

本仓库在官方代码基础上的主要工作：

- **FM-DP 轨迹提案生成**（论文第 3 章）：基于流匹配（Flow Matching）的轨迹生成模型，替代/增强扩散式提案生成。
- **DriveSuprim 配方的 AUG 评分模型**（论文第 4 章）：将 DriveSuprim 训练配方（旋转增强 + 级联精炼 + 软标签）移植到 `gtrs_aug` 评分器，并完成多骨干网络改造（ResNet-34 / ResNet-50 / ViT / VoV-99）。
- **生成-评分融合推理**：评分模型在推理时同时给词表候选与 FM-DP 生成提案打分，argmax 选出最终轨迹。

## 1. 环境

- OS: Linux（开发机 3 × RTX 3090 24GB）
- Conda 环境：`conda_gtrs`
- 依赖安装：按上游 [README_GTRS.md](README_GTRS.md) 安装 NAVSIM devkit 及依赖

必须的环境变量：

| 变量 | 示例值 | 说明 |
|---|---|---|
| `NAVSIM_DEVKIT_ROOT` | `/home/ws/navsim_workspace/GTRS_official` | 本仓库根目录 |
| `NAVSIM_EXP_ROOT` | `/home/ws/navsim_workspace/exp` | 实验输出（checkpoint / 日志） |
| `OPENSCENE_DATA_ROOT` | `/home/ws/navsim_workspace/dataset` | 数据集根目录 |
| `NAVSIM_TRAJPDM_ROOT` | `/home/ws/navsim_workspace/dataset/traj_pdm_v2` | 离线 PDM 标签缓存 |

## 2. 数据准备

```
$OPENSCENE_DATA_ROOT/
├── sensor_blobs/          # navtrain/navtest 传感器数据（本机为 /mnt/bigdisk 的软链接）
├── maps/                  # nuPlan 地图
└── traj_pdm_v2/           # 离线标签（训练 AUG 必需）
    ├── ori/vocab_score_8192_navtrain_final/navtrain.pkl
    └── random_aug/
        ├── rot_30-trans_0-va_0-p_0.5-ensemble.json        # 预采样旋转角
        └── rot_30-p_0.5-ensemble/vocab_score_8192_navtrain_final/split_pickles/
```

轨迹词表已随仓库提供：`traj_final/8192.npy`（kmeans 聚类 navtrain GT 轨迹得到）。

离线标签的生成脚本见 `navsim/agents/tools/`（`gen_vocab_full_score*.py`、`gen_offline_training_aug_file*.py`）。

## 3. 训练启动

当前在训配置（DriveSuprim 配方：rot±30°、p=0.5、3 个旋转增强视图、1 阶段精炼、top-256）：

```bash
conda activate conda_gtrs
export NAVSIM_DEVKIT_ROOT=/home/ws/navsim_workspace/GTRS_official
export NAVSIM_EXP_ROOT=/home/ws/navsim_workspace/exp
export NAVSIM_TRAJPDM_ROOT=/home/ws/navsim_workspace/dataset/traj_pdm_v2

BS=2 NPROC=3 MASTER_PORT=29502 PRECISION=32 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/drivesuprim/training/rot_30-p_0.5/train.sh gtrs_aug_drivesuprim_vov 1 3 256
```

位置参数：`train.sh <agent> <num_refinement_stage> <stage_layers> <topks>`

可选 agent：`gtrs_aug_drivesuprim_vov`（默认，VoV-99）、`gtrs_aug_drivesuprim_r34`、`gtrs_aug_drivesuprim_r50`、`gtrs_aug_drivesuprim_vit`

可调环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `BS` | 8 | 每卡 batch size（3090 24G 建议 2） |
| `LR` | 7.5e-5 | 学习率 |
| `NPROC` | 8 | GPU 进程数 |
| `MASTER_PORT` | 29500 | DDP 端口 |
| `PRECISION` | 32 | **必须用 32**；fp16 会出现 NaN（梯度溢出） |
| `RESUME_CKPT` | 无 | 断点续训（恢复权重+优化器+epoch），例：`RESUME_CKPT=/abs/path.ckpt` |

断点续训示例：

```bash
RESUME_CKPT=$NAVSIM_EXP_ROOT/training/gtrs_aug_drivesuprim_vov/rot_30-p_0.5/stage_layers_3-topks_256/lightning_logs/version_0/checkpoints/last.ckpt \
BS=2 NPROC=3 MASTER_PORT=29502 PRECISION=32 \
bash scripts/drivesuprim/training/rot_30-p_0.5/train.sh gtrs_aug_drivesuprim_vov 1 3 256
```

## 4. 评测启动

navtest 上评测（PDM 评分）：

```bash
# 仅评分器（词表候选）
bash scripts/evaluation/eval_drivesuprim_vov_navtest.sh

# 评分器 + FM-DP 生成提案（融合推理）
WITH_FM_DP=1 bash scripts/evaluation/eval_drivesuprim_vov_navtest.sh

# 指定 checkpoint / GPU
CKPT=/path/to.ckpt GPU=0 BS=8 WORKERS=8 bash scripts/evaluation/eval_drivesuprim_vov_navtest.sh
```

其他评测脚本（`scripts/evaluation/`）：`eval_dp_fm_joint_navtest.sh`（DP+FM 联合）、`eval_aug_with_fm_dp_navtest.sh`（AUG+FM）。

## 5. 参考结果（navtest PDM）

| 方法 | PDM 总分 |
|---|---|
| GTRS-DP（官方） | 0.7305 |
| DiffusionDrive | 0.7318 |
| FM-DP（本文第 3 章） | 0.738 |
| AUG + FM-DP 融合（本文） | 0.7935 |
| 官方 AUG（参考上限） | 0.7939 |

## 6. OPD 蒸馏（第 4 章）

教师 = 冻结 ViT-L（87.1），**离线打分、训练时完全不前向**；学生 = R34，三路蒸馏损失
（imi 分布 KL + 8 头逐轨迹二元 KL + 融合分数 listwise/召回）。与上面 §3 的 AUG 训练互斥（`ban_soft_label_loss` 已强制）。

### 三步跑法（远程 3×3090）

```bash
# 1) 教师打分缓存（一次性，navtrain ~103k token：含 8 头约 20GB；OPD_STORE_HEADS=0 则 ~6.8GB）
#    磁盘紧张时用 OPD_STORE_HEADS=0 并把 OPD_LAMBDA_HEAD=0（否则那一路自动置零，白填权重）
export OPD_TEACHER_SCORE_DIR=$NAVSIM_TRAJPDM_ROOT/opd/teacher/vit_l/8192_cache_ori
TEACHER_CKPT=$NAVSIM_EXP_ROOT/models/gtrs_aug_drivesuprim_vit/epoch=05-step=1330.ckpt \
NPROC=3 bash scripts/opd/training/cache_teacher.sh

# 2) 训练（主实验 rounds=0 即可跑）
OPD_TEACHER_SCORE_DIR=$OPD_TEACHER_SCORE_DIR \
bash scripts/opd/training/train_opd.sh gtrs_aug_opd_r34 1 3 256

# 3) 评估（按文件名 glob ckpt，不再算 step=epoch*1330）
bash scripts/opd/evaluation/eval_opd.sh 5 \
  training/opd/gtrs_aug_opd_r34/rot_30-p_0.5/stage_layers_3-topks_256 1 3 256
```

### OPD 训练旋钮（`OPD_*` env → `++agent.config.opd.*`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OPD_TEACHER_SCORE_DIR` | `$NAVSIM_TRAJPDM_ROOT/opd/...` | 教师缓存目录（必填；否则用 yaml 默认） |
| `OPD_TAU_IMI / _HEAD / _LIST` | 2.0 | 三路蒸馏温度（imi / 8-PDM 头 / 融合分数 listwise） |
| `OPD_LAMBDA_IMI / _HEAD / _REFINE / _RECALL` | 1.0/1.0/1.0/0.5 | 四路损失权重（消融逐个置 0） |
| `OPD_TOPK_REFINE / _RECALL` | 256 / 32 | 精排 listwise 与召回的 Top-K |
| `OPD_ON_POLICY_ROUNDS / _WEIGHT` | 0 / 0.5 | on-policy 轮数与权重（0=关） |
| `OPD_ON_POLICY_VIEW_IDX` | 1 | on-policy 缓存配对的学生视图下标；与缓存 `view_idx` 双向断言 |
| `OPD_STORE_HEADS` | 1 | 教师缓存是否含 8 头 logits；**须与 `cache_teacher.sh` 一致**，为 0 时 `lambda_head` 自动置零 |
| `OPD_TEACHER_MODE` | offline | `offline`（读缓存）/ `ema`（原在线软标签教师，消融对照）/ `none`（同路径纯学生基线） |
| `BS` / `NPROC` / `ACCUM` | 3 / 3 / 4 | 有效批量 = 三者乘积；LR 自动线性缩放（基线 7.5e-5 @ 64） |
| `PRECISION` | 32 | **必须 32**；`sync_batchnorm` 已开 |

### 消融 sweep

```bash
# 全量；或 DRY_RUN=1 先看命令
bash scripts/opd/ablation/sweep_opd.sh
GROUPS="default im_only tau_imi_1 rounds_1" bash scripts/opd/ablation/sweep_opd.sh
```

组名见 `scripts/opd/ablation/sweep_opd.sh` 顶部注释（λ 独立性 / τ / λ 组合 / on-policy 轮数 / EMA 对照）。
词表规模消融（4096/16384）需重算 PDM 分数与教师缓存，不在 sweep 里做。



| 路径 | 内容 |
|---|---|
| `navsim/agents/drivesuprim/` | DriveSuprim agent（agent/config/model/loss） |
| `navsim/agents/gtrs_aug/` | AUG 评分模型；`hydra_backbone.py` 已移植多骨干（R34/R50/ViT/VoV） |
| `navsim/agents/tools/` | 离线标签/词表分数生成脚本 |
| `navsim/planning/script/run_training_aug.py` | AUG 训练入口 |
| `navsim/planning/script/run_teacher_opd_cache.py` | OPD 教师离线打分缓存工具（流式 per-token 写盘，无 gather） |
| `navsim/agents/gtrs_aug/hydra_loss_fn_aug.py` | `opd_distill_loss`——三路蒸馏（imi-KL + 8头 BCE-soft + 融合 listwise/召回） |
| `navsim/agents/gtrs_aug/tests/test_opd.py` | 数值回归测试（远程 `python -m pytest navsim/agents/gtrs_aug/tests/test_opd.py -q`） |
| `scripts/opd/` | OPD 训练/缓存/评估/消融脚本（镜像 `drivesuprim/` 结构） |
| `scripts/drivesuprim/` | DriveSuprim 配方训练/评测脚本 |
| `scripts/evaluation/` | navtest 评测与融合推理脚本 |
| `traj_final/` | 轨迹词表（8192/16384）与聚类脚本 |
