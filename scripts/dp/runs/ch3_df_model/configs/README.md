# 训练启动代码与 Hydra 配置（第三章 ch3_df）

## 启动脚本（仓库路径）

| 脚本 | 作用 |
|---|---|
| `scripts/training/train_dp_official_fm.sh` | **主训练入口**：官方 DP → FM；读环境变量拼 Hydra 覆盖 |
| `scripts/dp/runs/ch3_df_model/run_phase1.sh` | P1：`p1_base` / `p1_center` 5ep + zero 评测 |
| `scripts/dp/runs/ch3_df_model/run_phase2.sh` | P2：`p2_center18` 18ep + zero 评测 |
| `scripts/dp/runs/ch3_df_model/run_phase0.sh` | P0：仅评测 |
| `scripts/evaluation/eval_dp_fm_joint_navtest.sh` | navtest FM/DP 评测 |
| `scripts/dp/runs/ch3_df_model/eval_ckpts.sh` | 多 ckpt × zero/first |

示例（与已跑 P2 一致）：

```bash
cd $NAVSIM_DEVKIT_ROOT   # GTRS_official
FM_SELF_CONDITIONING=true ARM=center EPOCHS=18 \
  bash scripts/dp/runs/ch3_df_model/run_phase2.sh
# 等价于内部调用：
# MAX_EPOCHS=18 EXP_NAME=ch3_df/p2_center18 REINIT_TRAJ_HEAD=true \
#   FM_CENTER_WEIGHT=0.5 FM_CENTER_STEPS=4 FM_CENTER_P=1.0 \
#   FM_CENTER_WARMUP_STEPS=2000 FM_SELF_CONDITIONING=true \
#   bash scripts/training/train_dp_official_fm.sh
```

## 仓库内基础 YAML

| 文件 | 说明 |
|---|---|
| `navsim/planning/script/config/training/competition_training.yaml` | `--config-name competition_training` |
| `navsim/planning/script/config/common/agent/gtrs_diffusion_policy.yaml` | `agent=gtrs_diffusion_policy` |
| 本目录 `training/`、`agent/` | 上述二者的副本（便于单目录浏览） |

`fm_center_*` / `dp_eval_zero_noise` 等默认在 **`DPConfig`（Python）**；训练时由 `train_dp_official_fm.sh` 用 `++agent.config.*` 覆盖，见各次实验的 `overrides.yaml`。

## 本目录：真实跑通快照

只保留 **overrides + hydra**（完整 `config.yaml` 含整表 log_names，约 3MB，不入库）。

| 子目录 | 对应实验 |
|---|---|
| `p1_base/` | center_w=0，5ep |
| `p1_center/` | center_w=0.5，5ep |
| `p2_center18/` | center_w=0.5，18ep |

关键覆盖（以 `p2_center18/overrides.yaml` 为准）：

- `agent=gtrs_diffusion_policy`，`train_test_split=navtrain`
- `reinit_traj_head=true`，`use_flow_matching=true`
- `fm_center_weight=0.5`，`fm_center_steps=4`，`fm_center_warmup_steps=2000`
- `fm_self_conditioning=true`，`scorer_guidance=false`
- `max_epochs=18`（P1 为 5），`devices=3`，`batch_size=8`
