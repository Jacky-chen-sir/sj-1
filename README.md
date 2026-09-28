# sj-opd-bases

Side-by-side source trees for the same-iter R34 ablation (3×3090, max_steps=13000).

| dir | local source | agent / role |
|---|---|---|
| `opd/` | `GTRS_official` @ `sj-opd` | `gtrs_aug_opd_r34` — offline ViT-L OPD |
| `drivesuprim/` | `DriveSuprim-main` | `drivesuprim_agent_r34` — EMA teacher + soft labels |
| `gtrsori/` | `GTRSori` | `gtrs_aug_r34` — official_gtrs_r34 same-iter baseline |
| `launch/` | ablation scripts + `COMMANDS.md` | train/eval launchers for all three |

Shared lock: `launch/ablation_same_iter_r34/hyperparams.lock.txt`  
Scores: `launch/ablation_same_iter_r34/scores.csv`
