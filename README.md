# sj-opd-bases

Side-by-side snapshots for analyzing OPD vs DriveSuprim/AUG EMA soft-label.

| dir | source | role |
|---|---|---|
| `opd/` | local `GTRS_official` @ `sj-opd` | Offline ViT-L distillation (OPD), `gtrs_aug_opd_r34` |
| `drivesuprim/` | local `DriveSuprim-main` | Official EMA teacher + soft labels (`drivesuprim_agent_r34`) |

Same-iter R34 ablation compared these two recipes (shared LR/BS/accum/13000 steps).

Branch: `sj-opd-bases` on https://github.com/Jacky-chen-sir/sj-1
OPD-only evolving branch: `sj-opd`
