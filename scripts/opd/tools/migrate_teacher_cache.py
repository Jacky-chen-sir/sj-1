# SPDX-License-Identifier: Apache-2.0
#
# 一次性迁移：把既有的教师缓存刷成「HC=2」的融合分数口径。
#
# 背景：`fused_coarse_score` 的舒适组 HC 权重由 1.0 改为 2.0（对齐 EPDMS 评测口径与论文式 4-7）。
# 缓存里的 `coarse`（粗筛融合分数）与 `topk_idx`（其 top-K）都是旧口径的产物，
# 不刷新的话训练端读到的是另一条公式的分数——静默错配，不报错。
#
# 关键点：**不需要重跑 ViT-L**。缓存里已经存了 8 个头的原始 logits（`OPD_STORE_HEADS=1` 时），
# 融合分数是这些 logits 的纯函数，本地 CPU 就能逐 token 重算。
#
# `topk_score`（精排头在 top-K 上的打分）**无法重算**——它依赖精排头的前向，缓存里没有。
# 但训练端从不读它（`opd_distill_loss` 只用 `coarse` + `topk_idx`），所以保持原值即可。
#
# 用法（本机或远程均可，纯 CPU）：
#   python scripts/opd/tools/migrate_teacher_cache.py --cache_dir <dir> [--dry_run] [--limit N]
#
# 建议先在 `--limit 200 --dry_run` 下核对：脚本会打印新旧 coarse 的差异分布与 topk 重合率。
# 验证通过后去掉 --dry_run 全量刷。脚本是**幂等**的（重复跑结果相同），可断点续跑。

import argparse
import glob
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from navsim.agents.gtrs_aug.hydra_model import fused_coarse_score  # noqa: E402

_METRICS = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound',
            'ego_progress', 'driving_direction_compliance', 'lane_keeping',
            'traffic_light_compliance', 'history_comfort')


class _Cfg:
    """`fused_coarse_score` 的 safe 分支不读 config；dual_stream 分支才读 opd.beta_imi。"""
    class _OPD:
        beta_imi = 0.02
    opd = _OPD()


def _recompute(payload, dual_stream: bool):
    """从缓存里的逐头 logits 重算 (coarse, topk_idx)；缺头则返回 (None, None)。"""
    if not all(m in payload for m in _METRICS):
        return None, None
    head_out = {m: torch.from_numpy(np.asarray(payload[m], dtype=np.float32)).unsqueeze(0) for m in _METRICS}
    head_out['imi'] = torch.from_numpy(np.asarray(payload['imi'], dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        coarse = fused_coarse_score(head_out, _Cfg(), safe=True, dual_stream=dual_stream)
    k = int(np.asarray(payload['topk_idx']).shape[-1])
    topk_idx = torch.topk(coarse, k=k, dim=1).indices
    return coarse[0].numpy().astype(np.float32), topk_idx[0].numpy().astype(np.int32)


def _atomic_dump(obj, final_path: str) -> None:
    tmp = f"{final_path}.tmp.{os.getpid()}.{datetime.now().microsecond}"
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, final_path)  # POSIX 原子


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', required=True, help='per-token pickle 目录')
    ap.add_argument('--dry_run', action='store_true', help='只统计不写盘')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个 token（0=全部）')
    ap.add_argument('--dual_stream', action='store_true',
                    help='缓存是用 opd.dual_stream_score=true 生成的；默认 false')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, '*.pkl')))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f'没有找到 *.pkl：{args.cache_dir}')
        return 1

    n_done = n_skip_nohead = n_missing = 0
    dmax_all, overlap_all = [], []
    for i, path in enumerate(files):
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
        except Exception as e:                                   # noqa: BLE001
            print(f'[warn] 读取失败 {os.path.basename(path)}: {e}')
            n_missing += 1
            continue

        new_coarse, new_topk = _recompute(payload, args.dual_stream)
        if new_coarse is None:
            n_skip_nohead += 1
            continue

        old_coarse = np.asarray(payload['coarse'], dtype=np.float32)
        old_topk = np.asarray(payload['topk_idx'])
        dmax_all.append(float(np.abs(new_coarse - old_coarse).max()))
        overlap_all.append(float(np.isin(new_topk, old_topk).mean()))

        if not args.dry_run:
            payload['coarse'] = new_coarse
            payload['topk_idx'] = new_topk
            # 顺带补上训练端一致性断言需要的键（老缓存没有）。
            payload.setdefault('dual_stream_score', bool(args.dual_stream))
            _atomic_dump(payload, path)
        n_done += 1

        if (i + 1) % 5000 == 0:
            print(f'  ... {i + 1}/{len(files)}')

    print(f'\n处理 {n_done} 个 token（缺 8 头跳过 {n_skip_nohead}，读取失败 {n_missing}）')
    if dmax_all:
        d = np.array(dmax_all)
        o = np.array(overlap_all)
        print(f'|Δcoarse|_max  : mean {d.mean():.4f}  p50 {np.percentile(d, 50):.4f}  '
              f'p99 {np.percentile(d, 99):.4f}  max {d.max():.4f}')
        print(f'topk 重合率    : mean {o.mean():.4f}  min {o.min():.4f}  '
              f'完全重合的比例 {float((o == 1.0).mean()):.4f}')
        if d.max() < 1e-6:
            print('\n⚠ 新旧 coarse 完全一致——说明这份缓存本来就是 HC=2 口径（或已被迁移过）。')
        elif not args.dry_run:
            print(f'\n✅ 已原地刷新 {args.cache_dir}')
        else:
            print('\n（dry run，未写盘；确认上面的差异合理后去掉 --dry_run 重跑）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
