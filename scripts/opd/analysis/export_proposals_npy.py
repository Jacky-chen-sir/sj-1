#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 eval 产出的 proposals.pkl 导出成 decision_pca.py 可读的 [N,H,3] npy。

默认取 `interpolated_proposal`（40 步 @10Hz），与 8192 词表同口径；
也可 `--key proposals` 取 8 步 @2Hz。
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--key', default='interpolated_proposal',
                    choices=['interpolated_proposal', 'proposals'])
    ap.add_argument('--max-scenes', type=int, default=None)
    ap.add_argument('--max-props-per-scene', type=int, default=None,
                    help='每场景最多保留多少条提案（默认全留）')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f'期望 token→dict，得到 {type(data)}')

    rng = np.random.default_rng(args.seed)
    chunks = []
    for i, (_tok, rec) in enumerate(data.items()):
        if args.max_scenes is not None and i >= args.max_scenes:
            break
        if not isinstance(rec, dict) or args.key not in rec:
            continue
        arr = np.asarray(rec[args.key], dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if args.max_props_per_scene is not None and arr.shape[0] > args.max_props_per_scene:
            idx = rng.choice(arr.shape[0], size=args.max_props_per_scene, replace=False)
            arr = arr[idx]
        chunks.append(arr)

    if not chunks:
        raise SystemExit(f'{args.pkl} 里没有可用的 {args.key}')
    out = np.concatenate(chunks, axis=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f'已写 {args.out}  shape={out.shape}  scenes={len(chunks)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
