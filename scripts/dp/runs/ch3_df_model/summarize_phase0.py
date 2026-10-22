#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总第三章 Phase 0（推断方式对照）的评测结果。

Phase 0 会产出若干个实验目录：

    $NAVSIM_EXP_ROOT/ch3_df/p0_first_1/...   （官方行为，重复 3 次）
    $NAVSIM_EXP_ROOT/ch3_df/p0_medoid/...
    $NAVSIM_EXP_ROOT/ch3_df/p0_zero/...

每个目录下是评测脚本写的 CSV。本脚本从每份 CSV 里取总分行
（`extended_pdm_score_stage_one` / `_combined`），输出对照表与
`summary_phase0.csv`。

用法：

    python scripts/dp/runs/ch3_df_model/summarize_phase0.py \
        --root $NAVSIM_EXP_ROOT/ch3_df \
        --out  $NAVSIM_EXP_ROOT/ch3_df/summary_phase0.csv
"""
import argparse
import csv
import statistics
from pathlib import Path

SUMMARY_TOKENS = (
    'extended_pdm_score_stage_one',
    'extended_pdm_score_combined',
    'pdm_score_stage_one',
    'average_all_frames',
)


def read_score(csv_path: Path):
    """返回 (score, aggregation_mode)；找不到返回 (None, None)。"""
    try:
        with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None, None
    if not rows:
        return None, None
    mode = rows[0].get('aggregation_mode', '')
    for tok in SUMMARY_TOKENS:
        for r in rows:
            if (r.get('token') or '').strip() == tok:
                v = (r.get('score') or '').strip()
                if v:
                    try:
                        return float(v), mode
                    except ValueError:
                        pass
    return None, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Phase 0 实验根目录')
    ap.add_argument('--out', default=None, help='输出 CSV（默认 <root>/summary_phase0.csv）')
    ap.add_argument('--group', default='p0_', help='只统计名字含该前缀的目录')
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f'目录不存在: {root}')

    records = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or args.group not in d.name:
            continue
        csvs = sorted(d.rglob('*.csv'))
        if not csvs:
            continue
        # 取最新的一份（评测脚本可能重跑多次）
        csvs = sorted(csvs, key=lambda p: p.stat().st_mtime, reverse=True)
        score, mode = read_score(csvs[0])
        records.append({'tag': d.name, 'score': score, 'mode': mode, 'csv': str(csvs[0])})

    if not records:
        raise SystemExit(f'没有在 {root} 下找到含 {args.group!r} 的评测结果')

    print(f'{"tag":28s} {"score":>10s}  aggregation_mode')
    print('-' * 62)
    for r in records:
        s = f'{r["score"]:.4f}' if r['score'] is not None else 'N/A'
        print(f'{r["tag"]:28s} {s:>10s}  {r["mode"]}')

    # first 的均值 ± 标准差（只统计 FM-DP；官方 DDPM 对照不参与）
    firsts = [r['score'] for r in records
              if 'first' in r['tag'] and 'official' not in r['tag']
              and r['score'] is not None]
    if len(firsts) >= 2:
        print(f'\nfirst（n={len(firsts)}）: mean={statistics.mean(firsts):.4f} '
              f'std={statistics.stdev(firsts):.4f} '
              f'min={min(firsts):.4f} max={max(firsts):.4f}')
    elif len(firsts) == 1:
        print(f'\nfirst 只有 1 次，无法估计方差；建议补跑 2 次。')

    # 官方 DDPM 对照 vs FM first 均值
    officials = [r['score'] for r in records
                 if 'official' in r['tag'] and r['score'] is not None]
    if officials and firsts:
        base = statistics.mean(firsts)
        print('\n官方 DDPM 对照：')
        for r in records:
            if 'official' not in r['tag'] or r['score'] is None:
                continue
            print(f'  {r["tag"]:26s} {r["score"]:.4f}（相对 FM first {base:.4f}：'
                  f'{r["score"] - base:+.4f}）')

    # 与 first 均值的差（FM 各档）
    if firsts:
        base = statistics.mean(firsts)
        print('\n相对 first 均值：')
        for r in records:
            if r['score'] is None or 'first' in r['tag']:
                continue
            print(f'  {r["tag"]:26s} {r["score"] - base:+.4f}')

    out = Path(args.out) if args.out else root / 'summary_phase0.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['tag', 'score', 'mode', 'csv'])
        w.writeheader()
        w.writerows(records)
    print(f'\n已写 {out}')


if __name__ == '__main__':
    main()
