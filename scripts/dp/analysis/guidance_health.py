#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三章评价器引导的「健康度」体检。

## 为什么需要它

引导项的量级没有天然上界：评分头输出的是对数空间分数（约 -3.5~0），
而流匹配主损失收敛后可能只有 0.05。若直接相加，引导项可以轻易压过主损失——
这正是 2026-10-14 那次 5-epoch 续训从 0.7380 掉到 0.6369 的机制之一
（主因是时间门控写反，见 3.4.7 节与 scripts/dp/runs/ch3_20261014_lambda01/RUN.md）。

训练时若出现下面任一信号，应当立即停下检查权重，而不是等跑完看 navtest：

  · guidance / base > 1        —— 引导项已经压过主损失
  · fm_safety_prod 长期 < 0.2   —— 安全 hinge 一直激活，会主导梯度
  · fm_guidance_frac ≈ 0        —— 时间区间配置错误，引导根本没生效

## 输入

Lightning 的训练日志（文本）或 CSVLogger 的 metrics.csv 都可以。
脚本按 `name=value` / `name,value` 两种格式抽取字段。

用法：

    python scripts/dp/analysis/guidance_health.py exp/logs/xxx.log
    python scripts/dp/analysis/guidance_health.py metrics.csv --plot health.png
"""
import argparse
import re
import sys
from pathlib import Path

KEYS = [
    'fm_loss_base',          # 不含引导的损失（需要 b2b3d58 之后的代码）
    'fm_guidance_score',     # 评分引导 l_a
    'fm_safe_feas',          # 安全 hinge l_b1
    'fm_safety_prod',        # 四项安全概率乘积
    'fm_comfort_div',        # 多样性损失 l_b2（需要 b2b3d58 之后的代码）
    'fm_comfort_std',        # 舒适分数的批内标准差（不是损失）
    'fm_guidance_frac',      # 落在引导时间区间内的样本比例
    'fm_guidance_warm',      # 当前 warmup 系数
    'fm_loss',               # 早期代码里出现过
    'dp_loss',
]
NUM = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')


def parse_text(path: Path):
    """从任意文本日志里抽 name=value。"""
    series = {k: [] for k in KEYS}
    pat = re.compile(r'\b(' + '|'.join(map(re.escape, KEYS)) + r')\s*[=:]\s*(' + NUM.pattern + r')')
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            for m in pat.finditer(line):
                series[m.group(1)].append(float(m.group(2)))
    return series


def parse_csv(path: Path):
    import csv
    series = {k: [] for k in KEYS}
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames:
            return series
        present = [k for k in KEYS if k in rd.fieldnames]
        for row in rd:
            for k in present:
                v = (row.get(k) or '').strip()
                if v:
                    try:
                        series[k].append(float(v))
                    except ValueError:
                        pass
    return series


def tail_stats(v, frac=0.2):
    if not v:
        return None
    n = max(1, int(len(v) * frac))
    tail = v[-n:]
    return {
        'n': len(v), 'mean': sum(tail) / len(tail),
        'min': min(tail), 'max': max(tail),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='训练日志（.log/.txt）或 metrics.csv')
    ap.add_argument('--plot', default=None, help='可选：输出 PNG 路径')
    ap.add_argument('--tail', type=float, default=0.2, help='统计最后多少比例的点（默认 0.2）')
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f'找不到 {p}')
    series = parse_csv(p) if p.suffix.lower() == '.csv' else parse_text(p)
    if not any(series.values()):
        raise SystemExit('没有解析到任何字段。确认日志里出现了 fm_* 指标（引导是否真的开启？）')

    print(f'文件: {p}')
    print(f'{"字段":22s} {"点数":>7s} {"尾部均值":>10s} {"最小":>9s} {"最大":>9s}')
    print('-' * 62)
    for k in KEYS:
        st = tail_stats(series[k], args.tail)
        if st:
            print(f'{k:22s} {st["n"]:7d} {st["mean"]:10.4f} {st["min"]:9.4f} {st["max"]:9.4f}')

    # ---- 健康判据 ----
    print('\n=== 判据 ===')
    base = tail_stats(series['fm_loss_base'], args.tail)
    if base is None:
        base = tail_stats(series['fm_loss'], args.tail)
        if base:
            print('[提示] 没有 fm_loss_base，用 fm_loss（含引导）近似——比值会偏乐观。')
    terms = {
        'L_score (×λ_score)': tail_stats(series['fm_guidance_score'], args.tail),
        'L_feas (×λ_safe)': tail_stats(series['fm_safe_feas'], args.tail),
        'L_div (×λ_div)': tail_stats(series['fm_comfort_div'], args.tail),
    }
    if base and base['mean'] > 0:
        for name, st in terms.items():
            if st:
                print(f'  {name:18s} 尾部均值 {st["mean"]:8.4f}   / base {base["mean"]:.4f} '
                      f'= {st["mean"]/base["mean"]:.2f}x')
        print('  ⚠ 任一项 > 1 表示引导已压过主损失；应先降 λ 或延长 warmup。')
        print('    注：上表比值是**损失项本身**（未乘 λ）；实际贡献 = λ × warmup × 该值，'
              '把 λ 代进去才是真正的占比。')
    frac = tail_stats(series['fm_guidance_frac'], args.tail)
    if frac:
        print(f'  区间内样本比例 {frac["mean"]:.2f}'
              + ('  ⚠ 接近 0：时间区间配置错误，引导没生效' if frac['mean'] < 0.05 else ''))
    prod = tail_stats(series['fm_safety_prod'], args.tail)
    if prod:
        print(f'  安全乘积 tail-mean {prod["mean"]:.3f}'
              + ('  ⚠ 长期偏低：安全 hinge 一直激活，会主导梯度' if prod['mean'] < 0.2 else ''))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for k, lab in (('fm_loss_base', 'base loss'), ('fm_guidance_score', 'guidance score'),
                       ('fm_safe_feas', 'safety hinge')):
            v = series[k]
            if v:
                ax.plot(range(len(v)), v, lw=1.1, label=lab)
        ax.set_yscale('log')
        ax.set_xlabel('logged step')
        ax.set_ylabel('loss (log)')
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print('\n已写', args.plot)


if __name__ == '__main__':
    main()
