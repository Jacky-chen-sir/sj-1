#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, pickle, argparse, json
from collections.abc import Mapping, Sequence

def try_load_pkl(path):
    # 常见的两种兼容加载方式
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            # 有些用 python2 存的需要指定编码
            f.seek(0)
            return pickle.load(f, encoding="latin1")

def short(v, max_len=120):
    s = repr(v)
    return s if len(s) <= max_len else s[:max_len] + " ..."

def summarize(obj, depth=0, max_items=5):
    indent = "  " * depth
    tname = type(obj).__name__
    if isinstance(obj, Mapping):
        print(f"{indent}{tname} with {len(obj)} keys: {list(obj)[:min(len(obj), max_items)]}")
        # 针对常见关键键，打印子信息
        for k in list(obj)[:max_items]:
            v = obj[k]
            print(f"{indent}  key={k!r}: type={type(v).__name__}")
            if isinstance(v, (Mapping, Sequence)) and not isinstance(v, (str, bytes, bytearray)):
                summarize(v, depth + 2, max_items)
            else:
                print(f"{indent}    sample: {short(v)}")
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        print(f"{indent}{tname} of length {len(obj)}")
        for i, v in enumerate(obj[:min(len(obj), max_items)]):
            print(f"{indent}  [{i}] type={type(v).__name__}")
            if isinstance(v, (Mapping, Sequence)) and not isinstance(v, (str, bytes, bytearray)):
                summarize(v, depth + 2, max_items)
            else:
                print(f"{indent}    sample: {short(v)}")
    else:
        print(f"{indent}{tname}: {short(obj)}")

def main():
    ap = argparse.ArgumentParser(description="Peek a PKL file safely")
    ap.add_argument("path", nargs="?", default="/home/ws/navsim_workspace/dataset/navsim_logs/trainval/2021.05.12.19.36.12_veh-35_00005_00204.pkl")
    ap.add_argument("--items", type=int, default=5, help="Max items to show per container")
    args = ap.parse_args()

    path = args.path
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"== PKL path: {path}")
    print(f"== File size: {os.path.getsize(path)/1024/1024:.2f} MB")

    obj = try_load_pkl(path)
    print(f"== Top-level type: {type(obj).__name__}")

    # 常见结构提示
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"== Top-level keys ({len(keys)}): {keys[:min(len(keys), 20)]}")
        # 如果含有常见字段，打印统计
        for k in ["scenes", "logs", "samples", "instances", "frames", "anns", "calibrations", "sensors"]:
            if k in obj:
                v = obj[k]
                try:
                    ln = len(v)
                except TypeError:
                    ln = "N/A"
                print(f"  - {k}: type={type(v).__name__}, len={ln}")
    elif isinstance(obj, list):
        print(f"== List length: {len(obj)}")

    print("\n== Structured summary (up to {} items per container) ==".format(args.items))
    summarize(obj, depth=0, max_items=args.items)

    # 如果是 list[dict] 或 dict[list/dict]，展示一个更可读的样本 JSON
    def first_record(o):
        if isinstance(o, list) and o and isinstance(o[0], dict):
            return o[0]
        if isinstance(o, dict):
            for v in o.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v[0]
        return None

    rec = first_record(obj)
    if rec is not None:
        print("\n== Pretty sample record ==")
        print(json.dumps(rec, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    main()
