#!/usr/bin/env python3
"""検証幅 k のエキスパート和集合を、実トレースから机上で引く。

[docs/qwen35moe/29 §5-1](../docs/qwen35moe/29-MTP-PREFETCH-OUTLOOK.md) の 1 番目
(「トレース 1 本からブロックの世界を再現する。モデル再実行不要」)。

sparse MoE の投機検証は、**費用が行数ではなくエキスパートの和集合で伸びる** —
同じエキスパートに routed した行は 1 回の重み読みに相乗りする
([32 §1-4](../docs/qwen35moe/32-NVMAI-ADOPT.md))。NVMAI は同型のモデルで
幅 2 = 12.68 experts/層 (1.585x) を測っている。**本ランタイムの実トレースで
引き直す**のが本スクリプト。

    ./bench/qwen35_mtp_union.py scratch/qwen35/phase6/trace-32lfu-m256.tsv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict


def read(path: str):
    """decode の (step, layer) → top-k エキスパート集合。"""
    by_step: dict[int, dict[int, set[int]]] = defaultdict(dict)
    layers = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[0] == "phase" or f[0] != "decode":
                continue
            step, layer = int(f[1]), int(f[2])
            experts = {int(x) for x in f[5].split(",")} if f[5] else set()
            by_step[step].setdefault(layer, set()).update(experts)
            layers.add(layer)
    return by_step, sorted(layers)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--widths", default="1,2,3,4,5,6")
    args = ap.parse_args()

    by_step, layers = read(args.trace)
    steps = sorted(by_step)
    if not steps:
        raise SystemExit("decode の記録が無い")
    print(f"## {args.trace}")
    print(f"   decode {len(steps)} ステップ × {len(layers)} 層")

    base = None
    print(f"\n  {'幅 k':>4}  {'和集合/層':>9}  {'幅 1 比':>7}  {'1 行あたり':>9}  {'標本':>7}")
    for k in [int(x) for x in args.widths.split(",")]:
        total = 0
        count = 0
        for i in range(len(steps) - k + 1):
            window = steps[i:i + k]
            if window[-1] - window[0] != k - 1:      # 連続していない窓は捨てる
                continue
            for layer in layers:
                union: set[int] = set()
                for s in window:
                    union |= by_step[s].get(layer, set())
                total += len(union)
                count += 1
        if not count:
            continue
        mean = total / count
        base = mean if base is None else base
        print(f"  {k:>4}  {mean:9.2f}  {mean / base:7.3f}  {mean / k:9.2f}  {count:7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
