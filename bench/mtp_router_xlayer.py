#!/usr/bin/env python3
"""エキスパート番号の層間相関を expert トレースから引く (29-M8-B-PROBE §0-2)。

先読みの一番安い形は「層 L が選んだ集合をそのまま層 L+1 の先読みに使う」で、
これなら router を先行適用する配管が要らない。当たるかどうかはトレースだけで
決まるので、実機を回す前にここで潰す。

  ./bench/mtp_router_xlayer.py scratch/mtp-traces/math-t0.tsv

比べる相手は 2 つ:
  - ランダム (top-k / エキスパート数)。これと差が無ければ層間に情報は無い。
  - 同じ層の直前ステップ。時間方向の相関はあるので、その水準が目安になる。
"""

import argparse
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from expert_sim import read_trace  # noqa: E402


def decode_steps(records):
    """decode の記録を [step][layer] -> set(experts) に畳む。"""
    steps = defaultdict(dict)
    for phase, step, layer, experts in records:
        if phase == "decode":
            steps[step][layer] = set(experts)
    return [steps[s] for s in sorted(steps)]


def overlap(pairs):
    """(予測集合, 実集合) の列から「実集合のうち予測に入っていた率」を出す。"""
    named = total = 0
    for predicted, actual in pairs:
        named += len(predicted & actual)
        total += len(actual)
    return named / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--layers", type=int, default=3,
                        help="いくつ先の層まで見るか (既定 3)")
    args = parser.parse_args()

    header, records = read_trace(args.trace)
    steps = decode_steps(records)
    layers = int(header["layers"])
    experts = int(header["experts"])
    topk = len(next(iter(steps[0].values()))) if steps and steps[0] else 8
    chance = topk / experts

    print(f"{args.trace}: decode {len(steps)} step / {layers} 層 / "
          f"{experts} エキスパート / top-{topk}")
    print(f"ランダムに当たる率: {chance:6.1%}")

    for d in range(1, args.layers + 1):
        pairs = [(st[l], st[l + d])
                 for st in steps
                 for l in range(layers - d)
                 if l in st and l + d in st]
        print(f"層 L の集合が層 L+{d} を当てる率: {overlap(pairs):6.1%}")

    for d in (1, 2):
        pairs = [(steps[i - d][l], steps[i][l])
                 for i in range(d, len(steps))
                 for l in range(layers)
                 if l in steps[i - d] and l in steps[i]]
        print(f"直前 {d} ステップの同層が当てる率:  {overlap(pairs):6.1%}")


if __name__ == "__main__":
    main()
