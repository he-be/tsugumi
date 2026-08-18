#!/usr/bin/env python3
"""MTP の verify ブロックが読むエキスパートを、decode トレースから机上で引く。

`expert_sim.py` は 1 トークン 1 要求 (decode) を再生する。MTP の verify ブロックは
k トークンを 1 forward で流すので、層ごとの要求は **k トークンの和集合 1 件** になる。
同じトレースからその要求列を作り直し、スロット数・ポリシー・プール構成を振る。

  ./bench/mtp_expert_sim.py trace.tsv --k 1,2,4,8
  ./bench/mtp_expert_sim.py trace.tsv --k 4 --pool per-layer,global,belady

出す数字は **産出トークンあたりのミス数**。decode の同じ数字と直接比べられる。
ブロックは受理長 a のぶんしか進まないので、ラウンドあたりのミスを (a+1) で割る。
"""

import argparse
import sys
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from expert_sim import LayerCache, read_trace  # noqa: E402

EXPERT_MB = 3.719168


def decode_stream(records):
    """decode の記録を [step][layer] -> experts に畳む。step 順・layer 順。"""
    steps = defaultdict(dict)
    for phase, step, layer, experts in records:
        if phase != "decode":
            continue
        steps[step][layer] = experts
    return [steps[s] for s in sorted(steps)]


def block_requests(stream, k, advance):
    """ラウンドごとの (層 -> 和集合) を返す。advance は平均受理長 + 1。"""
    rounds = []
    position = 0.0
    index = 0
    total = len(stream)
    while index < total:
        window = stream[index:index + k]
        merged = {}
        for token in window:
            for layer, experts in token.items():
                seen = merged.setdefault(layer, [])
                for expert in experts:
                    if expert not in seen:
                        seen.append(expert)
        rounds.append(merged)
        position += advance
        index = int(position)
    return rounds


class GlobalCache:
    """層をまたいで共有する 1 本のプール。キーは (layer, expert)。"""

    def __init__(self, slots, policy):
        self.inner = LayerCache(slots, policy)

    def request(self, layer, experts):
        return self.inner.request([(layer << 12) | e for e in experts])


def simulate_rounds(rounds, slots, policy, pool):
    """(要求数, ミス数, 置けなかったラウンド数) を返す。"""
    unplaceable = 0
    requests = 0
    misses = 0
    if pool == "global":
        cache = GlobalCache(slots * 30, policy)
        for layer_map in rounds:
            for layer in sorted(layer_map):
                experts = layer_map[layer]
                requests += len(experts)
                misses += cache.request(layer, experts)[1]
        return requests, misses, unplaceable
    caches = defaultdict(lambda: LayerCache(slots, policy))
    for layer_map in rounds:
        for layer in sorted(layer_map):
            experts = layer_map[layer]
            if len(experts) > slots:
                unplaceable += 1
                # 実機は per-tile 経路に落ちてスロットを返しながら回す。
                # 近似として先頭 slots 個だけ通す。
                experts = experts[:slots]
            requests += len(experts)
            misses += caches[layer].request(experts)[1]
    return requests, misses, unplaceable


def belady_misses(rounds, slots):
    """層ごとに最適置換 (次に使うのが最も遠いものを追い出す) を回した下限。"""
    per_layer_seq = defaultdict(list)
    for index, layer_map in enumerate(rounds):
        for layer, experts in layer_map.items():
            per_layer_seq[layer].append((index, experts))
    total_requests = 0
    total_misses = 0
    for layer, sequence in per_layer_seq.items():
        # 各要求位置での「次に出てくる位置」を前計算。
        future = defaultdict(list)
        for order, (_index, experts) in enumerate(sequence):
            for expert in experts:
                future[expert].append(order)
        cursor = {expert: 0 for expert in future}
        resident = set()
        for order, (_index, experts) in enumerate(sequence):
            for expert in experts:
                while (cursor[expert] < len(future[expert])
                       and future[expert][cursor[expert]] <= order):
                    cursor[expert] += 1
            total_requests += len(experts)
            need = [e for e in experts if e not in resident]
            total_misses += len(need)
            for expert in need:
                if len(resident) < slots:
                    resident.add(expert)
                    continue
                # 追い出し候補: 常駐かつ今回の要求に入っていないもののうち、
                # 次の使用が最も遠い (または二度と来ない) もの。
                victim = None
                victim_next = -1
                for candidate in resident:
                    if candidate in experts:
                        continue
                    nxt = (future[candidate][cursor[candidate]]
                           if cursor[candidate] < len(future[candidate]) else 1 << 30)
                    if nxt > victim_next:
                        victim_next = nxt
                        victim = candidate
                if victim is None:
                    continue
                resident.discard(victim)
                resident.add(expert)
    return total_requests, total_misses


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--k", default="1,2,4,8")
    parser.add_argument("--slots", default="32,48,64,80,96,112")
    parser.add_argument("--policy", default="lfu")
    parser.add_argument("--pool", default="per-layer,global")
    parser.add_argument("--accept", default="",
                        help="k ごとの平均受理長 a (例 1:0,2:0.93,4:1.947,8:3.0)。"
                             "未指定は 14-M3.5 の bs=4 実測から線形に置く")
    parser.add_argument("--gbps", type=float, default=5.0,
                        help="実効読み出し帯域 GB/s (18-M4.6 §2 の実測は 5.0)")
    args = parser.parse_args()

    _header, records = read_trace(args.trace)
    stream = decode_stream(records)
    if not stream:
        sys.exit("decode の記録がない")

    accept = {1: 0.0, 2: 0.93, 3: 1.50, 4: 1.947, 5: 2.35, 6: 2.65, 8: 3.10}
    for item in args.accept.split(",") if args.accept else []:
        key, _, value = item.partition(":")
        accept[int(key)] = float(value)

    ks = [int(x) for x in args.k.split(",")]
    slot_list = [int(x) for x in args.slots.split(",")]
    pools = args.pool.split(",")
    ms_per_miss = EXPERT_MB / 1024 / args.gbps * 1000

    print(f"trace: {args.trace}  decode tokens = {len(stream)}")
    print(f"ミス 1 件 = {EXPERT_MB:.3f} MB = {ms_per_miss:.3f} ms @ {args.gbps} GB/s")

    # decode の基準線 (k=1 は 1 トークン 1 要求そのもの)。
    for policy in args.policy.split(","):
        for pool in pools:
            print(f"\n=== policy={policy} pool={pool} ===")
            print("  k  a+1  slots   要求/産出tok  ミス/産出tok   MB/産出tok   ms/産出tok"
                  "   置けない層")
            for k in ks:
                advance = accept.get(k, 0.0) + 1.0
                rounds = block_requests(stream, k, advance)
                produced = len(rounds) * advance
                for slots in slot_list:
                    requests, misses, unplaceable = simulate_rounds(
                        rounds, slots, policy, pool)
                    print(f"{k:3d} {advance:4.2f} {slots:6d} "
                          f"{requests / produced:13.1f} {misses / produced:13.2f} "
                          f"{misses / produced * EXPERT_MB:12.1f} "
                          f"{misses / produced * ms_per_miss:12.1f} {unplaceable:11d}")

    if "belady" in pools or True:
        print("\n=== 最適置換 (Belady) の下限、層ごとプール ===")
        print("  k  a+1  slots   ミス/産出tok   ms/産出tok")
        for k in ks:
            advance = accept.get(k, 0.0) + 1.0
            rounds = block_requests(stream, k, advance)
            produced = len(rounds) * advance
            for slots in slot_list:
                _requests, misses = belady_misses(rounds, slots)
                print(f"{k:3d} {advance:4.2f} {slots:6d} {misses / produced:13.2f} "
                      f"{misses / produced * ms_per_miss:12.1f}")


if __name__ == "__main__":
    main()
