#!/usr/bin/env python3
"""スロット数 vs エキスパートキャッシュのヒット率をトレースから机上で引く。

`TsugumiCLI --dump-expert-trace <path>` が吐いた TSV を読み、
`PreadExpertStreamer.makeExpertCachePlan` と同じ追い出し規則を再現する。
1 回のトレースから任意のスロット数・任意のポリシーの結果が出るので、
実機で条件ごとに測り直す必要がない (PLAN Phase 0 の出口条件)。

  ./bench/expert_sim.py trace.tsv
  ./bench/expert_sim.py trace.tsv --slots 16,32,64,96,128 --policy lfu,lru
  ./bench/expert_sim.py trace.tsv --skew        層ごとの偏りを見る (Phase 6 用)

再現している規則 (PreadExpertStreamer.swift:217-268):
  - 層ごとに独立したキャッシュ
  - 追い出し順は「空きスロット → 使用回数の少ない順 → 最終使用の古い順」
  - 使用回数はヒット・ミスを問わず要求された全エキスパートで増える
  - 使用回数の加算は追い出し候補のソートの *後*

注意: prefill の `avoidingSlots` (タイルのパイプライン用) はトレースに
出ていないので、prefill 側の数字は近似。decode 側は厳密に一致する。
"""

import argparse
import sys
from collections import Counter, defaultdict


class LayerCache:
    """1 層ぶんのスロットキャッシュ。ランタイムの追い出し規則をそのまま写す。"""

    def __init__(self, slots, policy):
        self.slots = slots
        self.policy = policy
        self.slot_expert = [-1] * slots
        self.slot_last_use = [0] * slots
        self.expert_use_count = Counter()
        self.clock = 0

    def _evict_key(self, slot):
        expert = self.slot_expert[slot]
        if self.policy == "lru":
            return (self.slot_last_use[slot],)
        # 空きスロット (-1) は使用回数の比較より前に来る
        if expert < 0:
            return (0, 0, self.slot_last_use[slot])
        return (1, self.expert_use_count[expert], self.slot_last_use[slot])

    def request(self, experts):
        """要求を 1 件処理して (hits, misses) を返す。"""
        self.clock += 1
        clock = self.clock
        reserved = [False] * self.slots
        assigned = [-1] * len(experts)

        # 位置ごとにヒットを探す。同じエキスパートが 2 回来たら 2 枠使う
        # (ランタイムも reserved で同じ挙動になる)。
        for index, expert in enumerate(experts):
            for slot in range(self.slots):
                if not reserved[slot] and self.slot_expert[slot] == expert:
                    assigned[index] = slot
                    reserved[slot] = True
                    break

        misses = [i for i in range(len(experts)) if assigned[i] == -1]
        evictable = sorted(
            (s for s in range(self.slots) if not reserved[s]), key=self._evict_key
        )
        if len(misses) > len(evictable):
            raise RuntimeError(
                f"slots={self.slots} ではこの要求 ({len(experts)} 個) を置けない"
            )

        # 使用回数の加算はソートの後 (ランタイムと同じ順序)。
        for expert in experts:
            self.expert_use_count[expert] += 1
        for slot in assigned:
            if slot >= 0:
                self.slot_last_use[slot] = clock
        for offset, index in enumerate(misses):
            slot = evictable[offset]
            assigned[index] = slot
            self.slot_last_use[slot] = clock
            self.slot_expert[slot] = experts[index]

        return len(experts) - len(misses), len(misses)


def read_trace(path):
    header = {}
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if "=" in line:
                    key, _, value = line[1:].strip().partition("=")
                    header[key] = value
                continue
            fields = line.split("\t")
            if fields[0] == "phase":
                continue
            phase, step, layer = fields[0], int(fields[1]), int(fields[2])
            experts = [int(x) for x in fields[5].split(",")] if fields[5] else []
            records.append((phase, step, layer, experts))
    return header, records


def simulate(records, slots, policy, phases):
    caches = defaultdict(lambda: LayerCache(slots, policy))
    stats = {p: [0, 0] for p in ("prefill", "decode")}
    for phase, _step, layer, experts in records:
        if len(experts) > slots:
            # ランタイムなら precondition で落ちる構成。
            return None
        hits, misses = caches[layer].request(experts)
        if phase in stats:
            stats[phase][0] += hits
            stats[phase][1] += misses
    return {p: tuple(stats[p]) for p in phases}


def decode_tokens(records):
    return len({step for phase, step, _l, _e in records if phase == "decode"})


def skew_report(records, phase):
    """層ごとの偏り。上位 N エキスパートが要求の何 % を占めるか。"""
    per_layer = defaultdict(Counter)
    for rec_phase, _step, layer, experts in records:
        if rec_phase != phase:
            continue
        per_layer[layer].update(experts)
    if not per_layer:
        print(f"({phase} の記録がない)")
        return
    print(f"\n=== {phase} のエキスパート使用の偏り (層ごと) ===")
    print("layer   要求数  上位16  上位32  上位64  上位96  使われた種類")
    tops = (16, 32, 64, 96)
    totals = Counter()
    for layer in sorted(per_layer):
        counts = per_layer[layer]
        total = sum(counts.values())
        totals.update(counts)
        ordered = sorted(counts.values(), reverse=True)
        cover = [sum(ordered[:n]) / total for n in tops]
        print(
            f"{layer:5d} {total:8d}  "
            + "  ".join(f"{c * 100:5.1f}%" for c in cover)
            + f"  {len(counts):5d}"
        )
    total = sum(totals.values())
    ordered = sorted(totals.values(), reverse=True)
    cover = [sum(ordered[:n]) / total for n in tops]
    print(
        "  all "
        + f"{total:8d}  "
        + "  ".join(f"{c * 100:5.1f}%" for c in cover)
        + f"  {len(totals):5d}"
    )
    print("(上位N = 各層でよく使われる N 個が要求全体に占める割合。"
          "Phase 3 の先読みリストと Phase 6 の非一様配分の根拠になる)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--slots", default="8,16,24,32,48,64,80,96,112,128")
    parser.add_argument("--policy", default="lfu,lru")
    parser.add_argument("--phase", default="decode,prefill")
    parser.add_argument("--skew", action="store_true",
                        help="層ごとの使用分布も出す")
    parser.add_argument("--io-ms", type=float, default=None,
                        help="実測の decode I/O ms/token。ミス 1 件あたりのコストに"
                             "換算して他のスロット数へ外挿する")
    parser.add_argument("--ms-per-token", type=float, default=None,
                        help="実測の decode ms/token。--io-ms と併せて tok/s を投影")
    args = parser.parse_args()

    header, records = read_trace(args.trace)
    if not records:
        sys.exit("トレースが空")

    print(f"trace: {args.trace}")
    for key in sorted(header):
        print(f"  {key} = {header[key]}")
    tokens = decode_tokens(records)
    print(f"  records = {len(records)}  decode tokens = {tokens}")

    slot_list = [int(s) for s in args.slots.split(",")]
    policies = args.policy.split(",")
    phases = args.phase.split(",")

    # ミス 1 件あたりの実測コスト。実機の 1 点から他のスロット数へ外挿する。
    miss_cost_ms = None
    if args.io_ms is not None:
        base = simulate(records, int(header.get("slots", 16)), "lfu", ["decode"])
        base_misses = base["decode"][1] if base else 0
        if base_misses and tokens:
            miss_cost_ms = args.io_ms * tokens / base_misses

    for policy in policies:
        print(f"\n=== hit rate / policy={policy} ===")
        head = "slots  常駐率"
        for phase in phases:
            head += f"  {phase:>8} hit   ミス数"
        if miss_cost_ms is not None:
            head += "   io ms/tok"
            if args.ms_per_token is not None:
                head += "   tok/s"
        print(head)
        experts_per_layer = int(header.get("experts", 128))
        for slots in slot_list:
            result = simulate(records, slots, policy, phases)
            if result is None:
                print(f"{slots:5d}  (要求がスロット数を超える構成)")
                continue
            row = f"{slots:5d} {slots / experts_per_layer * 100:5.1f}%"
            for phase in phases:
                hits, misses = result[phase]
                total = hits + misses
                rate = hits / total if total else 0
                row += f"  {rate * 100:9.1f}%  {misses:7d}"
            if miss_cost_ms is not None:
                io = result["decode"][1] * miss_cost_ms / max(tokens, 1)
                row += f"  {io:9.2f}"
                if args.ms_per_token is not None:
                    base_io = args.io_ms
                    projected = args.ms_per_token - base_io + io
                    row += f"  {1000 / projected:7.1f}"
            print(row)

    if miss_cost_ms is not None:
        print(f"\nミス 1 件あたり {miss_cost_ms:.3f} ms (実測 io={args.io_ms} ms/token から逆算)")

    if args.skew:
        skew_report(records, "decode")
        skew_report(records, "prefill")


if __name__ == "__main__":
    main()
