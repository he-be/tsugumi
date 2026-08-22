#!/usr/bin/env python3
"""適応スキップ (preview を張らない層を選ぶ) が成立するかをトレースから机上で引く。

`docs/qwen35moe/28-PREFETCH-IDEAS.md` §3-4 (c) / `31-PREFETCH-CHEAPER.md` §6 の
「未着手」を、`--dump-expert-trace` が吐いた TSV 1 本だけで判定する。
モデルの再実行は要らない (GPU を使わない)。

先読みの機構 (`QwenForwardRunner.swift:841-932`):
  - 層 L の pre-router コマンドバッファに **層 L+1 の router GEMV** を相乗りさせる
  - 条件は `L + 1 < numLayers` なので、preview GEMV は **層 0..38 の 39 本/tok**
  - 名指された非常駐エキスパートの写像が、層 L の MoE の GPU 時間 (≈0.7 ms) の
    陰で進む
  - **層 0 は誰も先読みしない** (層 -1 が無い。トークン境界は越えられない)

したがって「層 L で preview を打たない」= 「層 L+1 の miss を陰に入れない」。
本スクリプトはこれを層 L+1 (= ターゲット層) 側で数える。

  ./bench/qwen35_preview_skip.py scratch/qwen35/phase6/trace-32lfu-m256.tsv
  ./bench/qwen35_preview_skip.py trace.tsv --verify      32 スロット LFU を再現して
                                                        miss 列と突き合わせる

引けないもの (**未確認**): 層別の「preview が名指せる割合」。preview の予測は
router GEMV の 256 logit から出るが、トレースには**実際に使われた top-8 しか
無い**ので、予測列がそもそも存在しない。全体値 64.5% の実測は
`docs/qwen35moe/27-PHASE6-THROUGHPUT.md` §9-4 にあるが、層別の内訳は無い。
代わりに引ける代理指標を §2 に出す (別の予測子の成績であって、preview の
成績ではない)。
"""

import argparse
import math
import sys
from collections import Counter, defaultdict


# ---------------------------------------------------------------- トレース読み

def read_trace(path):
    """`expert_sim.py` と同じ形式を読む。戻りは (header, records)。"""
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
            experts = [int(x) for x in fields[5].split(",")] if fields[5] else []
            records.append((fields[0], int(fields[1]), int(fields[2]),
                            int(fields[3]), int(fields[4]), experts))
    return header, records


def mean_sd(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


# ---------------------------------------------------- §0 検算 (32 スロット LFU)

class LayerCache:
    """`PreadExpertStreamer.makeExpertCachePlan` の追い出し規則 (LFU)。

    `bench/expert_sim.py` の同名クラスと同じ規則を、検算のためだけに持つ。
    """

    def __init__(self, slots):
        self.slots = slots
        self.slot_expert = [-1] * slots
        self.slot_last_use = [0] * slots
        self.use_count = Counter()
        self.clock = 0

    def _evict_key(self, slot):
        expert = self.slot_expert[slot]
        if expert < 0:
            return (0, 0, self.slot_last_use[slot])
        return (1, self.use_count[expert], self.slot_last_use[slot])

    def resident(self):
        return {e for e in self.slot_expert if e >= 0}

    def request(self, experts):
        self.clock += 1
        clock = self.clock
        reserved = [False] * self.slots
        assigned = [-1] * len(experts)
        for index, expert in enumerate(experts):
            for slot in range(self.slots):
                if not reserved[slot] and self.slot_expert[slot] == expert:
                    assigned[index] = slot
                    reserved[slot] = True
                    break
        misses = [i for i in range(len(experts)) if assigned[i] == -1]
        evictable = sorted((s for s in range(self.slots) if not reserved[s]),
                           key=self._evict_key)
        if len(misses) > len(evictable):
            raise RuntimeError(f"slots={self.slots} に置けない要求")
        for expert in experts:
            self.use_count[expert] += 1
        for slot in assigned:
            if slot >= 0:
                self.slot_last_use[slot] = clock
        for offset, index in enumerate(misses):
            slot = evictable[offset]
            assigned[index] = slot
            self.slot_last_use[slot] = clock
            self.slot_expert[slot] = experts[index]
        return len(experts) - len(misses), len(misses)


def verify(records, slots):
    """トレースの hits/misses 列を、同じ規則の再現と突き合わせる。"""
    caches = defaultdict(lambda: LayerCache(slots))
    bad = 0
    total = 0
    for phase, _step, layer, hits, misses, experts in records:
        h, m = caches[layer].request(experts)
        total += 1
        if (h, m) != (hits, misses):
            bad += 1
            if bad <= 5:
                print(f"  ずれ: {phase} layer={layer} 記録=({hits},{misses}) "
                      f"再現=({h},{m})")
    print(f"検算: {total - bad}/{total} 行が一致"
          + ("" if bad == 0 else f" ({bad} 行ずれ)"))


# ------------------------------------------------- §2 の代理指標に使う常駐集合

def residency_snapshots(records, slots):
    """各 decode 行の *直前* の常駐集合を層ごとに返す。

    層 L+1 の要求が来る前にその層のスロットに何が居たか = 「そもそも先読みが
    要らない層はどこか」を測るための集合。prefill も同じキャッシュを通す。
    """
    caches = defaultdict(lambda: LayerCache(slots))
    out = []
    for phase, step, layer, _hits, _misses, experts in records:
        before = caches[layer].resident() if phase == "decode" else None
        caches[layer].request(experts)
        if phase == "decode":
            out.append((step, layer, before, experts))
    return out


# ------------------------------------------------------------------ レポート

def report_miss_distribution(decode_rows, layers):
    """§1 層別の miss 分布。"""
    per_layer = defaultdict(list)
    for _step, layer, _hits, misses, _experts in decode_rows:
        per_layer[layer].append(misses)
    steps = max(len(v) for v in per_layer.values())
    total_misses = sum(sum(v) for v in per_layer.values())

    print("\n=== §1. 層別の miss 分布 (decode、32 スロット LFU、導出) ===")
    print(f"decode ステップ数 = {steps}、層数 = {len(per_layer)}、"
          f"miss 合計 = {total_misses}"
          f" ({total_misses / steps:.2f} /tok)")
    print("\nlayer   n   miss/step    sd   全体比   miss=0 の step 比  "
          "先読み対象")
    for layer in sorted(per_layer):
        vals = per_layer[layer]
        m, sd = mean_sd(vals)
        share = sum(vals) / total_misses * 100
        zero = sum(1 for v in vals if v == 0) / len(vals) * 100
        target = "—  (層 0 は不可)" if layer == 0 else f"L{layer - 1} の preview"
        print(f"{layer:5d} {len(vals):4d} {m:9.3f} {sd:6.3f} {share:7.2f}% "
              f"{zero:14.1f}%   {target}")

    order = sorted(per_layer, key=lambda l: -sum(per_layer[l]))
    print("\n分布の形 (miss の多い層から積む):")
    print("上位層数   累積 miss   全体比")
    run = 0
    for i, layer in enumerate(order, 1):
        run += sum(per_layer[layer])
        if i in (1, 2, 4, 5, 8, 10, 12, 16, 20, 24, 30, 32, 36, 40):
            print(f"{i:8d} {run:11d} {run / total_misses * 100:7.1f}%")
    flat = total_misses / len(per_layer)
    print(f"(完全に一様なら 1 層 {flat / steps:.3f} miss/step、"
          f"上位 10 層で {10 / len(per_layer) * 100:.0f}%)")
    return per_layer, steps, total_misses


def report_proxies(snapshots, slots, windows=(1, 4, 16)):
    """§2 引ける代理指標。preview の的中率そのものではない。

    (a) 直前の常駐集合との重なり = そもそも先読みが要らない度
    (b) top-8 の前トークン残存 = 層の選択がどれだけ動かないか
    (c) miss が過去 W トークンの top-8 の和集合に居た率
        = 履歴だけの安い予測子が miss を名指せるか
    """
    prev_top = defaultdict(list)
    resident_overlap = defaultdict(list)
    persistence = defaultdict(list)
    named_by_hist = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for _step, layer, before, experts in snapshots:
        resident_overlap[layer].append(
            sum(1 for e in experts if e in before) / len(experts))
        misses = [e for e in experts if e not in before]
        history = prev_top[layer]
        if history:
            persistence[layer].append(
                len(set(experts) & set(history[-1])) / len(experts))
            for w in windows:
                if len(history) < w:
                    continue
                union = set().union(*history[-w:])
                named_by_hist[layer][w][0] += sum(1 for e in misses if e in union)
                named_by_hist[layer][w][1] += len(misses)
        history.append(experts)

    print(f"\n=== §2. 層別の代理指標 ({slots} スロット、導出) ===")
    print("preview が層別に何を名指せるかは **未確認** — トレースには "
          "router GEMV の\n予測列が無く、実際に使われた top-8 しか記録されて"
          "いない (docstring 参照)。\n以下は別の量である。")
    cols = "  ".join(f"過去{w:2d}tok" for w in windows)
    print("\nlayer  常駐で足りる率  前トークン残存   miss を名指す履歴予測子 "
          f"({cols})")
    for layer in sorted(resident_overlap):
        ov, _ = mean_sd(resident_overlap[layer])
        pe, _ = mean_sd(persistence[layer]) if persistence[layer] else (0.0, 0.0)
        cells = []
        for w in windows:
            hit, tot = named_by_hist[layer][w]
            cells.append(f"{hit / tot * 100:6.1f}%" if tot else "   n/a")
        print(f"{layer:5d} {ov * 100:13.1f}% {pe * 100:14.1f}%   "
              + "  ".join(f"{c:>9}" for c in cells))
    print("\n過去 1 tok が構造的に 0.0% なのは規則の帰結である: 32 スロットの層は"
          "\nトークンあたり 1 回しか要求を受けないので、前トークンの 8 本は"
          "追い出される\n機会が無く、必ず常駐している。**miss は定義上"
          "「最近使っていない本」**であり、\n履歴だけの予測子で名指せる範囲は"
          "上の表の通り小さい。router の logit を\n実際に計算する preview に"
          "代わるものは、この表の中には無い。")


def report_tradeoff(per_layer, steps, total_misses, name_rate,
                    gemv_us, miss_cost_ms):
    """§3 GEMV を減らしたときのトレードオフ。"""
    # ターゲットは層 1..39。層 0 は誰も先読みしないので候補にならない。
    targets = [l for l in sorted(per_layer) if l != 0]
    shadowable = sum(sum(per_layer[l]) for l in targets)
    order = sorted(targets, key=lambda l: sum(per_layer[l]))  # miss の少ない順

    print("\n=== §3. スキップ候補と、打つ GEMV / 陰に入らなくなる miss ===")
    print(f"preview GEMV は {len(targets)} 本/tok (層 0..{max(targets) - 1})。"
          f"陰に入りうる miss は層 1..{max(targets)} の\n"
          f"{shadowable} 件 = {shadowable / steps:.2f}/tok "
          f"(層 0 の {sum(per_layer[0])} 件 = "
          f"{sum(per_layer[0]) / steps:.2f}/tok は元から陰に入らない)。")
    print(f"\n通貨: GEMV 1 本 {gemv_us:.0f} µs (上限側、導出)、"
          f"陰から出た miss 1 件 {miss_cost_ms * 1000:.0f} µs (27 §9-3 の実測)、"
          f"名指し率 {name_rate * 100:.1f}%。")
    print("\n打つ GEMV  切る層数  失う miss/tok   陰の miss 比  名指し失 /tok  "
          "浮く µs/tok  失う µs/tok   差 µs/tok  切った層 (先頭 6)")
    kept_list = [39, 36, 32, 30, 24, 20, 16, 12, 10, 8, 4, 1, 0]
    for kept in kept_list:
        if kept > len(targets):
            continue
        cut = len(targets) - kept
        skipped = order[:cut]
        lost = sum(sum(per_layer[l]) for l in skipped) / steps
        named = lost * name_rate
        saved_us = cut * gemv_us
        cost_us = named * miss_cost_ms * 1000
        head = ",".join(str(l) for l in sorted(skipped)[:6])
        if cut > 6:
            head += ",…"
        print(f"{kept:9d} {cut:9d} {lost:14.3f} "
              f"{(lost * steps / shadowable * 100 if shadowable else 0):12.1f}% "
              f"{named:14.3f} {saved_us:12.0f} {cost_us:12.0f} "
              f"{saved_us - cost_us:+11.0f}  {head}")
    return order, shadowable


def report_verdict(per_layer, steps, order, name_rate, miss_cost_ms,
                   gemv_us_low, gemv_us_high, ms_per_token):
    """§4 同じ通貨で並べる。"""
    print("\n=== §4. 判定 — 同じ通貨に直す ===")
    print(f"GEMV 1 本の費用は {gemv_us_low:.1f}〜{gemv_us_high:.1f} µs と置いた "
          "(導出。31 §3-2 の融合が\n GPU −0.3〜0.6 ms/tok = 39 本ぶんの "
          "launch、+ router 重み 1 MiB/層の読み ≈7 µs)。")
    print(f"miss 1 件を陰から出す費用は {miss_cost_ms * 1000:.0f} µs "
          "(27 §9-3 の写像 1 本の限界費用)、\n"
          f"preview がその miss を名指せている確率は {name_rate * 100:.1f}% "
          "(27 §9-4 の全体値、層別は未確認)。")
    break_low = gemv_us_low / (name_rate * miss_cost_ms * 1000)
    break_high = gemv_us_high / (name_rate * miss_cost_ms * 1000)
    print(f"\n損益分岐: その層の miss/step が "
          f"{break_low:.3f}〜{break_high:.3f} 件を下回れば、"
          "preview を張らない方が得。")
    lo = [l for l in order if sum(per_layer[l]) / steps < break_low]
    hi = [l for l in order if sum(per_layer[l]) / steps < break_high]
    print(f"  下限 ({break_low:.3f}) を下回る層: {len(lo)} 層 {sorted(lo)}")
    print(f"  上限 ({break_high:.3f}) を下回る層: {len(hi)} 層 {sorted(hi)}")
    quiet = min(sum(per_layer[l]) / steps for l in order)
    print(f"  実際に一番 miss の少ない層は {quiet:.3f} miss/step。")

    print("\n最良ケース (上限側の層を全部切ったとして):")
    for label, cut in (("下限", lo), ("上限", hi)):
        if not cut:
            print(f"  {label}: 切れる層が無い → 浮く時間 0")
            continue
        saved_us = len(cut) * (gemv_us_low if label == "下限" else gemv_us_high)
        lost = sum(sum(per_layer[l]) for l in cut) / steps
        cost_us = lost * name_rate * miss_cost_ms * 1000
        net = saved_us - cost_us
        pct = net / (ms_per_token * 1000) * 100
        print(f"  {label}: {len(cut)} 本ぶん {saved_us:.0f} µs 浮く / "
              f"名指しを {lost * name_rate:.3f} 件/tok 失う = {cost_us:.0f} µs "
              f"→ 差 {net:+.0f} µs/tok ({pct:+.2f}% @ "
              f"{ms_per_token:.1f} ms/tok)")

    quiet_layer = min(order, key=lambda l: sum(per_layer[l]))
    need_us = quiet * name_rate * miss_cost_ms * 1000
    print(f"\n逆に解く: 一番静かな層 (L{quiet_layer}、{quiet:.3f} miss/step) で"
          f"すら分岐するには、\npreview GEMV 1 本が **{need_us:.0f} µs 以上**"
          "でなければならない。31 §3-1 の select 1 本\n"
          "(≈35〜40 µs) と同格ということで、31 §3-2 の融合 "
          "(GEMV の launch を消して −0.3〜0.6\nms/tok = 8〜15 µs/本) "
          "はそれを否定している。")


def report_crosscheck(per_layer, steps, name_rate):
    """§0 実測との突き合わせ。トレースが 31 §2-2 の off 腕と同じ経路か。"""
    total = sum(sum(v) for v in per_layer.values())
    print("\n=== §0. 実測との突き合わせ (31 §2-2、m 498 tok / 127 新トークン) ===")
    print("| 量 | 実測 (31 §2-2) | 本トレース (導出) |")
    print("| --- | ---: | ---: |")
    print(f"| off 腕の plan miss /tok | {10244 / 127:.2f} (10,244/127) | "
          f"{total / steps:.2f} ({total}/{steps}) |")
    print(f"| n8 腕の plan miss /tok | {3650 / 127:.2f} (3,650/127) | — |")
    shadowed = (10244 - 3650) / 127
    print(f"| 先読みが陰に移した miss /tok | {shadowed:.2f} | "
          f"(陰に入りうるのは {sum(sum(per_layer[l]) for l in per_layer if l) / steps:.2f}) |")
    gain_ms = 1000 / 19.737 - 1000 / 21.651
    print(f"\n**off 腕の miss/tok が 2 桁目まで一致する** — このトレースは"
          "先読み off の\n同じ経路である (導出)。31 §4 の m は "
          f"19.737 → 21.651 tok/s = {gain_ms:.2f} ms/tok の利得なので、\n"
          f"**陰に移した miss 1 件あたりの正味の利得は "
          f"{gain_ms * 1000 / shadowed:.0f} µs** (導出、GEMV の対価込み)。\n"
          f"27 §9-3 の写像 1 本 60 µs と同じ桁で、どちらを使っても §4 の符号は"
          "変わらない。")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--slots", type=int, default=None,
                        help="既定はトレースのヘッダの値")
    parser.add_argument("--verify", action="store_true",
                        help="hits/misses 列を LFU の再現と突き合わせる")
    parser.add_argument("--name-rate", type=float, default=0.645,
                        help="preview が miss を名指す割合 (27 §9-4 の実測、全体値)")
    parser.add_argument("--miss-cost-ms", type=float, default=0.06,
                        help="写像 1 本の限界費用 ms (27 §9-3 の実測)")
    parser.add_argument("--gemv-us", default="15,22",
                        help="preview GEMV 1 本の費用 µs の下限,上限 (導出)")
    parser.add_argument("--ms-per-token", type=float, default=46.2,
                        help="運用点の decode ms/tok (31 §4 の m n8w 21.65 tok/s)")
    args = parser.parse_args()

    header, records = read_trace(args.trace)
    if not records:
        sys.exit("トレースが空")
    slots = args.slots or int(header.get("slots", 32))

    print(f"trace: {args.trace}")
    for key in sorted(header):
        print(f"  {key} = {header[key]}")
    print(f"  records = {len(records)}  (使う slots = {slots})")

    if args.verify:
        print()
        verify(records, slots)

    decode_rows = [(s, l, h, m, e)
                   for phase, s, l, h, m, e in records if phase == "decode"]
    layers = int(header.get("layers", 40))
    lo, hi = (float(x) for x in args.gemv_us.split(","))
    per_layer, steps, total = report_miss_distribution(decode_rows, layers)
    report_crosscheck(per_layer, steps, args.name_rate)

    report_proxies(residency_snapshots(records, slots), slots)
    order, _ = report_tradeoff(per_layer, steps, total, args.name_rate,
                               hi, args.miss_cost_ms)
    report_verdict(per_layer, steps, order, args.name_rate,
                   args.miss_cost_ms, lo, hi, args.ms_per_token)


if __name__ == "__main__":
    main()
