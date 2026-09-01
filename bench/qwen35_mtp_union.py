#!/usr/bin/env python3
"""検証幅 k のエキスパート和集合と、**実際に増える新規取得 (ミス)** を実トレースから引く。

[docs/qwen35moe/29 §5-1](../docs/qwen35moe/29-MTP-PREFETCH-OUTLOOK.md) の 1 番目
(「トレース 1 本からブロックの世界を再現する。モデル再実行不要」)。

sparse MoE の投機検証は、**費用が行数ではなくエキスパートの和集合で伸びる** —
同じエキスパートに routed した行は 1 回の重み読みに相乗りする
([32 §1-4](../docs/qwen35moe/32-NVMAI-ADOPT.md))。NVMAI は同型のモデルで
幅 2 = 12.68 experts/層 (1.585x) を測っている。**本ランタイムの実トレースで
引き直す**のが本スクリプトの前半 (`## 和集合`)。

後半 (`## 新規取得`) は [33 §3-2](../docs/qwen35moe/33-MTP-ACCEPTANCE.md) の
**悲観の置き**を外しにいく。あの表は「和集合が 1.554 倍だからパス費用も 1.554 倍」
と置いているが、[28 §1-1](../docs/qwen35moe/28-PREFETCH-IDEAS.md) の通り
**常駐しているエキスパートを 2 行目が引いても追加費用はゼロ**である
(既定の mmap 腕では費用の通貨はバイトではなく**ホストのページ写像**で、
プランはヒットとして見る)。実際に増えるのは 32 スロットに載っていない
エキスパートの**新規取得 = ミス**だけなので、そちらを直に数える。

追い出し規則は `bench/expert_sim.py` の `LayerCache` をそのまま使う
(= `PreadExpertStreamer.makeExpertCachePlan` の写し。層ごとに独立、
空き → 使用回数の少ない順 → 最終使用の古い順、使用回数はヒット/ミス問わず加算、
加算はソートの後)。decode 側は実機と厳密に一致する。

    ./bench/qwen35_mtp_union.py scratch/qwen35/phase6/trace-32lfu-m256.tsv
    ./bench/qwen35_mtp_union.py <trace> --slots 16,32,48,64 --policy lfu

**モデルの再実行は不要。**すべてトレース 1 本からの導出である。
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_sim import LayerCache, read_trace  # noqa: E402  (読むだけ・変更しない)


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


# --------------------------------------------------------------------------
# 前半: 和集合そのもの (既存の表。壊さない)
# --------------------------------------------------------------------------


def union_table(by_step, layers, steps, widths):
    print(f"\n  {'幅 k':>4}  {'和集合/層':>9}  {'幅 1 比':>7}  {'1 行あたり':>9}  {'標本':>7}")
    base = None
    for k in widths:
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
    return base


def union_pairs(by_step, layers, steps, base, reject):
    """隣接ペア / 無相関ペア / 混合の和集合 (幅 2 だけ)。どちらも**全ペアを尽くす**。"""
    print("\n  幅 2 の和集合を「2 行目の相関」で分ける (どちらも全ペア、乱択なし)")
    print(f"  {'2 行目':<22}  {'和集合/層':>9}  {'幅 1 比':>7}  {'ペア数':>8}")

    # 層ごとにビットマスク化して popcount で数える
    masks = {}
    for layer in layers:
        col = []
        for s in steps:
            m = 0
            for e in by_step[s].get(layer, set()):
                m |= 1 << e
            col.append(m)
        masks[layer] = col

    tot = cnt = 0
    for i in range(len(steps) - 1):
        if steps[i + 1] - steps[i] != 1:
            continue
        for layer in layers:
            tot += (masks[layer][i] | masks[layer][i + 1]).bit_count()
            cnt += 1
    adjacent = tot / cnt
    print(f"  {'隣接 (t, t+1)':<22}  {adjacent:9.3f}  {adjacent / base:7.3f}  {cnt // len(layers):8d}")

    # 無相関: t と、それ以外の全ステップ t' (順不同の全ペアを尽くす = 厳密値)
    tot = cnt = 0
    n = len(steps)
    for layer in layers:
        col = masks[layer]
        for i in range(n):
            mi = col[i]
            for j in range(i + 1, n):
                tot += (mi | col[j]).bit_count()
                cnt += 1
    shuffled = tot / cnt
    label = "無相関 (t, 別の t2)"
    print(f"  {label:<20}  {shuffled:9.3f}  {shuffled / base:7.3f}  "
          f"{cnt // len(layers):8d}")

    mixed = (1 - reject) * adjacent + reject * shuffled
    label = f"混合 {1 - reject:.3f}/{reject:.3f}"
    print(f"  {label:<22}  {mixed:9.3f}  {mixed / base:7.3f}")
    return adjacent, shuffled, mixed


# --------------------------------------------------------------------------
# 後半: 32 スロット LFU に当てたときの「新規取得 (ミス)」
# --------------------------------------------------------------------------


def warm_caches(records, layers, slots, policy):
    """prefill をそのまま流してキャッシュを暖める (expert_sim の simulate と同じ順序)。"""
    caches = {layer: LayerCache(slots, policy) for layer in layers}
    for phase, _step, layer, experts in records:
        if phase != "prefill":
            continue
        caches.setdefault(layer, LayerCache(slots, policy))
        caches[layer].request(experts)
    return caches


def decode_rows(records):
    """decode の (step → layer → 要求リスト)。順序 (= 層の並び) も保つ。"""
    rows: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for phase, step, layer, experts in records:
        if phase == "decode":
            rows[step].append((layer, experts))
    return rows


def run_width1(caches, rows, steps):
    """幅 1: 普通に 1 ステップずつ流す。expert_sim の decode と厳密に同じ。"""
    hits = misses = 0
    for step in steps:
        for layer, experts in rows[step]:
            h, m = caches[layer].request(experts)
            hits += h
            misses += m
    return hits, misses, len(steps)


def run_width2(caches, rows, steps, mode, rng, reject=0.0):
    """幅 2 の検証パス。1 パス = (t, 2 行目) の和集合を 1 回の要求として当てる。

    mode:
      adjacent  2 行目は t+1 (受理された軌道)。t=0,2,4,… と 2 歩ずつ進む
      shuffled  2 行目はランダムな別ステップ (無相関の代理)。同じく 2 歩ずつ
      mixed     2 行目は確率 (1-reject) で t+1、reject でランダム。2 歩ずつ (固定歩)
      walk      受理歩行。受理なら t+1 を使って 2 歩・2 トークン、棄却なら
                ランダム 2 行目で 1 歩・1 トークン
    """
    misses = requested = passes = tokens = 0
    i = 0
    while i + 1 < len(steps):
        t = steps[i]
        if mode == "adjacent":
            other, advance, emitted = steps[i + 1], 2, 2
        elif mode == "shuffled":
            other, advance, emitted = pick_other(rng, steps, t), 2, 2
        elif mode == "mixed":
            accept = rng.random() >= reject
            other = steps[i + 1] if accept else pick_other(rng, steps, t)
            advance, emitted = 2, 2
        elif mode == "walk":
            accept = rng.random() >= reject
            if accept:
                other, advance, emitted = steps[i + 1], 2, 2
            else:
                other, advance, emitted = pick_other(rng, steps, t), 1, 1
        else:
            raise ValueError(mode)

        second = {layer: set(experts) for layer, experts in rows[other]}
        for layer, experts in rows[t]:
            union = sorted(set(experts) | second.get(layer, set()))
            _h, m = caches[layer].request(union)
            misses += m
            requested += len(union)
        passes += 1
        tokens += emitted
        i += advance
    return misses, requested, passes, tokens


def pick_other(rng, steps, t):
    other = rng.choice(steps)
    while other == t:
        other = rng.choice(steps)
    return other


LABEL = {
    "adjacent": "幅2 隣接 (t, t+1)",
    "mixed":    "幅2 混合 (固定 2 歩)",
    "shuffled": "幅2 無相関 (t, 乱択)",
    "walk":     "幅2 受理歩行",
}


def slot_table(records, layers, steps, rows, slots_list, policy, reject, seed, reps):
    nlayer = len(layers)
    print(f"\n## 新規取得 (ミス) — policy={policy}、prefill で暖めてから decode")
    print("   幅 1 = 1 ステップ / 幅 2 = 1 検証パス (和集合を 1 回の要求として当てる)")
    print("   「ミス」= 32 スロットに載っていないエキスパートの新規取得 = ホストのページ写像")
    print(f"   乱択を含む変種は seed={seed} × {reps} 反復")

    results = {}
    for slots in slots_list:
        base_caches = warm_caches(records, layers, slots, policy)

        c1 = copy.deepcopy(base_caches)
        hits1, miss1, nsteps = run_width1(c1, rows, steps)
        per_step = miss1 / nsteps
        row = {"hits1": hits1, "miss1": miss1, "steps": nsteps,
               "per_step": per_step, "per_layer": per_step / nlayer,
               "union_per_layer": (hits1 + miss1) / nsteps / nlayer}

        for mode in ("adjacent", "mixed", "shuffled", "walk"):
            reps_here = 1 if mode == "adjacent" else reps
            tot_miss = tot_req = tot_pass = tot_tok = 0
            for r in range(reps_here):
                rng = random.Random(seed + 1000 * r + (7 if mode == "walk" else 0))
                caches = copy.deepcopy(base_caches)
                m, q, p, tok = run_width2(caches, rows, steps, mode, rng, reject)
                tot_miss += m
                tot_req += q
                tot_pass += p
                tot_tok += tok
            miss_pass = tot_miss / tot_pass
            row[mode] = {
                "miss_per_pass": miss_pass,
                "miss_per_layer": miss_pass / nlayer,
                "union_per_layer": tot_req / tot_pass / nlayer,
                "miss_rate": tot_miss / tot_req,
                "tokens_per_pass": tot_tok / tot_pass,
                "miss_per_token": tot_miss / tot_tok,
                "passes": tot_pass,
                "ratio": miss_pass / per_step,
                "ratio_per_token": (tot_miss / tot_tok) / per_step,
            }
        results[slots] = row

    for slots in slots_list:
        row = results[slots]
        print(f"\n  --- slots={slots} " + "-" * 62)
        print(f"  {'':<22}  {'和集合/層':>9}  {'ミス/層':>8}  {'ミス率':>7}  "
              f"{'ミス/pass':>9}  {'幅1 step 比':>11}  {'tok/pass':>8}  "
              f"{'ミス/tok':>8}  {'幅1 比':>7}  {'標本pass':>8}")
        print(f"  {'幅1 (1 ステップ)':<20}  {row['union_per_layer']:9.3f}  "
              f"{row['per_layer']:8.3f}  {row['per_layer'] / row['union_per_layer'] * 100:6.1f}%  "
              f"{row['per_step']:9.2f}  {1.0:11.3f}  {1.0:8.3f}  "
              f"{row['per_step']:8.2f}  {1.0:7.3f}  {row['steps']:8d}")
        for mode in ("adjacent", "mixed", "shuffled", "walk"):
            d = row[mode]
            print(f"  {LABEL[mode]:<22}  {d['union_per_layer']:9.3f}  "
                  f"{d['miss_per_layer']:8.3f}  {d['miss_rate'] * 100:6.1f}%  "
                  f"{d['miss_per_pass']:9.2f}  {d['ratio']:11.3f}  "
                  f"{d['tokens_per_pass']:8.3f}  {d['miss_per_token']:8.2f}  "
                  f"{d['ratio_per_token']:7.3f}  {d['passes']:8d}")
        a = row["adjacent"]
        d_union = a["union_per_layer"] - row["union_per_layer"]
        d_miss = a["miss_per_layer"] - row["per_layer"]
        print(f"  2 行目が増やす分 (隣接): 和集合 +{d_union:.3f}/層 のうちミスは "
              f"+{d_miss:.3f}/層 = {d_miss / d_union * 100:.1f}% "
              f"(幅 1 の 1 行目は {row['per_layer'] / row['union_per_layer'] * 100:.1f}%)")

    print(f"\n  ## 3 の答え — 幅 2 の 1 パスあたりミス数 ÷ 幅 1 の 1 ステップあたりミス数")
    print(f"\n  {'slots':>5}  {'和集合比':>8}  {'隣接':>7}  {'混合':>7}  {'無相関':>7}  {'受理歩行':>8}")
    for slots in slots_list:
        row = results[slots]
        u = row["adjacent"]["union_per_layer"] / row["union_per_layer"]
        print(f"  {slots:5d}  {u:8.3f}  {row['adjacent']['ratio']:7.3f}  "
              f"{row['mixed']['ratio']:7.3f}  {row['shuffled']['ratio']:7.3f}  "
              f"{row['walk']['ratio']:8.3f}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--widths", default="1,2,3,4,5,6")
    ap.add_argument("--slots", default="16,32,48,64",
                    help="キャッシュに当てるスロット数 (運用点は 32)")
    ap.add_argument("--policy", default="lfu")
    ap.add_argument("--reject", type=float, default=0.213,
                    help="ドラフト 2 行目が棄却される確率 (1 - P1、33 §2-1)")
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--reps", type=int, default=40,
                    help="乱択を含む変種の反復数")
    args = ap.parse_args()

    by_step, layers = read(args.trace)
    steps = sorted(by_step)
    if not steps:
        raise SystemExit("decode の記録が無い")
    print(f"## {args.trace}")
    print(f"   decode {len(steps)} ステップ × {len(layers)} 層")

    base = union_table(by_step, layers, steps,
                       [int(x) for x in args.widths.split(",")])
    union_pairs(by_step, layers, steps, base, args.reject)

    header, records = read_trace(args.trace)
    rows = decode_rows(records)
    slot_list = [int(s) for s in args.slots.split(",")]
    for policy in args.policy.split(","):
        results = slot_table(records, layers, steps, rows, slot_list, policy,
                             args.reject, args.seed, args.reps)

        # expert_sim との突き合わせ ([27 §6-1] の 32/lfu = 74.9% と一致するはず)
        print(f"\n## 幅 1 の突き合わせ (bench/expert_sim.py の decode と同じ数字か) "
              f"policy={policy}")
        print(f"  {'slots':>5}  {'decode hit':>10}  {'hits/要求':>16}")
        for slots in slot_list:
            r = results[slots]
            total = r["hits1"] + r["miss1"]
            print(f"  {slots:5d}  {r['hits1'] / total * 100:9.1f}%  "
                  f"{r['hits1']:8d}/{total:<7d}")
    print(f"\n  (トレース header: slots={header.get('slots')} policy={header.get('policy')} "
          f"topK={header.get('topK')} experts={header.get('experts')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
