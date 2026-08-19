#!/usr/bin/env python3
"""N1 (37 §3) — 48 スロットの台帳。四則のみ。"""
import re, csv, pathlib, statistics as st
D = pathlib.Path("/Users/mh/LLM/turbo-fieldfare/bench/logs/39_n1")
SLOTS = [16, 24, 32, 48]

print("=" * 76)
print("1. costProbe の c 曲線 — スロット軸 (画像 A / thinking on / 144 位置)")
print("=" * 76)
print(f"{'slots':>6} {'base ms/tok':>12} " + " ".join(f"k={k}".rjust(7) for k in (1,2,3,4,6,8)))
print(f"{'':>6} {'':>12} " + " ".join("step/base".rjust(7) for _ in range(6)))
cs = {}
for s in SLOTS:
    t = (D / f"cost_{s}slots.log").read_text()
    base = float(re.search(r"cost: decode=([\d.]+) ms/tok", t).group(1))
    steps = {int(k): float(v) for k, v in re.findall(r"k=(\d+) \d+ ms/block = ([\d.]+) decode", t)}
    hit = float(re.search(r"decode hit=([\d.]+)%", t).group(1))
    cs[s] = (base, steps, hit)
    print(f"{s:6} {base:12.2f} " + " ".join(f"{steps[k]:7.2f}" for k in (1,2,3,4,6,8)))
print()
print(f"{'slots':>6} " + " ".join(f"c(k={k})".rjust(8) for k in (2,3,4,6,8)) + f" {'hit%':>7}")
for s in SLOTS:
    base, steps, hit = cs[s]
    row = [(steps[k] - 1) / (k - 1) for k in (2,3,4,6,8)]
    print(f"{s:6} " + " ".join(f"{c:8.3f}" for c in row) + f" {hit:7.1f}")
print("\n  36 §8a の同じ量 (旧画像 IMG_2113、消失): 16=0.673 / 24=0.557 / 32=0.453 (k=4)")
print(f"  本書 (画像 A):                          " +
      " / ".join(f"{s}={(cs[s][1][4]-1)/3:.3f}" for s in SLOTS))

print("\n" + "=" * 76)
print("2. 本番スイープ — 最適 bs はスロット数でどう動くか (運用点 temp 1.0)")
print("=" * 76)
def load(tag, s):
    rows = list(csv.DictReader((D / f"sweep_{tag}_{s}.tsv").open(), delimiter="\t"))
    return [r for r in rows if r["status"] == "OK"]

for tag in ("A", "B"):
    print(f"\n--- 画像 {tag} ---")
    print(f"{'slots':>6} {'base t/s':>9} " + " ".join(f"bs={b}".rjust(16) for b in (2,3,4)) + f" {'最適':>6}")
    print(f"{'':>6} {'':>9} " + " ".join("t/s (base比)".rjust(16) for _ in range(3)))
    for s in SLOTS:
        rows = load(tag, s)
        base = next((float(r["tok_s"]) for r in rows if r["slot"] == "head_base"), None)
        by = {}
        for r in rows:
            if r["slot"] == "head_base": continue
            b = int(r["bs"])
            if b > 0: by[b] = float(r["tok_s"])
            elif base is None: base = float(r["tok_s"])
        if not by: continue
        best = max(by, key=lambda b: by[b])
        cells = " ".join(f"{by[b]:7.2f} ({by[b]/base:5.3f})" if b in by else " " * 16
                         for b in (2,3,4))
        print(f"{s:6} {base:9.2f} {cells} {'bs='+str(best):>6}")

print("\n" + "=" * 76)
print("3. 48 スロットの MTP off decode — 35 §6(a) の入力 (register 明示: 日本語散文)")
print("=" * 76)
print(f"{'img':>4} {'slots':>6} {'ms/tok':>8} {'t/s':>7} {'decode_io ms/tok':>17} {'hit%':>7}")
for tag in ("A", "B"):
    for s in SLOTS:
        rows = load(tag, s)
        r0 = next((r for r in rows if r["slot"] == "head_base" or r["bs"] == "0"), None)
        if not r0: continue
        ms = float(r0["ms_per_tok"]); io = float(r0["decode_io"]); new = float(r0["new"])
        print(f"{tag:>4} {s:6} {ms:8.2f} {float(r0['tok_s']):7.2f} "
              f"{io/new*1000:17.2f} {float(r0['decode_hit']):7.1f}")
print("\n  35 §6(a) が参照実装に置いた値: llama.cpp 26.9 ms/tok (register 未定義)")
