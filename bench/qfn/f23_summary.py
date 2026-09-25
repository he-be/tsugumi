"""docs/qfn-prefill/12 §3: F2 (A) / F3 (B) の模擬の壁時計を記録のラウンド分布に掛けて 1 ターンの prefill 合計を出す (導出)。

    python3 bench/qfn/f23_summary.py [label]      # label 既定 f23 (bench/qfn/results/2026-09-25-pair-<label>/)

1 本の run は 2 チャンク。t1 = チャンク 0 の終わり (1 チャンクのラウンド)、t2 = 2 チャンクの合計。
c チャンクのラウンドは t2 + (c - 2)(t2 - t1) (c >= 2)。分布は formula.py --turns の記録 (1 チャンク 55 本、2 チャンク 10 本、3 チャンク 3 本)。
各条件は n 本の中央値。式の値 (格子を絞った式、12 §3-4 のずれ) も並べる。
"""
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
label = sys.argv[1] if len(sys.argv) > 1 else "f23"
d = f"{R}/2026-09-25-pair-{label}"
DIST = {1: 55, 2: 10, 3: 3}

runs = defaultdict(list)
for f in sorted(glob.glob(f"{d}/[AB]-*-[0-9].log")):
    if f.endswith(".m6.log"):
        continue
    s = open(f).read()
    m0 = re.search(r"^chunk 0: [\d.]+ s .*?, ([\d.]+) s since the start", s, re.M)
    mt = re.search(r"^total: 2 chunks ([\d.]+) s", s, re.M)
    if not (m0 and mt) or "GUARD: exit 0" not in s:
        print(f"skip {os.path.basename(f)} (incomplete)")
        continue
    cfg = re.sub(r"-\d+\.log$", "", os.path.basename(f))
    m6 = f[:-4] + ".m6.log"
    sw = [int(x) for x in re.findall(r"Swapouts delta peak (\d+)", s + (open(m6).read() if os.path.exists(m6) else ""))]
    runs[cfg].append((float(m0.group(1)), float(mt.group(1)), max(sw)))

def turn(t1, t2):
    return sum(n * (t1 if c == 1 else t2 + (c - 2) * (t2 - t1)) for c, n in DIST.items())

def key(c):
    s, p, r = c.split("-")[0], float(c.split("-")[1]), c.split("-r")[1]
    return (p != 0, s, r != "0", p)

print(f"{'条件':16s} {'n':>2s} {'t1 s (範囲)':>22s} {'t2 s (範囲)':>22s} {'1 ターン s':>10s} {'範囲':>16s}  Swapouts")
out = {}
for c in sorted(runs, key=key):
    v = runs[c]
    t1s, t2s = [x[0] for x in v], [x[1] for x in v]
    turns = sorted(turn(a, b) for a, b, _ in v)
    t1, t2 = st.median(t1s), st.median(t2s)
    out[c] = turn(t1, t2)
    print(f"{c:16s} {len(v):2d} {t1:7.3f} ({min(t1s):.3f}-{max(t1s):.3f}) {t2:7.3f} ({min(t2s):.3f}-{max(t2s):.3f}) "
          f"{out[c]:10.1f} ({turns[0]:.1f}-{turns[-1]:.1f})  {max(x[2] for x in v)}")
for shape in "AB":
    for res, name in (("11.1", "常駐"), ("0", "冷")):
        cs = {c: t for c, t in out.items() if c.startswith(shape + "-") and c.endswith("-r" + res) and float(c.split("-")[1]) != 0}
        if cs:
            b = min(cs, key=cs.get)
            print(f"{shape} {name} の最小: {b} {cs[b]:.1f} s")
