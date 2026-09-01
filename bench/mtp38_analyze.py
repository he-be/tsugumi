#!/usr/bin/env python3
"""38 — register 2x2 を運用点 (temp 1.0) で 3 反復した結果の集計。
36 §11a (temp 0 / n=1) と並べる。四則と標本標準偏差のみ。"""
import re, sys, math, statistics as st
from collections import Counter, defaultdict
sys.path.insert(0, "/Users/mh/LLM/tsugumi/bench")

OUT = "/Users/mh/LLM/tsugumi/bench/logs/38_reg_t1"
CELLS = ["ja_prose", "ja_bullets", "en_prose", "en_bullets"]
# 36 §11a の temp 0 / n=1 の値 (hit%, tok/s)
T0 = {"ja_prose": (93.2, 25.30), "ja_bullets": (90.3, 23.53),
      "en_prose": (87.8, 22.69), "en_bullets": (81.0, 19.75)}

def footer(tag):
    try:
        t = open(f"{OUT}/{tag}.err", encoding="utf-8").read()
    except FileNotFoundError:
        return None
    def g(p, cast=float):
        m = re.search(p, t)
        return cast(m.group(1)) if m else None
    return dict(
        toks=g(r"tok/s=([\d.]+)"),
        hit=g(r"decode hit=([\d.]+)%"),
        new=g(r"new=(\d+)tok", int),
        io_ms=g(r"decode/tok io=([\d.]+)ms"),
        dio=g(r"decode hit=[\d.]+% \d+/\d+ io=([\d.]+)s"),
    )

print("=" * 78)
print("38. register 2x2 @ 運用点 (temp 1.0 / top-k 64 / top-p 0.95) — 3 反復")
print("=" * 78)
print(f"\n{'cell':12} {'tok/s (3反復)':>26} {'平均':>7} {'sd':>6} {'temp0 n=1':>10} {'差':>7}")
res = {}
for c in CELLS:
    vs = [footer(f"{c}_r{i}") for i in (1, 2, 3)]
    ts = [v["toks"] for v in vs if v and v["toks"]]
    if len(ts) < 2:
        print(f"{c:12} (データ不足)"); continue
    m, s = st.mean(ts), st.stdev(ts)
    res[c] = (m, s, ts)
    print(f"{c:12} {', '.join(f'{x:.2f}' for x in ts):>26} {m:7.2f} {s:6.2f} "
          f"{T0[c][1]:10.2f} {m-T0[c][1]:+7.2f}")

print(f"\n{'cell':12} {'decode hit% (3反復)':>26} {'平均':>7} {'sd':>6} {'temp0 n=1':>10} {'差':>7}")
hits = {}
for c in CELLS:
    vs = [footer(f"{c}_r{i}") for i in (1, 2, 3)]
    hs = [v["hit"] for v in vs if v and v["hit"] is not None]
    if len(hs) < 2: continue
    m, s = st.mean(hs), st.stdev(hs)
    hits[c] = (m, s, hs)
    print(f"{c:12} {', '.join(f'{x:.1f}' for x in hs):>26} {m:7.2f} {s:6.2f} "
          f"{T0[c][0]:10.1f} {m-T0[c][0]:+7.2f}")

print("\n--- 判定 1: 最良 (ja_prose) と最悪 (en_bullets) の差は運用点でも残るか ---")
if "ja_prose" in res and "en_bullets" in res:
    a, b = res["ja_prose"], res["en_bullets"]
    ha, hb = hits["ja_prose"], hits["en_bullets"]
    # 独立 2 標本 (Welch) の 95% 区間。n=3 なので t(自由度~4)=2.776 で保守的に。
    def welch(x, y):
        d = x[0] - y[0]
        se = math.sqrt(x[1]**2/3 + y[1]**2/3)
        return d, 2.776*se
    d, ci = welch(a, b)
    print(f"  t/s   差 {d:+.2f} ± {ci:.2f}  ({a[0]:.2f} 対 {b[0]:.2f}, "
          f"{d/b[0]*100:+.1f}%)  -> {'有意' if abs(d) > ci else '有意でない'}")
    hd, hci = welch(ha, hb)
    print(f"  hit   差 {hd:+.2f}pt ± {hci:.2f}pt  ({ha[0]:.1f}% 対 {hb[0]:.1f}%)"
          f"  -> {'有意' if abs(hd) > hci else '有意でない'}")
    print(f"  参考: temp 0 / n=1 の主張は t/s -21.9%, hit -12.2pt")

print("\n--- 判定 2: 熱ドリフト (先頭 ja_prose_r1 と末尾 ja_prose_drift は同一条件) ---")
r1, dr = footer("ja_prose_r1"), footer("ja_prose_drift")
if r1 and dr and r1["toks"] and dr["toks"]:
    dd = (dr["toks"] - r1["toks"]) / r1["toks"] * 100
    print(f"  先頭 {r1['toks']:.2f} t/s -> 末尾 {dr['toks']:.2f} t/s  ({dd:+.2f}%)")
    print(f"  hit  {r1['hit']:.1f}% -> {dr['hit']:.1f}%")
    print(f"  -> {'ドリフトあり。セル間の差はこの幅を超えるものだけ読む' if abs(dd) > 2 else 'ドリフトは 2% 未満。セル間の差を読んでよい'}")

print("\n--- 判定 3: 層あたり実効エキスパート数 2^H (運用点) ---")
try:
    from expert_sim import read_trace
    print(f"  {'cell':12} {'H(bits)':>8} {'2^H':>7} {'temp0 (36 §12b)':>16}")
    T0H = {"ja_prose": 23.5, "en_bullets": 41.8, "ja_bullets": None, "en_prose": None}
    for c in CELLS:
        try:
            h, rec = read_trace(f"{OUT}/tr_{c}.trace.tsv")
        except Exception as e:
            print(f"  {c:12} (トレース無し: {e})"); continue
        steps = defaultdict(dict)
        for p, s, l, e in rec:
            if p == "decode": steps[s][l] = e
        stream = [steps[s] for s in sorted(steps)]
        PPs = []
        for L in range(30):
            cnt = Counter()
            for s in stream: cnt.update(s.get(L, []))
            tot = sum(cnt.values())
            if not tot: continue
            H = -sum((v/tot)*math.log2(v/tot) for v in cnt.values())
            PPs.append(2**H)
        ref = T0H.get(c)
        print(f"  {c:12} {math.log2(sum(PPs)/len(PPs)):8.3f} {sum(PPs)/len(PPs):7.1f} "
              f"{(f'{ref:.1f}' if ref else '(未測定)'):>16}")
except Exception as e:
    print(f"  (2^H 計算不可: {e})")
print()
