"""docs/qfn-prefill/07「決め方」: E0〜E2 の証拠から 08 §3 の式の量を抜き、A (k) と B (sigma) の秒を出す (導出)。

    python3 bench/qfn/formula.py [--turns]      # --turns: アプリの記録 (turn-metrics.jsonl、Qwen3.8、3 ラウンド以上) を式に通す

量は E1 の 2 本目のチャンク ([2048..<4096)、3 run の中央値) から 1 層あたりに割る。
  d = pre-router GPU / 48, r = routed GPU / 48, t = topk / 48, s = advise (pread) / 48
  o = (チャンクの壁時計 - ple - head - pre GPU - routed GPU - topk - advise) / 48   ... 式に無かった host の残り (08 §3 への追加)
  p = PLE 秒/チャンク (冷えた値: MBP は run-1、M6 は全 run)
M6 の写しあり s+ は s_M6 × (E2 同時 / E2 内蔵だけ) の中央値の比。l_1・l_48 は E0 (MBP→M6) の中央値。
"""
import glob
import json
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
med = st.median
PAT = re.compile(r"\[\s*(\d+)\.\.<\s*(\d+)\]\s+([\d.]+)s.*?ple (\d+) pre (\d+) \[gpu (\d+)\] route (\d+) routed (\d+) \[gpu (\d+)\] head (\d+) ms.*?topk (\d+) views (\d+) advise (\d+)")

def e1(host):
    rows = []
    for f in sorted(glob.glob(f"{R}/2026-09-25-{host}-e1-stages/run-*.log")):
        run = int(re.search(r"run-(\d+)", f).group(1))
        for m in PAT.finditer(open(f).read()):
            a = [float(x) for x in m.groups()]
            rows.append(dict(run=run, start=int(a[0]), wall=a[2], ple=a[3] / 1e3, pre=a[4] / 1e3, pregpu=a[5] / 1e3, route=a[6] / 1e3,
                             routedgpu=a[8] / 1e3, head=a[9] / 1e3, topk=a[10] / 1e3, views=a[11] / 1e3, advise=a[12] / 1e3))
    c2 = [r for r in rows if r["start"] == 2048]
    q = {k: med(r[k] for r in c2) for k in ("wall", "pregpu", "routedgpu", "topk", "advise", "head", "ple")}
    L = 48
    v = dict(d=q["pregpu"] / L, r=q["routedgpu"] / L, t=q["topk"] / L, s=q["advise"] / L,
             o=(q["wall"] - q["ple"] - q["head"] - q["pregpu"] - q["routedgpu"] - q["topk"] - q["advise"]) / L,
             head=q["head"], wall=q["wall"])
    cold = [r["ple"] for r in rows if r["run"] == 1] if host == "Mac15-6" else [r["ple"] for r in rows]
    v["p"] = med(cold)
    return v

def e2_ratio():
    def walls(label):
        return [float(re.search(r"wall ([\d.]+) s", open(f).read()).group(1))
                for f in glob.glob(f"{R}/2026-09-25-Mac18-5-{label}/layer-*.log")
                if float(re.search(r"resident before ([\d.]+) GB\), [\d.]+ GB/s, threads", open(f).read()).group(1)) < 0.05]
    return med(walls("e2-ssd-both32")) / med(walls("e2-ssd-internal"))

def e0():
    rows = [l.split("\t") for l in open(f"{R}/2026-09-25-pair-e0-link/mbp-to-m6.tsv") if l[:1].isdigit()]
    return med(float(r[1]) for r in rows), med(float(r[2]) for r in rows)

M, S = e1("Mac15-6"), e1("Mac18-5")
ratio = e2_ratio()
l1, l48 = e0()
S_plus = dict(S, s=S["s"] * ratio)

def e4_arm_c(host):
    """E4-(b): arm C (fp16 展開 + MPS fp16) の 1 層の GPU 秒、3 層の中央値。"""
    v = [float(re.search(r"C fp16 dequant \+ MPS fp16 GEMM \(group \d+\): GPU ([\d.]+) ms", open(f).read()).group(1)) / 1e3
         for f in glob.glob(f"{R}/2026-09-25-{host}-e4-gemm/layer-*.log")]
    return med(v)

r_nax = e4_arm_c("Mac18-5")
S_nax, S_plus_nax = dict(S, r=r_nax), dict(S_plus, r=r_nax)
L = 48

def w(X):   # one layer on one machine, everything
    return X["d"] + X["t"] + X["s"] + X["r"] + X["o"]

def solo():
    return L * w(M) + M["p"]

def A(k, X6, steady):
    m6, mbp = k * w(X6) + X6["p"], (L - k) * w(M)
    return (max(m6, mbp) if steady else m6 + mbp) + l1

def B(sig, X6):
    return L * (M["d"] + M["t"] + M["o"] + max((1 - sig) * (M["s"] + M["r"]), sig * (X6["s"] + X6["r"]) + l48 / L)) + M["p"]

def show():
    print("1 層あたりの量 (s、導出。E1 の 2 本目のチャンク、3 run の中央値 ÷ 48):")
    for name, X in (("MBP", M), ("M6", S), ("M6 写しあり", S_plus), ("M6 fp16", S_nax), ("M6 fp16 写し", S_plus_nax)):
        print(f"  {name:10s} d {X['d']:.4f}  r {X['r']:.4f}  t {X['t']:.4f}  s {X['s']:.4f}  o {X['o']:.4f}  | w {w(X):.4f}  p {X['p']:.2f}/チャンク")
    print(f"  M6 fp16 の r は E4 の腕 C (MPS fp16) の 1 層の GPU 秒 {r_nax:.4f} (3 層の中央値、本番の混合ではなく単体ベンチ)")
    print(f"  E2 の比 (同時 / 内蔵だけ) {ratio:.3f}、l_1 {l1:.4f} s、l_48 {l48:.4f} s")
    print(f"  単体の式 {solo():.2f} s/チャンク (= E1 の 2 本目の壁時計 {M['wall']:.2f} - head {M['head']:.2f} - 暖かい ple + 冷えた ple {M['p']:.2f}。o はこの一致のために置いた)")

KS = range(2, 48)   # A: the M6 runs the PLE (layer 1), so it holds at least layers 0-1

def best_A(X6, steady):
    return min((A(k, X6, steady), k) for k in KS)

def best_B(X6):
    return min((B(s / 100, X6), s / 100) for s in range(0, 101, 5))

def turns():
    f = os.path.expanduser("~/Library/Application Support/Tsugumi/turn-metrics.jsonl")
    by = defaultdict(list)
    for l in open(f):
        r = json.loads(l)
        if r.get("model") == "qwen38-flash-next-iq2" and r["recordedAt"][:10] in ("2026-09-15", "2026-09-17"):
            by[r["turnID"]].append(r)
    out = []
    for tid, rs in by.items():
        rs.sort(key=lambda r: r["round"])
        if len(rs) >= 3:
            out.append((tid, [r["promptTokens"] - r.get("cachedPromptTokens", 0) for r in rs], sum(r["prefillSeconds"] for r in rs)))
    return out

def e3_check():
    """E3: 模擬の各側の秒 (sigma 0 の MBP だけ、sigma 1 の M6 だけ、冷えた 3 run の中央値) から式の B の routed 部分を出し、実測と並べる。"""
    def side(label, m6):
        tot = []
        for f in sorted(glob.glob(f"{R}/2026-09-25-pair-{label}/run-*.{'m6.' if m6 else ''}log")):
            if not m6 and f.endswith(".m6.log"):
                continue
            rows = [l.split("\t") for l in open(f) if l[:1].isdigit() and len(l.split("\t")) == 9]
            tot.append(sum(float(r[4]) + float(r[6]) for r in rows) if m6 else sum(float(r[7]) for r in rows))
        return med(tot)
    mbp, m6 = side("e3-cold-s0", False), side("e3-cold-s1", True)
    print(f"E3 (模擬、arm B の GEMM、main の down): MBP だけ {mbp:.2f} s、M6 だけ (read + GEMM) {m6:.2f} s /チャンク")
    for sg in ("0.35", "0.43", "0.5"):
        x = float(sg)
        pred = L * max((1 - x) * mbp / L, x * m6 / L + l48 / L)
        meas = side(f"e3-cold-s{sg}", False)
        print(f"  sigma {sg}: 式 {pred:.2f} s / 実測 {meas:.2f} s / ずれ {100 * (meas - pred) / pred:+.1f}%")

if __name__ == "__main__":
    show()
    print()
    e3_check()
    print()
    print("1 チャンク (2048) の秒 (導出):")
    print(f"  単体 {solo():.2f}")
    for lab, X6 in (("写しなし", S), ("写しあり", S_plus), ("fp16 写しなし", S_nax), ("fp16 写しあり", S_plus_nax)):
        a1, k1 = best_A(X6, False); a2, k2 = best_A(X6, True); b, sg = best_B(X6)
        print(f"  {lab}: A 1 チャンクのラウンド {a1:.2f} (k {k1}) / A 定常 {a2:.2f} (k {k2}) / B {b:.2f} (sigma {sg:.2f})")
    if "--turns" in sys.argv:
        ts = turns()
        print()
        print(f"記録: アプリ turn-metrics.jsonl、Qwen3.8、2026-09-15/17、3 ラウンド以上のターン {len(ts)} 本、ラウンド {sum(len(t[1]) for t in ts)}")
        for lab, X6 in (("写しなし", S), ("写しあり", S_plus), ("fp16 写しなし", S_nax), ("fp16 写しあり", S_plus_nax)):
            def total_A(k):
                return sum((A(k, X6, False) if c == 1 else c * A(k, X6, True)) for _, ns, _ in ts for c in [math.ceil(n / 2048) for n in ns] if c > 0)
            def total_B(sig):
                return sum(c * B(sig, X6) for _, ns, _ in ts for c in [math.ceil(n / 2048) for n in ns] if c > 0)
            tot_solo = sum(c * solo() for _, ns, _ in ts for c in [math.ceil(n / 2048) for n in ns] if c > 0)
            ka = min(KS, key=total_A); sb = min((s / 100 for s in range(0, 101, 5)), key=total_B)
            print(f"  {lab}: 単体 {tot_solo:.1f} s / A (k {ka}) {total_A(ka):.1f} s / B (sigma {sb:.2f}) {total_B(sb):.1f} s   (全ターンの prefill 合計、導出)")
        chunks = [math.ceil(n / 2048) for _, ns, _ in ts for n in ns]
        print(f"  ラウンドのチャンク数: " + ", ".join(f"{c} チャンク {chunks.count(c)} 本" for c in sorted(set(chunks))))
        print(f"  記録の prefill 実測合計 {sum(t[2] for t in ts):.1f} s (アプリ、当時のコード。式の単体とは別物)")
