"""docs/qfn-prefill/12 §2 F2・F3 の k / sigma の格子を絞った式 (導出、P5: 格子を絞るだけ)。測定の前に 1 度回した MBP 単体の模擬
(bench/qfn/results/2026-09-25-pair-f23/grid-input-solo.log、--qfn-pair-sim run ... A 0 2) の 1 層の秒と、F1 (15) の腕 M の GPU 秒、
F4 (17) の常駐の上限 11.1 GB と流す読み 3.3 GB/s、E0 の l_1 / l_48 から、記録のラウンド分布に掛けた 1 ターンの合計を出す。
--measured: 模擬の条件ごとの式の値 (f23_summary.py と並べて 12 §3-4 のずれを出す)。
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/2026-09-25-pair-f23"))
import re, statistics as st
rows=[l.split("\t") for l in open("grid-input-solo.log") if re.match(r"^\d+\t\d+\t",l)]
L=48
rd=st.mean(float(r[3]) for r in rows); dn=st.mean(float(r[4]) for r in rows); ro=st.mean(float(r[5]) for r in rows); ly=st.mean(float(r[7]) for r in rows)
mb=st.mean(float(r[2]) for r in rows)
print(f"MBP sim per layer: read {rd:.3f} dense {dn:.3f} routed {ro:.3f} layer {ly:.3f} other {ly-rd-dn-ro:.3f}, read MB {mb:.0f}")
# M6 (F1 arm M GPU, F4 streamed 3.3 GB/s), owned buffers
g6_dense, g6_routed = 0.024, 0.037   # F1: proj+hc+shexp+fp32 ~24 ms, routed D ~37 ms
bw=3.3e9
dense_mb=95e6; sel_exp=(mb*1e6-dense_mb)            # selected experts bytes per layer (this route)
full_layer=dense_mb+512*1.38752e6
cap=11.1e9
l1=0.0178; l48=0.431
def A(k, cold, steady):
    nres=0 if cold else min(k,int(cap//full_layer))
    m6=nres*(g6_dense+g6_routed)+(k-nres)*(g6_dense+g6_routed+(dense_mb+sel_exp)/bw)
    mbp=(L-k)*ly
    return (max(m6,mbp) if steady else m6+mbp)+l1
def B(sig, cold):
    cut=round(512*(1-sig)); s=(512-cut)/512
    nres=0 if cold or cut==512 else min(L,int(cap//((512-cut)*1.38752e6*1.0)))
    nres=min(nres,L)
    m6=lambda i: s*g6_routed+(0 if i<nres else s*sel_exp/bw)
    tot=0
    for i in range(L):
        mbp_own=(1-s)*(rd-dense_mb/ (mb*1e6)*rd)+(1-s)*ro
        tot+=dense_mb/(mb*1e6)*rd+dn+(ly-rd-dn-ro)+max(mbp_own, m6(i)+l48/L)
    return tot
dist={1:55,2:10,3:3}
def turn(f):
    return sum(n*(f(False) if c==1 else c*f(True)) for c,n in dist.items())
for cold in (False,True):
    ta=sorted((turn(lambda st_: A(k,cold,st_)),k) for k in range(2,48))
    tb=sorted((sum(n*c*B(s/100,cold) for c,n in dist.items()),s/100) for s in range(0,101,5))
    print("cold" if cold else "resident", "A best", [(round(t,1),k) for t,k in ta[:4]], " B best", [(round(t,1),s) for t,s in tb[:4]])
    print("   solo", round(sum(n*c*L*ly for c,n in dist.items()),1), " A k 17/24/31/40/47:", [round(turn(lambda st_: A(k,cold,st_)),1) for k in (17,24,31,40,47)])
if "--measured" in sys.argv:
    print("式の 1 ターン (模擬と同じ条件):")
    for k, cold in ((20, True), (27, True), (34, True), (24, False), (32, False), (40, False)):
        print(f"  A-{k}-r{'0' if cold else '11.1'} {turn(lambda st_: A(k, cold, st_)):.1f}")
    for s, cold in ((0.4, True), (0.5, True), (0.6, True), (0.45, False), (0.55, False), (0.65, False)):
        print(f"  B-{s}-r{'0' if cold else '11.1'} {sum(n * c * B(s, cold) for c, n in dist.items()):.1f}")
