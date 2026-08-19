import sys
from collections import defaultdict
sys.path.insert(0,"/Users/mh/LLM/turbo-fieldfare/bench")
from expert_sim import read_trace, LayerCache
BASE="/Users/mh/LLM/turbo-fieldfare/bench/logs/36_trace/%s.tsv"
CAND=list(range(8,97,2))
for r in ["ja_prose","en_bullets","code"]:
    h,rec=read_trace(BASE%r)
    byl=defaultdict(list)
    for p,s,l,e in rec: byl[l].append((p,e))
    # miss curve per layer (decode misses only, prefill replayed for warmth)
    curve={}
    for L in range(30):
        curve[L]={}
        for s in CAND:
            c=LayerCache(s,"lfu"); m=0
            for p,e in byl[L]:
                hi,mi=c.request(e)
                if p=="decode": m+=mi
            curve[L][s]=m
    base=sum(curve[L][32] for L in range(30))
    # greedy: start all at 8, add 2 slots to the layer with best marginal gain, budget 960
    alloc={L:8 for L in range(30)}
    used=30*8; budget=30*32
    while used+2<=budget:
        best=None
        for L in range(30):
            n=alloc[L]+2
            if n>CAND[-1]: continue
            g=curve[L][alloc[L]]-curve[L][n]
            if best is None or g>best[0]: best=(g,L)
        if best is None or best[0]<=0: break
        alloc[best[1]]+=2; used+=2
    opt=sum(curve[L][alloc[L]] for L in range(30))
    print(f"{r:12s} uniform32 misses {base} hit {100*(1-base/45840):.2f}%  |  greedy-realloc (same 960 slots) misses {opt} hit {100*(1-opt/45840):.2f}%  gain {100*(base-opt)/45840:+.2f}pp")
    print(f"{'':12s} alloc: {[alloc[L] for L in range(30)]}")
