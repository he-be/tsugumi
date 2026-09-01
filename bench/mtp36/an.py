#!/usr/bin/env python3
import sys, statistics
from collections import Counter, defaultdict
sys.path.insert(0, "/Users/mh/LLM/turbo-fieldfare/bench")
from expert_sim import read_trace, LayerCache

BASE = "/Users/mh/LLM/turbo-fieldfare/bench/logs/36_trace/%s.tsv"
REGS = ["ja_prose", "en_bullets", "code"]

def load(r):
    h, rec = read_trace(BASE % r)
    return h, rec

def decode_stream(rec):
    steps = defaultdict(dict)
    for phase, step, layer, ex in rec:
        if phase == "decode":
            steps[step][layer] = ex
    order = sorted(steps)
    return order, [steps[s] for s in order]

def replay(rec, slots, policy, include_prefill=True):
    caches = defaultdict(lambda: LayerCache(slots, policy))
    st = {"prefill":[0,0], "decode":[0,0]}
    per_layer = defaultdict(lambda:[0,0])
    per_step = Counter()
    for phase, step, layer, ex in rec:
        if phase == "prefill" and not include_prefill:
            continue
        if len(ex) > slots: return None
        hi, mi = caches[layer].request(ex)
        st[phase][0]+=hi; st[phase][1]+=mi
        if phase=="decode":
            per_layer[layer][0]+=hi; per_layer[layer][1]+=mi
            per_step[step]+=mi
    return st, per_layer, per_step

def belady(stream, layers, slots):
    """per-layer optimal (Belady) miss count over decode only."""
    total_m = 0
    for L in range(layers):
        seq = [s[L] for s in stream]
        # next-use index per position
        cache = set()
        misses = 0
        for i, ex in enumerate(seq):
            need = list(dict.fromkeys(ex))
            for e in need:
                if e in cache: continue
                misses += 1
                if len(cache) >= slots:
                    # evict the one used farthest in future (excluding this req's set)
                    protect = set(need)
                    nxt = {}
                    for c in cache:
                        nxt[c] = 10**9
                    for j in range(i+1, len(seq)):
                        for e2 in seq[j]:
                            if e2 in nxt and nxt[e2] == 10**9:
                                nxt[e2] = j
                    cand = [c for c in cache if c not in protect]
                    victim = max(cand, key=lambda c: nxt[c])
                    cache.discard(victim)
                cache.add(e)
        total_m += misses
    return total_m

out = {}
for r in REGS:
    h, rec = load(r)
    layers = int(h["layers"]); nexp = int(h["experts"]); slots = int(h["slots"])
    order, stream = decode_stream(rec)
    P = len(stream)
    d = {}
    d["header"]=h; d["P"]=P; d["order"]=order

    # --- 1. distinct experts per layer
    cnt = {L: Counter() for L in range(layers)}
    for s in stream:
        for L in range(layers):
            cnt[L].update(s[L])
    d["distinct"] = {L: len(cnt[L]) for L in range(layers)}
    d["cnt"] = cnt

    # --- 2. concentration
    top8={}; top32={}
    for L in range(layers):
        tot = sum(cnt[L].values())
        mc = [c for _,c in cnt[L].most_common()]
        top8[L] = sum(mc[:8])/tot
        top32[L] = sum(mc[:32])/tot
    d["top8"]=top8; d["top32"]=top32

    # --- 3. reuse distance (positional gap + LRU stack distance)
    gaps=[]; stack=[]
    for L in range(layers):
        last={}
        seen_since=defaultdict(set)  # not efficient; do per-layer scan
        seq=[s[L] for s in stream]
        lastpos={}
        for i,ex in enumerate(seq):
            for e in ex:
                if e in lastpos:
                    gaps.append(i-lastpos[e])
                    # stack distance: distinct experts in this layer between
                    dist=set()
                    for j in range(lastpos[e]+1, i):
                        dist.update(seq[j])
                    dist.discard(e)
                    stack.append(len(dist))
                else:
                    gaps.append(None); stack.append(None)
            for e in ex: lastpos[e]=i
    d["gaps"]=gaps; d["stack"]=stack

    # --- 4. consecutive overlap
    ov=[]
    ovL={L:[] for L in range(layers)}
    for i in range(P-1):
        for L in range(layers):
            k=len(set(stream[i][L]) & set(stream[i+1][L]))
            ov.append(k); ovL[L].append(k)
    d["ov"]=ov; d["ovL"]=ovL

    # union size for speculative blocks k=2,4,8
    d["union"]={}
    for k in (1,2,4,8):
        tot=0; n=0
        for i in range(0, P-k+1):
            for L in range(layers):
                u=set()
                for j in range(i,i+k): u|=set(stream[j][L])
                tot+=len(u); n+=1
        d["union"][k]=tot/n

    # --- 5. per-position miss / novelty
    st, per_layer, per_step = replay(rec, slots, h["policy"])
    d["replay"]=st; d["per_layer"]=per_layer; d["per_step"]=per_step
    # novelty: experts (layer,expert) never seen before in decode
    seen=set(); nov=Counter()
    for i,s in enumerate(stream):
        for L in range(layers):
            for e in s[L]:
                if (L,e) not in seen:
                    seen.add((L,e)); nov[order[i]]+=1
    d["nov"]=nov

    # counterfactual sims
    d["sims"]={}
    for slots2 in (16,24,32,40,48,64,96,128):
        for pol in ("lfu","lru"):
            rr=replay(rec, slots2, pol)
            if rr: d["sims"][(slots2,pol)]=tuple(rr[0]["decode"])
    d["belady32"]=belady(stream, layers, 32)
    out[r]=d

# ---------- report ----------
def pct(h,m): return 100.0*h/(h+m)

print("== replay check (decode hits/misses, lfu 32) ==")
for r in REGS:
    h,m = out[r]["replay"]["decode"]
    print(f"{r:12s} {h}/{h+m} = {pct(h,m):.2f}%   prefill {out[r]['replay']['prefill']}")

print("\n== 1. distinct experts per layer (of 128), decode 191 pos ==")
print(f"{'':12s} {'mean':>6s} {'min':>4s} {'max':>4s}   per-layer")
for r in REGS:
    v=[out[r]["distinct"][L] for L in range(30)]
    print(f"{r:12s} {sum(v)/30:6.1f} {min(v):4d} {max(v):4d}   {v}")

print("\n== 2. concentration (share of 1528 req/layer) ==")
for r in REGS:
    t8=[out[r]["top8"][L] for L in range(30)]; t32=[out[r]["top32"][L] for L in range(30)]
    print(f"{r:12s} top8 {100*sum(t8)/30:5.1f}%  top32 {100*sum(t32)/30:5.1f}%  (top32 min {100*min(t32):.1f} max {100*max(t32):.1f})")

print("\n== 3. reuse distance ==")
for r in REGS:
    g=[x for x in out[r]["gaps"] if x is not None]
    s=[x for x in out[r]["stack"] if x is not None]
    cold=sum(1 for x in out[r]["gaps"] if x is None)
    q=lambda a,p: sorted(a)[int(p*(len(a)-1))]
    lt32=100*sum(1 for x in s if x<32)/len(s)
    print(f"{r:12s} cold {cold:5d}  gap med {statistics.median(g):5.1f} p25 {q(g,.25)} p75 {q(g,.75)} p90 {q(g,.90)}")
    print(f"{'':12s} stackdist med {statistics.median(s):5.1f} p25 {q(s,.25)} p75 {q(s,.75)} p90 {q(s,.90)}  |  <32: {lt32:.1f}%  (LRU-32 ceiling incl cold: {100*sum(1 for x in s if x<32)/(len(s)+cold):.1f}%)")

print("\n== 4. consecutive overlap (of 8) & union sizes ==")
for r in REGS:
    o=out[r]["ov"]
    print(f"{r:12s} mean overlap {sum(o)/len(o):.3f}/8   union k=1 {out[r]['union'][1]:.2f} k=2 {out[r]['union'][2]:.2f} k=4 {out[r]['union'][4]:.2f} k=8 {out[r]['union'][8]:.2f}")

print("\n== 5. non-stationarity: per-position misses (of 240) ==")
for r in REGS:
    ps=out[r]["per_step"]; order=out[r]["order"]
    v=[ps[s] for s in order]
    nv=[out[r]["nov"][s] for s in order]
    print(f"{r:12s} miss/pos mean {sum(v)/len(v):6.2f} sd {statistics.pstdev(v):5.2f} min {min(v)} max {max(v)}  p90 {sorted(v)[int(.9*(len(v)-1))]}")
    print(f"{'':12s} novel/pos mean {sum(nv)/len(nv):6.2f} first20 {nv[:20]}")
    print(f"{'':12s} miss series (20-pos means): {[round(sum(v[i:i+20])/len(v[i:i+20]),1) for i in range(0,len(v),20)]}")

print("\n== counterfactual sims (decode hit %) ==")
hdr=f"{'':12s}" + "".join(f"{s}{p[0]}".rjust(9) for s in (16,24,32,40,48,64,96,128) for p in ("lfu","lru"))
print(hdr)
for r in REGS:
    row=f"{r:12s}"
    for s in (16,24,32,40,48,64,96,128):
        for p in ("lfu","lru"):
            k=(s,p)
            row += (f"{pct(*out[r]['sims'][k]):8.1f}%" if k in out[r]["sims"] else "        -")
    print(row)
print("\nBelady optimal @32 slots (decode misses / hit%):")
for r in REGS:
    b=out[r]["belady32"]
    print(f"{r:12s} misses {b}  hit {100*(45840-b)/45840:.2f}%")

print("\n== per-layer decode hit% (lfu32) ==")
for r in REGS:
    v=[100*out[r]["per_layer"][L][0]/(out[r]["per_layer"][L][0]+out[r]["per_layer"][L][1]) for L in range(30)]
    print(f"{r:12s} " + " ".join(f"{x:.0f}" for x in v))
