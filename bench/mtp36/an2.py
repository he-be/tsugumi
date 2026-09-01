import sys, statistics
from collections import Counter, defaultdict
sys.path.insert(0,"/Users/mh/LLM/tsugumi/bench")
from expert_sim import read_trace, LayerCache
BASE="/Users/mh/LLM/tsugumi/bench/logs/36_trace/%s.tsv"
REGS=["ja_prose","en_bullets","code"]
D={}
for r in REGS:
    h,rec=read_trace(BASE%r)
    steps=defaultdict(dict)
    for p,s,l,e in rec:
        if p=="decode": steps[s][l]=e
    order=sorted(steps); stream=[steps[s] for s in order]
    D[r]=(h,rec,order,stream)

print("== working set: distinct experts per layer in a sliding window of W positions (mean over layers & windows) ==")
print(f"{'':12s}" + "".join(f"W={w}".rjust(9) for w in (2,4,8,16,32,64,191)))
for r in REGS:
    h,rec,order,stream=D[r]; P=len(stream)
    row=f"{r:12s}"
    for w in (2,4,8,16,32,64,191):
        tot=0;n=0
        for i in range(0,P-w+1):
            for L in range(30):
                u=set()
                for j in range(i,i+w): u|=set(stream[j][L])
                tot+=len(u);n+=1
        row+=f"{tot/n:9.1f}"
    print(row)

print("\n== per-layer: top32 share vs actual lfu32 hit (decode) ==")
for r in REGS:
    h,rec,order,stream=D[r]
    cnt={L:Counter() for L in range(30)}
    for s in stream:
        for L in range(30): cnt[L].update(s[L])
    caches=defaultdict(lambda: LayerCache(32,"lfu")); pl=defaultdict(lambda:[0,0])
    for p,s,l,e in rec:
        hi,mi=caches[l].request(e)
        if p=="decode": pl[l][0]+=hi; pl[l][1]+=mi
    t32=[];act=[]
    for L in range(30):
        tot=sum(cnt[L].values()); mc=[c for _,c in cnt[L].most_common()]
        t32.append(100*sum(mc[:32])/tot); act.append(100*pl[L][0]/(pl[L][0]+pl[L][1]))
    diff=[a-t for a,t in zip(act,t32)]
    print(f"{r:12s} mean top32 {sum(t32)/30:5.1f}  mean actual {sum(act)/30:5.1f}  mean(actual-top32) {sum(diff)/30:+5.1f}  max|diff| {max(abs(d) for d in diff):.1f}")

print("\n== per-position miss series, full (of 240) ==")
for r in REGS:
    h,rec,order,stream=D[r]
    caches=defaultdict(lambda: LayerCache(32,"lfu")); ps=Counter()
    for p,s,l,e in rec:
        hi,mi=caches[l].request(e)
        if p=="decode": ps[s]+=mi
    v=[ps[s] for s in order]
    print(f"--- {r} ---")
    for i in range(0,len(v),32):
        print("  "+" ".join(f"{x:3d}" for x in v[i:i+32]))
    # autocorrelation
    mu=sum(v)/len(v); var=sum((x-mu)**2 for x in v)
    ac=[sum((v[i]-mu)*(v[i+k]-mu) for i in range(len(v)-k))/var for k in range(1,13)]
    print("  autocorr lag1..12: "+" ".join(f"{a:+.2f}" for a in ac))
    # burst share: what fraction of total misses live in the worst 10% of positions
    sv=sorted(v,reverse=True); n10=max(1,len(v)//10)
    print(f"  total {sum(v)}  worst10%pos share {100*sum(sv[:n10])/sum(v):.1f}%  positions>2x mean: {sum(1 for x in v if x>2*mu)}")

print("\n== first-vs-rest: excluding the first 8 positions (warm-up) ==")
for r in REGS:
    h,rec,order,stream=D[r]
    caches=defaultdict(lambda: LayerCache(32,"lfu")); ps=Counter()
    for p,s,l,e in rec:
        hi,mi=caches[l].request(e)
        if p=="decode": ps[s]+=mi
    v=[ps[s] for s in order]
    for cut in (0,8,32,96):
        tail=v[cut:]
        print(f"{r:12s} from pos {cut:3d}: hit {100*(1-sum(tail)/(240*len(tail))):.2f}%  ({len(tail)} pos)")
