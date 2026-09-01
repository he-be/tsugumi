import sys, math
from collections import Counter, defaultdict
sys.path.insert(0,"/Users/mh/LLM/tsugumi/bench")
from expert_sim import read_trace
BASE="/Users/mh/LLM/tsugumi/bench/logs/36_trace/%s.tsv"
print(f"{'reg':12s} {'H(bits)':>8s} {'perplexity':>11s} {'eff.experts/layer':>18s}")
for r in ["ja_prose","en_bullets","code"]:
    h,rec=read_trace(BASE%r)
    steps=defaultdict(dict)
    for p,s,l,e in rec:
        if p=="decode": steps[s][l]=e
    stream=[steps[s] for s in sorted(steps)]
    Hs=[];PPs=[]
    for L in range(30):
        c=Counter()
        for s in stream: c.update(s[L])
        tot=sum(c.values())
        H=-sum((v/tot)*math.log2(v/tot) for v in c.values())
        Hs.append(H); PPs.append(2**H)
    print(f"{r:12s} {sum(Hs)/30:8.3f} {sum(PPs)/30:11.1f} {'':18s}")
    # deep vs shallow
    print(f"{'':12s} layers0-9 pp {sum(PPs[:10])/10:.1f}  10-19 {sum(PPs[10:20])/10:.1f}  20-29 {sum(PPs[20:])/10:.1f}")
