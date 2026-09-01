import sys
from collections import Counter, defaultdict
sys.path.insert(0,"/Users/mh/LLM/turbo-fieldfare/bench")
from expert_sim import read_trace
BASE="/Users/mh/LLM/turbo-fieldfare/bench/logs/36_trace/%s.tsv"
print("== overlap of topK8 at lag d (mean over layers/positions) ==")
print(f"{'':12s}"+"".join(f"d={d}".rjust(7) for d in range(1,9)))
S={}
for r in ["ja_prose","en_bullets","code"]:
    h,rec=read_trace(BASE%r)
    st=defaultdict(dict)
    for p,s,l,e in rec:
        if p=="decode": st[s][l]=set(e)
    stream=[st[s] for s in sorted(st)]; S[r]=stream; P=len(stream)
    row=f"{r:12s}"
    for d in range(1,9):
        t=0;n=0
        for i in range(P-d):
            for L in range(30):
                t+=len(stream[i][L]&stream[i+d][L]); n+=1
        row+=f"{t/n:7.2f}"
    print(row)
print("\n== distribution of per-position mean overlap with t-1 (avg over 30 layers) ==")
for r in ["ja_prose","en_bullets","code"]:
    stream=S[r]; P=len(stream)
    per=[sum(len(stream[i][L]&stream[i+1][L]) for L in range(30))/30 for i in range(P-1)]
    sp=sorted(per)
    q=lambda p: sp[int(p*(len(sp)-1))]
    print(f"{r:12s} mean {sum(per)/len(per):.2f} p10 {q(.1):.2f} p25 {q(.25):.2f} med {q(.5):.2f} p75 {q(.75):.2f} p90 {q(.90):.2f}  frac<3: {100*sum(1 for x in per if x<3)/len(per):.0f}%  frac>6: {100*sum(1 for x in per if x>6)/len(per):.0f}%")
print("\n== how many decode requests are for an expert in that layer's decode-top32 (static oracle) ==")
for r in ["ja_prose","en_bullets","code"]:
    stream=S[r]
    tot=0;inset=0
    for L in range(30):
        c=Counter()
        for s in stream: c.update(s[L])
        top=set(e for e,_ in c.most_common(32))
        for s in stream:
            for e in s[L]:
                tot+=1
                if e in top: inset+=1
    print(f"{r:12s} static-top32 coverage {100*inset/tot:.2f}%  (tot {tot})")
