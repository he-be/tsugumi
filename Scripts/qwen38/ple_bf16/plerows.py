import sys, numpy as np
MULT=[23703573157769, 20109073645365, 8052911324071]
EOS=248044
sys.path.insert(0, sys.path[0]); from ggufhdr import read
kv,_,_=read('/Users/mh/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf')
OFF=kv['qwen4exp.ple.head_offsets']; VOC=kv['qwen4exp.ple.head_vocab_sizes']; PER=kv['qwen4exp.ple.heads_per_ngram']; NG=kv['qwen4exp.ple.ngram_size']
def rows_for(tokens):
    prev=[EOS,EOS]; out=[]
    for tok in tokens:
        ctx=[tok]; cut=False
        for s in range(1,NG):
            t=EOS if cut else prev[s-1]; cut=cut or t==EOS; ctx.append(EOS if cut else t)
        for n in range(2,NG+1):
            m=(ctx[0]*MULT[0])&0xFFFFFFFFFFFFFFFF
            for j in range(1,n): m^=(ctx[j]*MULT[j])&0xFFFFFFFFFFFFFFFF
            for g in range(PER):
                h=(n-2)*PER+g; out.append(m%VOC[h]+OFF[h])
        prev=[tok]+prev[:-1]
    return out
def load(p): return [int(x) for x in open(p).read().replace('\n',',').split(',') if x.strip()]
if __name__=='__main__':
    allr=set()
    for p in sys.argv[1:]:
        t=load(p); r=rows_for(t); allr|=set(r)
        print(p.split('scratch/qwen38/')[-1], 'tokens',len(t),'rows',len(r),'unique',len(set(r)))
    print('union',len(allr), 'bytes bf16', len(allr)*320)
