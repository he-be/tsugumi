import sys, numpy as np
V=248320
a=np.fromfile(sys.argv[1],np.float32).reshape(-1,V).astype(np.float64)  # reference arm (bf16)
b=np.fromfile(sys.argv[2],np.float32).reshape(-1,V).astype(np.float64)
lo,hi=int(sys.argv[3]),int(sys.argv[4])  # positions [lo, hi)
def lp(x): x=x/0.7; x=x-x.max(1,keepdims=True); return x-np.log(np.exp(x).sum(1,keepdims=True))
pa,pb=lp(a[lo:hi]),lp(b[lo:hi])
kl=(np.exp(pa)*(pa-pb)).sum(1)
t1=(a[lo:hi].argmax(1)==b[lo:hi].argmax(1)).sum()
print(f'positions {lo}..{hi-1}: top1 {t1}/{hi-lo}  KL mean {kl.mean():.4f} max {kl.max():.4f}  max|Δ| {np.abs(a[lo:hi]-b[lo:hi]).max():.3f}')
print('per-position KL', ' '.join(f'{k:.3f}' for k in kl))
