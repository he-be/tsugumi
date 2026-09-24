# MLX の headDim 512 の全注意 (Gemma 4 の full 層: 16/2 ヘッド、2478 トークンを 2048 + 430、5 層分) を、
# mx.fast.scaled_dot_product_attention と、GEMM → softmax → GEMM を手で書いた形で並べる (docs/m6-prefill/36)。
# 2 つが同じ秒なら、MLX は d512 で融合カーネルを使っていない。校正であって台帳の行ではない。
#   python bench/m6/ref_mlx_sdpa512.py 5 float16
# 出力は ref_mlx_ops.sh の summary.tsv と同じ「段 \t 秒」の行。
import sys
import time

import mlx.core as mx

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
dt = getattr(mx, sys.argv[2]) if len(sys.argv) > 2 else mx.float16
LF, CHUNKS = 5, [(2048, 2048), (430, 2478)]


def timed(fn, q, k, m):
    mx.eval(fn(q, k, m))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn(q, k, m))
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def fused(q, k, m):
    return mx.fast.scaled_dot_product_attention(q, k, k, scale=1.0, mask=m)


def unfused(q, k, m):
    kk = mx.repeat(k, 8, axis=1)
    s = mx.where(m, q @ kk.transpose(0, 1, 3, 2), -mx.inf)
    return mx.softmax(s, axis=-1, precise=True) @ kk


inputs = []
for n, kend in CHUNKS:
    q = (mx.random.normal((1, 16, n, 512)) * 0.1).astype(dt)
    k = (mx.random.normal((1, 2, kend, 512)) * 0.1).astype(dt)
    m = mx.arange(kend)[None, :] <= mx.arange(kend - n, kend)[:, None]
    mx.eval(q, k, m)
    inputs.append((q, k, m))

print('stage\tseconds')
for name, fn in [('sdpa.fused', fused), ('sdpa.unfused', unfused)]:
    print(f'{name}\t{LF * sum(timed(fn, *x) for x in inputs):.4f}')
