#!/usr/bin/env python3
"""MLX kernels on Tsugumi's prefill shapes, stage by stage (docs/m6-prefill/31).

  python ref_mlx_ops.py [reps] [dtype]

One layer's weights, random, in Tsugumi's format (affine int4, group 32) with
fp16 activations (dtype=bfloat16 to see MLX's own model dtype). T = 2478 in the
ledger's chunks (2048 + 430), 25 sliding and 5 full layers, the calls mlx_lm's
gemma4_text makes (separate gate / up gather_qmm with sorted_indices, argsort
routing). Each call is evaluated and synchronized alone; a stage's seconds are
the sum over its layers and chunks, the median of `reps` passes after one warm
pass. Routing is uniform random top-8 (the model's is not uniform).
Prints one `stage<TAB>seconds<TAB>TFLOP/s` line per stage, comparable to
floor_table.py's rows (same FLOP counts).
"""
import sys, time, statistics
import mlx.core as mx

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
dt = getattr(mx, sys.argv[2]) if len(sys.argv) > 2 else mx.float16
CHUNKS = (2048, 430)
T = sum(CHUNKS)
H, LS, LF, E, K, I, IS = 2816, 25, 5, 128, 8, 704, 2112
GS, BITS = 32, 4
mx.random.seed(0)


def qw(out_dim, in_dim, experts=None):
    shape = (experts, out_dim, in_dim) if experts else (out_dim, in_dim)
    w = (mx.random.normal(shape) * in_dim ** -0.5).astype(dt)
    return mx.quantize(w, group_size=GS, bits=BITS)


def qmm(x, w):
    return mx.quantized_matmul(x, *w, transpose=True, group_size=GS, bits=BITS)


def gqmm(x, w, idx):
    return mx.gather_qmm(x, *w, rhs_indices=idx, transpose=True,
                         group_size=GS, bits=BITS, sorted_indices=True)


def act(n, d):
    return (mx.random.normal((n, d)) * 0.5).astype(dt)


def timed(fn, calls):
    """Sum of per-call wall seconds (eval + synchronize) over `calls` calls."""
    total = 0.0
    for _ in range(calls):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        total += time.perf_counter() - t0
    return total


def mask(q, k, window=None):
    # Chunk of q rows ending at key k (offset k - q), causal, optional window.
    qi = mx.arange(k - q, k)[:, None]
    ki = mx.arange(k)[None, :]
    m = ki <= qi
    if window:
        m = m & (ki > qi - window)
    return m


# Weights (one layer each).
w_qkv_s, w_qkv_f = qw(4096 + 2048 + 2048, H), qw(8192 + 1024, H)
w_o_s, w_o_f = qw(H, 4096), qw(H, 8192)
w_sh_gu, w_sh_d = qw(2 * IS, H), qw(H, IS)
w_up, w_gate, w_down = qw(I, H, E), qw(I, H, E), qw(H, I, E)
w_router = (mx.random.normal((E, H)) * H ** -0.5).astype(dt)

# Inputs per chunk.
st = {}
for c, n in enumerate(CHUNKS):
    k_end = sum(CHUNKS[: c + 1])
    idx = mx.argsort(mx.random.uniform(shape=(n, E)), axis=-1)[:, :K].astype(mx.uint32)
    flat = idx.flatten()
    order = mx.argsort(flat)
    st[c] = dict(
        n=n, x=act(n, H), xs=act(n, 4096), xf=act(n, 8192), xi=act(n, IS),
        idx=idx, xm=act(n * K, H)[:, None, :], sidx=flat[order], xd=act(n * K, I)[:, None, :],
        qs=act(16 * n, 256).reshape(1, 16, n, 256), ks=act(8 * k_end, 256).reshape(1, 8, k_end, 256),
        vs=act(8 * k_end, 256).reshape(1, 8, k_end, 256), ms=mask(n, k_end, 1024),
        qf=act(16 * n, 512).reshape(1, 16, n, 512), kf=act(2 * k_end, 512).reshape(1, 2, k_end, 512),
        mf=mask(n, k_end), g=act(n * K, I)[:, None, :], u=act(n * K, I)[:, None, :],
        y=act(n * K, H), inv=mx.argsort(order),
    )
    mx.eval([v for v in st[c].values() if isinstance(v, mx.array)])


def stage(fn_by_chunk, layers):
    return sum(timed(lambda s=st[c]: fn_by_chunk(s), layers) for c in st)


STAGES = {
    # name: (fn(chunk state), layers, FLOP over the whole prompt)
    'qkv': (None, None, LS * 2 * T * H * 8192 + LF * 2 * T * H * 9216),
    'oproj': (None, None, LS * 2 * T * 4096 * H + LF * 2 * T * 8192 * H),
    'attn.swa': (lambda s: mx.fast.scaled_dot_product_attention(s['qs'], s['ks'], s['vs'], scale=1.0, mask=s['ms']),
                 LS, LS * 4 * 16 * 256 * sum(min(t, 1024) for t in range(1, T + 1))),
    'attn.full': (lambda s: mx.fast.scaled_dot_product_attention(s['qf'], s['kf'], s['kf'], scale=1.0, mask=s['mf']),
                  LF, LF * 4 * 16 * 512 * (T * (T + 1) // 2)),
    'rope': (lambda s: [mx.fast.rope(s['qs'], 256, traditional=False, base=10000.0, scale=1.0, offset=0),
                        mx.fast.rope(s['ks'][:, :, -s['n']:], 256, traditional=False, base=10000.0, scale=1.0, offset=0)],
             LS + LF, 0),
    'post.router': (lambda s: s['x'] @ w_router.T, LS + LF, (LS + LF) * 2 * T * H * E),
    'shared': (lambda s: qmm(mx.split(qmm(s['x'], w_sh_gu), 2, axis=-1)[0], w_sh_d), LS + LF,
               (LS + LF) * 2 * T * H * IS * 3),
    'moe.sort': (lambda s: [s['x'][mx.argsort(s['idx'].flatten()) // K], mx.argsort(mx.argsort(s['idx'].flatten()))],
                 LS + LF, 0),
    'moe.gateup': (lambda s: [gqmm(s['xm'], w_up, s['sidx']), gqmm(s['xm'], w_gate, s['sidx'])], LS + LF,
                   (LS + LF) * 2 * T * K * H * I * 2),
    'moe.gelu': (lambda s: (0.5 * s['g'] * (1 + mx.tanh(0.7978845608 * (s['g'] + 0.044715 * s['g'] ** 3)))) * s['u'],
                 LS + LF, 0),
    'moe.down': (lambda s: gqmm(s['xd'], w_down, s['sidx']), LS + LF, (LS + LF) * 2 * T * K * I * H),
    'moe.unsort': (lambda s: s['y'][s['inv']].reshape(s['n'], K, H).sum(-2), LS + LF, 0),
}


def qkv(s, full):
    return qmm(s['x'], w_qkv_f if full else w_qkv_s)


def oproj(s, full):
    return qmm(s['xf'] if full else s['xs'], w_o_f if full else w_o_s)


def run_pass():
    out = {}
    out['qkv'] = stage(lambda s: qkv(s, False), LS) + stage(lambda s: qkv(s, True), LF)
    out['oproj'] = stage(lambda s: oproj(s, False), LS) + stage(lambda s: oproj(s, True), LF)
    for name, (fn, layers, _) in STAGES.items():
        if fn is not None:
            out[name] = stage(fn, layers)
    return out


print(f'mlx {mx.__version__} dtype {dt} T {T} chunks {CHUNKS} layers {LS}+{LF} gs {GS} bits {BITS} reps {reps}')
print('device', mx.metal.device_info() if hasattr(mx.metal, 'device_info') else mx.device_info())
run_pass()  # warm: pipeline compilation, first-touch of weights
passes = [run_pass() for _ in range(reps)]
print('stage\tseconds\tTFLOP/s\tall_passes')
tot = 0.0
for name, (_, _, fl) in STAGES.items():
    xs = [p[name] for p in passes]
    m = statistics.median(xs)
    tot += m
    print(f'{name}\t{m:.4f}\t{fl / m / 1e12 if fl else 0:.2f}\t' + ','.join(f'{x:.4f}' for x in xs))
print(f'sum\t{tot:.4f}')
print(f'peak_gb\t{mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory() / 1e9:.3f}')
