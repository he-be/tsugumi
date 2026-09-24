#!/usr/bin/env python3
"""B1 floor table (docs/m6-prefill/04-GATES.md B1): per-stage lower bound vs measured.

  python3 bench/m6/floor_table.py <level-1 profile dir> <level-2 profile dir> [TFLOP/s] [GB/s]

Floor = max(FLOP / compute ceiling, bytes / bandwidth ceiling), from the manifest
arch of gemma4-qat-sym (constants below) and T = 2478. Bytes are the minimum a
stage must move: int4 g32 weights read once (0.5625 B/param), fp16 activations
read and written once. Measured = level-1 median per group; the attn group is
split by the level-2 fractions of the median run, because level 2 inflates.
Everything printed is derived; the evidence is the two directories.
"""
import re, statistics, sys
from pathlib import Path

T, H, L, LF = 2478, 2816, 30, 5
LS = L - LF
W = 0.5 + 2 / 32
flops = float(sys.argv[3]) * 1e12 if len(sys.argv) > 3 else 18.61e12
bw = float(sys.argv[4]) * 1e9 if len(sys.argv) > 4 else 144.4e9

sw = sum(min(t, 1024) for t in range(1, T + 1))   # sliding keys attended
fc = T * (T + 1) // 2                              # causal keys attended
qkv = [(LS, 4096 + 2048 + 2048, 4096), (LF, 8192 + 1024, 8192)]  # full: attentionKEqV
floor = {
    'qkv': (sum(n * 2 * T * H * o for n, o, _ in qkv), sum(n * (H * o * W + T * H * 2 + T * o * 2) for n, o, _ in qkv)),
    'oproj': (sum(n * 2 * T * q * H for n, _, q in qkv), sum(n * (q * H * W + T * q * 2 + T * H * 2) for n, _, q in qkv)),
    'rope': (0, LS * T * (4096 + 2048) * 4 + LF * T * (8192 + 1024) * 4),
    'attn.swa': (LS * 4 * 16 * 256 * sw, LS * T * (4096 + 2048 + 2048 + 4096) * 2),
    'attn.full': (LF * 4 * 16 * 512 * fc, LF * T * (8192 + 1024 + 8192) * 2),
    'post': (L * 2 * T * H * 128, L * (4 * T * H * 2 + 128 * H * 2)),
    # post split (level 2 from 24's follow-up on): the norms read hidden and
    # the attention output and write hidden plus the three FFN inputs; the
    # router reads its input and its bf16 weight once.
    'post.norm': (0, L * 6 * T * H * 2),
    'post.router': (L * 2 * T * H * 128, L * (T * H * 2 + 128 * H * 2)),
    'shared': (L * 2 * T * H * 2112 * 3, L * (3 * H * 2112 * W + 2 * T * H * 2)),
    'moe': (L * 2 * T * 8 * H * 704 * 3, L * (128 * 3 * H * 704 * W + 2 * T * H * 2)),
    'tail': (0, L * 3 * T * H * 2),
    # moe split, for the stages as they are built (gate/up written to scratch,
    # gelu reads both and writes act, down writes one partial per pair).
    'moe.gateup': (L * 2 * T * 8 * H * 704 * 2, L * (128 * 2 * H * 704 * W + T * H * 2 + T * 8 * 704 * 2 * 2)),
    'moe.gelu': (0, L * T * 8 * 704 * 2 * 3),
    'moe.down': (L * 2 * T * 8 * 704 * H, L * (128 * H * 704 * W + T * 8 * 704 * 2 + T * 8 * H * 2)),
}

def lines(d, prefix):
    out = []
    for log in sorted(Path(d).glob('run-*.log')):
        for s in log.read_text().splitlines():
            if s.startswith(prefix):
                out.append({k: float(v) for k, v in re.findall(r'([\w.]+)=([\d.]+)s', s)})
    return out

l1 = lines(sys.argv[1], '[prefill gpu attn=')
l2 = lines(sys.argv[2], '[prefill gpu attn embed=')
l2m = lines(sys.argv[2], '[prefill gpu moe ')
med = {k: statistics.median(r[k] for r in l1) for k in ('attn', 'shared', 'moe', 'tail')}
mid = sorted(l2, key=lambda r: r['total'])[len(l2) // 2]
parts = {k: v for k, v in mid.items() if k != 'total'}
meas = {k: med['attn'] * v / sum(parts.values()) for k, v in parts.items()}
meas.update({k: med[k] for k in ('shared', 'moe', 'tail')})
if 'post.norm' in meas:
    meas['post'] = meas['post.norm'] + meas['post.router']
if l2m:
    # Same rule as attn: level-2 fractions of the median run, on level 1's moe.
    midm = sorted(l2m, key=lambda r: r['total'])[len(l2m) // 2]
    for k in ('gateup', 'gelu', 'down', 'other'):
        meas['moe.' + k] = med['moe'] * midm.get(k, 0) / midm['total']

print(f'ceilings: {flops / 1e12:.2f} TFLOP/s, {bw / 1e9:.1f} GB/s; level-1 runs {len(l1)}, level-2 runs {len(l2)}')
print('stage\tTFLOP\tGB\tcompute_s\tbytes_s\tfloor_s\tmeasured_s\tgap_s')
rows = []
for k, (f, b) in floor.items():
    if k not in meas:
        continue
    lb = max(f / flops, b / bw)
    rows.append((meas[k] - lb, k, f, b, lb))
for gap, k, f, b, lb in sorted(rows, reverse=True):
    print(f'{k}\t{f / 1e12:.3f}\t{b / 1e9:.3f}\t{f / flops:.3f}\t{b / bw:.3f}\t{lb:.3f}\t{meas[k]:.3f}\t{gap:.3f}')
rest = {k: v for k, v in meas.items() if k not in floor}
print('not in table (no floor derived):', ', '.join(f'{k} {v:.3f}' for k, v in rest.items()))
top = [r for r in rows if '.' not in r[1] or r[1].startswith('attn.')]
print(f'sum floor {sum(r[4] for r in top):.3f}  sum measured {sum(meas[r[1]] for r in top):.3f} (stages without a split row)')
