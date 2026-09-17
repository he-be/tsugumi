"""0 段: Q4_1 の行と BF16 の行の誤差。引数はトークンファイル。取っていない行があれば止まる。"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from plerows import rows_for, load
Q = os.path.expanduser('~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/ple/Qwen3.8-Flash-Next-PLE-Q4_1.gguf'); Q0 = 3552
B = os.path.expanduser('~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/ple-bf16-sparse/Qwen3.8-Flash-Next-PLE-BF16-sparse.gguf'); B0 = 3744
have = set(np.load(B + '.rows.npy').tolist())
qm = np.memmap(Q, np.uint8, 'r'); bm = np.memmap(B, np.uint8, 'r')
def q41(rows):
    raw = np.stack([qm[Q0 + r*100: Q0 + r*100 + 100] for r in rows]).reshape(len(rows), 5, 20)
    d = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32)
    m = raw[:, :, 2:4].copy().view(np.float16).astype(np.float32)
    qs = raw[:, :, 4:20]
    v = np.concatenate([qs & 0x0F, qs >> 4], axis=2).astype(np.float32)
    return (d * v + m).reshape(len(rows), 160)
def bf16(rows):
    raw = np.stack([bm[B0 + r*320: B0 + r*320 + 320] for r in rows]).copy().view(np.uint16).astype(np.uint32)
    return (raw << 16).view(np.float32).reshape(len(rows), 160)
print('| プロンプト | トークン | 行 (延べ) | 相対誤差 ‖q−b‖/‖b‖ 中央値 / p99 / 最大 | コサイン 最小 | ‖b‖ 中央値 | 誤差の和の相対 (16 行を連結した emb 単位) 中央値 / 最大 |')
for p in sys.argv[1:]:
    t = load(p); rows = rows_for(t)
    miss = set(rows) - have
    if miss: print(p, 'missing', len(miss)); continue
    uq = sorted(set(rows)); qa = q41(uq); ba = bf16(uq)
    idx = {r: i for i, r in enumerate(uq)}
    nb = np.linalg.norm(ba, axis=1); rel = np.linalg.norm(qa - ba, axis=1) / nb
    cos = (qa * ba).sum(1) / (np.linalg.norm(qa, axis=1) * nb)
    ii = np.array([idx[r] for r in rows]).reshape(len(t), 16)
    ebq = qa[ii].reshape(len(t), -1); ebb = ba[ii].reshape(len(t), -1)
    erel = np.linalg.norm(ebq - ebb, axis=1) / np.linalg.norm(ebb, axis=1)
    name = p.split('qwen38/')[-1]
    print(f'| {name} | {len(t)} | {len(rows)} | {np.median(rel):.4f} / {np.percentile(rel,99):.4f} / {rel.max():.4f} | {cos.min():.4f} | {np.median(nb):.3f} | {np.median(erel):.4f} / {erel.max():.4f} |')
