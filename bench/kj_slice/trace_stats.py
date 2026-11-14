"""ルーター記録 (layer, n_expert, n_tokens, float32 probs) を読み、play ごとに層 × エキスパートの
選択回数と正規化後の重みの和を出す。 python trace_stats.py <trace.bin> <arm.jsonl> <out.npz>"""
import json, sys
import numpy as np
tr, arm, outp = sys.argv[1:4]
rows = [json.loads(l) for l in open(arm, encoding="utf-8")]
plays = list(dict.fromkeys(r["play"] for r in rows))
kinds = list(dict.fromkeys(r["name"] for r in rows))
L, E, U = 30, 128, 8
cnt = np.zeros((len(plays), len(kinds), L, E)); mass = np.zeros_like(cnt); ntok = np.zeros((len(plays), len(kinds)))
top8mass = np.zeros((L,)); top8n = 0
f = open(tr, "rb")
for r in rows:
    pi, ki = plays.index(r["play"]), kinds.index(r["name"])
    f.seek(r["trace"][0]); end = r["trace"][1]
    while f.tell() < end:
        l, e, n = np.frombuffer(f.read(12), dtype=np.int32)
        if e == 1:  # the "(reshaped)" view of the same probs [1, n_expert, n_tokens]: same bytes again, skip
            f.seek(4 * E * last_n, 1); continue
        last_n = int(n)
        p = np.frombuffer(f.read(4 * e * n), dtype=np.float32).reshape(n, e)
        idx = np.argpartition(-p, U - 1, axis=1)[:, :U]
        w = np.take_along_axis(p, idx, axis=1)
        top8mass[l] += w.sum()
        w = w / w.sum(axis=1, keepdims=True)
        np.add.at(cnt[pi, ki, l], idx.ravel(), 1); np.add.at(mass[pi, ki, l], idx.ravel(), w.ravel())
        if l == 0: ntok[pi, ki] += n; top8n += n
np.savez_compressed(outp, cnt=cnt, mass=mass, ntok=ntok, plays=np.array(plays), kinds=np.array(kinds), top8mass=top8mass / max(top8n, 1))
print("plays", len(plays), "tokens", int(ntok.sum()), "per kind", dict(zip(kinds, ntok.sum(0).astype(int))))
print("mean prob mass inside top-8, by layer:", np.round(top8mass / max(top8n, 1), 3))
