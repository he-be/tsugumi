"""選ぶ用 (偶数番の play) の集計で層ごとに上位 K を残し、確かめる用 (奇数番) でどれだけ覆うかを見る。keep_K.json も書く。"""
import json, sys
import numpy as np
z = np.load("stats_full.npz"); cnt, mass = z["cnt"], z["mass"]
tr, te = cnt[0::2].sum((0, 1)), cnt[1::2].sum((0, 1))          # [L, E] selection counts
mtr, mte = mass[0::2].sum((0, 1)), mass[1::2].sum((0, 1))
L, E = tr.shape
print("tokens train/test (layer 0):", int(tr[0].sum() / 8), int(te[0].sum() / 8))
order = np.argsort(-mtr, axis=1)
srt = np.sort(te, axis=1)[:, ::-1].cumsum(1) / te.sum(1, keepdims=True)
print("test-set oracle: experts needed per layer to cover 90%% / 99%% of selections: mean %.1f / %.1f  (max %d / %d)" % (
    (srt < 0.90).sum(1).mean() + 1, (srt < 0.99).sum(1).mean() + 1, (srt < 0.90).sum(1).max() + 1, (srt < 0.99).sum(1).max() + 1))
print("never selected in train, per layer: mean %.1f  max %d;   in all 96 plays: mean %.1f" % ((tr == 0).sum(1).mean(), (tr == 0).sum(1).max(), ((tr + te) == 0).sum(1).mean()))
print("K\tGiB(Q4_0)\tsel kept (test) mean/min-layer\tweight kept mean/min-layer\tby kind sel kept (read item pick deed)")
kinds = list(z["kinds"])
for K in (120, 112, 104, 96, 88, 80, 72, 64, 48):
    keep = order[:, :K]
    m = np.zeros((L, E), bool); np.put_along_axis(m, keep, True, 1)
    sel = (te * m).sum(1) / te.sum(1); wt = (mte * m).sum(1) / mte.sum(1)
    bk = [((cnt[1::2, k].sum(0) * m).sum() / cnt[1::2, k].sum()) for k in range(len(kinds))]
    print("%d\t%.2f\t%.4f / %.4f\t%.4f / %.4f\t%s" % (K, 1.29 + 11.96 * K / 128, sel.mean(), sel.min(), wt.mean(), wt.min(), " ".join("%.4f" % b for b in bk)))
    json.dump({str(l): sorted(int(e) for e in keep[l]) for l in range(L)}, open(f"keep_{K}.json", "w"))
# all-96 selection (for the final candidate) and a random control
allm = np.argsort(-(mtr + mte), axis=1)
for K in (96, 80, 64):
    json.dump({str(l): sorted(int(e) for e in allm[l, :K]) for l in range(L)}, open(f"keepall_{K}.json", "w"))
rng = np.random.default_rng(0)
json.dump({str(l): sorted(int(e) for e in rng.permutation(E)[:96]) for l in range(L)}, open("keeprand_96.json", "w"))
m = np.zeros((L, E), bool)
for l in range(L): m[l, json.load(open("keeprand_96.json"))[str(l)]] = True
print("random 96: sel kept %.4f" % ((te * m).sum() / te.sum()))
# per-layer picture for K=96
keep = order[:, :96]; m = np.zeros((L, E), bool); np.put_along_axis(m, keep, True, 1)
print("K=96 sel kept by layer:", np.round((te * m).sum(1) / te.sum(1), 3))
