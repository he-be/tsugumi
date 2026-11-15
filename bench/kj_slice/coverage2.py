"""全セット (A, B, C, D) のルーター集計で、残すエキスパートの選び方を比べる。確かめる用は C (どの選び方にも使わない)。
    python coverage2.py   (stats_full.npz = A, stats_B.npz, stats_C.npz, stats_D.npz を読む。keep*.json を書く)"""
import json
import numpy as np
S = {}
for name, f in (("A", "stats_full.npz"), ("B", "stats_B.npz"), ("C", "stats_C.npz"), ("D", "stats_D.npz")):
    z = np.load(f); S[name] = dict(cnt=z["cnt"].sum(1), mass=z["mass"].sum(1), plays=[str(p) for p in z["plays"]])   # [plays, L, E]
L, E = S["A"]["cnt"].shape[1:]
cat = lambda names, key: np.concatenate([S[n][key] for n in names])
def keepset(score, K):                      # score [L, E] -> bool mask
    m = np.zeros((L, E), bool); np.put_along_axis(m, np.argsort(-score, axis=1)[:, :K], True, 1); return m
def kept(mask, cnt):                        # cnt [plays, L, E] -> overall fraction, worst layer, per-play fractions
    tot = cnt.sum(0); per_layer = (tot * mask).sum(1) / tot.sum(1)
    per_play = (cnt * mask).sum((1, 2)) / cnt.sum((1, 2))
    return (tot * mask).sum() / tot.sum(), per_layer.min(), per_play
def dump(mask, path): json.dump({str(l): [int(e) for e in np.flatnonzero(mask[l])] for l in range(L)}, open(path, "w"))
C = S["C"]["cnt"]
print("plays: A %d B %d C %d D %d" % tuple(len(S[n]["plays"]) for n in "ABCD"))
print("\n== 1. 選ぶ用の枚数 (確かめる用は C の %d 枚)。値は C で残る選択の割合 / 最悪の層 / 最悪の写真" % len(C))
pools = {"A の偶数番 48 枚 (48 の版)": S["A"]["mass"][0::2], "A 96 枚": S["A"]["mass"], "A+B 192 枚": cat("AB", "mass"), "A+B+D": cat("ABD", "mass")}
print("K\t" + "\t".join(pools))
for K in (64, 72, 80, 88, 96, 104):
    row = []
    for nm, m in pools.items():
        f, wl, pp = kept(keepset(m.sum(0), K), C); row.append("%.4f / %.3f / %.3f" % (f, wl, pp.min()))
    print("%d\t%s" % (K, "\t".join(row)))
print("\n== 2. 選ぶ基準 (A+B+D で選び、C で確かめる)")
mass, cnt = cat("ABD", "mass"), cat("ABD", "cnt")
pres = (cnt > 0).sum(0)                      # 何枚の写真で 1 回でも選ばれたか
for K in (72, 80, 88, 96):
    print("K=%d  重みの和 %.4f   選択回数 %.4f   選ばれた写真の枚数 %.4f   写真ごとに正規化した重みの平均 %.4f" % (K,
          kept(keepset(mass.sum(0), K), C)[0], kept(keepset(cnt.sum(0), K), C)[0], kept(keepset(pres + 1e-9 * cnt.sum(0), K), C)[0],
          kept(keepset((mass / mass.sum(2, keepdims=True)).sum(0), K), C)[0]))
print("\n== 3. 日付で分ける (全 %d 枚を名前 = 撮った時刻の順に並べ、前半で選んで後ろで確かめる)" % sum(len(S[n]["plays"]) for n in "ABCD"))
names = sum((S[n]["plays"] for n in "ABCD"), []); order = np.argsort(names)
allc, allm = cat("ABCD", "cnt")[order], cat("ABCD", "mass")[order]; names = [names[i] for i in order]
n = len(names); days = sorted({x[:8] for x in names}); print("days:", {d: sum(x.startswith(d) for x in names) for d in days})
rng = np.random.default_rng(0)
for frac in (0.25, 0.5, 0.75):
    cut = int(n * frac); tail = slice(int(n * 0.75), n) if frac < 0.75 else slice(cut, n)
    f_time = kept(keepset(allm[:cut].sum(0), 80), allc[tail])[0]
    rs = []
    for _ in range(5):
        p = rng.permutation(n); test = p[: (tail.stop - tail.start)]; train = p[(tail.stop - tail.start):][:cut]
        rs.append(kept(keepset(allm[train].sum(0), 80), allc[test])[0])
    print("K=80  前 %d 枚 (〜%s) で選び、後ろ %d 枚 (%s〜) で確かめる: %.4f   同じ枚数を無作為に分けた場合: %.4f (5 回の平均、%.4f〜%.4f)" % (
        cut, names[cut - 1], tail.stop - tail.start, names[tail.start], f_time, np.mean(rs), min(rs), max(rs)))
print("\n== 4. 日ごと (その日を除いた全部で K=80 を選び、その日で確かめる)")
for d in days:
    idx = np.array([x.startswith(d) for x in names])
    if idx.sum() < 5: continue
    f, wl, pp = kept(keepset(allm[~idx].sum(0), 80), allc[idx]); print("%s  %3d 枚  %.4f  最悪の写真 %.3f" % (d, idx.sum(), f, pp.min()))
print("\n== 5. 写真ごとのばらつき (A+B+D で選んだ K、C の写真ごとに残る選択の割合)")
for K in (72, 80, 88, 96):
    pp = kept(keepset(mass.sum(0), K), C)[2]
    print("K=%d  最小 %.3f  5%% 点 %.3f  中央値 %.3f  最大 %.3f   最悪の 3 枚 %s" % (K, pp.min(), np.quantile(pp, .05), np.median(pp), pp.max(),
          [S["C"]["plays"][i] for i in np.argsort(pp)[:3]]))
print("\n== 6. 残すリストどうしの重なり (K=80、層ごとの共通個数の平均)")
old = keepset(S["A"]["mass"][0::2].sum(0), 80); new = keepset(mass.sum(0), 80); allk = keepset(cat("ABCD", "mass").sum(0), 80)
print("48 の版 と A+B+D: %.1f / 80   A+B+D と 全部: %.1f / 80" % ((old & new).sum(1).mean(), (new & allk).sum(1).mean()))
for K in (64, 72, 80, 88, 96):
    dump(keepset(mass.sum(0), K), f"keepABD_{K}.json")
dump(allk, "keepALL_80.json"); dump(keepset(cnt.sum(0), 80), "keepABDcount_80.json")
