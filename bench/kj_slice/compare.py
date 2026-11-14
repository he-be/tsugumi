"""腕ごとの答えをフルモデルと比べる。 python compare.py <ref-arm> <arm> ...
greedy: フルの greedy と文字列が完全一致した割合、JSON として同じ値の割合、文字の類似度 (difflib)、選択 (pick) の値が同じ割合。
sample: JSON として読めない数、上限で切れた数、生成トークンの中央値と最大。"""
import difflib, json, statistics as st, sys
ref, arms = sys.argv[1], sys.argv[2:]
load = lambda a, m: [json.loads(l) for l in open(f"ans_{a}_{m}.jsonl", encoding="utf-8")]
def js(t):
    try: return json.loads(t)
    except ValueError: return None
R = load(ref, "greedy")
kinds = list(dict.fromkeys(r["name"] for r in R))
print("arm\tVRAM\t| greedy vs %s: exact (%s)\tsame JSON value\tchar sim\t| sample: invalid\tcut at max\tpredicted_n median/max (%s)\ttok/s" % (ref, " ".join(kinds), " ".join(kinds)))
for a in arms:
    G, S = load(a, "greedy"), load(a, "sample")
    assert [(g["play"], g["name"]) for g in G] == [(r["play"], r["name"]) for r in R]
    ex = [sum(g["text"] == r["text"] for g, r in zip(G, R) if r["name"] == k) / sum(r["name"] == k for r in R) for k in kinds]
    sj = [sum(js(g["text"]) is not None and js(g["text"]) == js(r["text"]) for g, r in zip(G, R) if r["name"] == k) / sum(r["name"] == k for r in R) for k in kinds]
    sim = [st.mean(difflib.SequenceMatcher(None, g["text"], r["text"]).ratio() for g, r in zip(G, R) if r["name"] == k) for k in kinds]
    inv = sum(not s["valid"] for s in S) ; cut = sum(s["finish"] == "length" for s in S)
    ginv = sum(not g["valid"] for g in G)
    pn = ["%d/%d" % (st.median(s["predicted_n"] for s in S if s["name"] == k), max(s["predicted_n"] for s in S if s["name"] == k)) for k in kinds]
    tps = st.median(s["predicted_tps"] for s in S if s["predicted_n"] >= 16)
    try: vram = open(f"vram-{a}.txt").read().split(",")[1].strip()
    except OSError: vram = "-"
    print("%s\t%s\t| %s\t%s\t%s\t| %d/%d (greedy %d)\t%d\t%s\t%.0f" % (a, vram, " ".join("%.2f" % x for x in ex), " ".join("%.2f" % x for x in sj),
          " ".join("%.2f" % x for x in sim), inv, len(S), ginv, cut, " ".join(pn), tps))
