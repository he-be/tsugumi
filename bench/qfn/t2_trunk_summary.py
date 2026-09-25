#!/usr/bin/env python3
# docs/qfn-prefill/25 §0-3: medians over steps 50..200 (1-based step lines) of the MTP spec step lines of --qwen38-generate.
#   bench/qfn/t2_trunk_summary.py <evidence dir>...    # every run-<i>.log; I/O = kernel>gpu + advise, non-I/O = verify - I/O,
#                                                        # wait = verify pre wall - verify pre GPU. Per dir: the n runs and their median / range.
import re, sys, glob, statistics as st
F = re.compile(r"step (\S+) = draft (\S+) \(T=\d+ .*?\) \+ verify (\S+) \(pre (\S+) \[gpu (\S+)\] route (\S+) routed (\S+) \[gpu (\S+)\] head (\S+)\) \+ rollback (\S+)"
               r".*?\| route topk (\S+) views (\S+) advise (\S+) \(miss \S+\) calls \d+ \| routed commit>kernel \S+ kernel>gpu (\S+) \(behind shared \S+ shared gpu \S+\) gpu (\S+) after \S+ \| experts (\d+)")
K = ["step", "draft", "verify", "preWall", "preGPU", "route", "routed", "routedGPUv", "head", "rollback", "topk", "views", "advise", "k>gpu", "gpu", "experts"]
D = ["io", "nonIO", "wait"]
def run(path):
    rows = [list(map(float, m.groups())) for m in map(F.search, open(path, errors="replace")) if m]
    w = rows[49:200]
    if not w: return None
    col = {k: [r[i] for r in w] for i, k in enumerate(K)}
    col["io"] = [a + b for a, b in zip(col["k>gpu"], col["advise"])]
    col["nonIO"] = [v - io for v, io in zip(col["verify"], col["io"])]
    col["wait"] = [a - b for a, b in zip(col["preWall"], col["preGPU"])]
    return {k: st.median(v) for k, v in col.items()}, len(rows), len(w)
for d in sys.argv[1:]:
    print(f"== {d}")
    rs = []
    for p in sorted(glob.glob(f"{d}/run-*.log")):
        r = run(p)
        if not r: print(f"  {p.split('/')[-1]}: no step lines"); continue
        m, n, nw = r; rs.append(m)
        acc = next((l.strip() for l in open(p) if "accepted by position" in l), "")
        print(f"  {p.split('/')[-1]}: steps {n} (window {nw}) " + " ".join(f"{k} {m[k]:.1f}" for k in K + D))
        print(f"    {acc[:160]}")
    if rs:
        print("  median " + " ".join(f"{k} {st.median([r[k] for r in rs]):.1f}" for k in K + D))
        print("  range  " + " ".join(f"{k} {max(r[k] for r in rs) - min(r[k] for r in rs):.1f}" for k in K + D))
