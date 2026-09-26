#!/usr/bin/env python3
"""Where a small round's fixed seconds go (docs/qfn-prefill/04 Q8), from the Q38_ROUND_LOG files already taken.

  q8_fixed_cost.py <result dir> [...]

For each dir, over its b2-*/b3-* .rounds.jsonl: the chunks by size (T 1 / 2-31 / 32-255 / 256-2047 / 2048), with the
chunk's wall, the stages of its StepProfile, the MBP's disk read over the chunk (the `disk_read` difference from the
row before; the first chunk of a file has none), and the M6 wait on the pair. The 1-token chunks are split by what ran
before them in the round (a chunk of 32 and up, or a smaller one). Then the rounds' fit prefill_s = a + new / r, with
the chunks under 32 tokens (the checkpoint cuts: P - 1 and U - m, Qwen38ServerSession.checkpointPositions) and without.
"""
import glob
import json
import statistics
import sys

BUCKETS = [(1, 1), (2, 31), (32, 255), (256, 2047), (2048, 2048)]


def rng(xs):
    xs = sorted(x for x in xs if x is not None)
    return f"{statistics.median(xs):.3f} ({xs[0]:.3f}-{xs[-1]:.3f})" if xs else "-"


def fit(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    r2 = 1 - sum((y - a - b * x) ** 2 for x, y in pts) / sum((y - my) ** 2 for y in ys)
    return a, 1 / b, r2


for d in sys.argv[1:]:
    chunks, rounds = {}, []
    for f in sorted(glob.glob(d + "/b[23]-*.rounds.jsonl")):
        prev_read, cur = None, []
        for line in open(f):
            e = json.loads(line)
            if e["event"] == "chunk":
                T = e["T"]
                lo, hi = next(b for b in BUCKETS if b[0] <= T <= b[1])
                key = f"T {lo}-{hi}" if lo != hi else f"T {T}"
                if T == 1:
                    key += " after " + ("32+" if cur and cur[-1][0] >= 32 else "small")
                p = e["profile"]
                read = (e["disk_read"] - prev_read) / 1e9 if prev_read is not None else None
                chunks.setdefault(key, []).append((e["wall_s"], read, p["pre"], p["pre_gpu"], p["route"],
                                                   p["routed_gpu"], e.get("mtp_s", 0), e.get("pair_wait_s", 0)))
                cur.append((T, e["wall_s"]))
            elif e["event"] == "run":
                n = sum(T for T, _ in cur)
                tail = sum(w for T, w in cur if T < 32)
                rounds.append((n, e["prefill_s"], tail))
                cur = []
            if "disk_read" in e:
                prev_read = e["disk_read"]
    print(f"== {d}")
    print("chunk\tn\twall_s\tread_GB\tpre_s\tpre_gpu_s\troute_s\trouted_gpu_s\tmtp_s\tpair_wait_s")
    for key in sorted(chunks, key=lambda k: (int(k.split()[1].split("-")[0]), k)):
        v = chunks[key]
        print(key, len(v), *(rng([x[i] for x in v]) for i in range(8)), sep="\t")
    a, r, r2 = fit([(n, s) for n, s, _ in rounds])
    a2, r_2, r22 = fit([(n, s - t) for n, s, t in rounds])
    print(f"rounds {len(rounds)}: prefill_s = {a:.2f} + new / {r:.1f} (R2 {r2:.3f}); "
          f"without the chunks under 32 tokens {a2:.2f} + new / {r_2:.1f} (R2 {r22:.3f}); "
          f"those chunks {sum(t for *_, t in rounds):.2f} s of {sum(s for _, s, _ in rounds):.2f} s")
