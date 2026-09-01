#!/usr/bin/env python3
"""48 §5a/§5b/§11/§12 の数字を bench/mtp48/ の一次ログから出し直す (導出のみ)。

新しい測定はしない。入力は
  bench/mtp48/gate_only_5_invocations.log   -- ゲート (ABBA, 片側 n=50)
  bench/mtp48/p1_p4_full_trials5_run{1..4}.log -- P-1〜P-4 (反復 5 × 4 run)
"""
import glob
import random
import re
import statistics as st

GATE = "bench/mtp48/gate_only_5_invocations.log"
RUNS = sorted(glob.glob("bench/mtp48/p1_p4_full_trials5_run*.log"))
PAGE = 16384
STRIDE = 3358720          # 205 pages
EXPERTS = 8               # top-k, 47 §3
KERNEL_MS = 0.25          # 48 §3, charged to the mmap side only


def quantile(values, f):
    s = sorted(values)
    i = f * (len(s) - 1)
    lo = int(i)
    return s[lo] + (i - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def read_gate():
    """Per invocation, the two 10-sample lists. ABBA order inside a round is
    pread(1) mmap(2) mmap(3) pread(4), so index 0,2,4.. is each arm's first
    slot in its round and 1,3,5.. is its second."""
    text = open(GATE).read()
    pread, mmap_ = [], []
    for block in text.split("=== gate-only invocation ")[1:]:
        lines = block.splitlines()
        for i, line in enumerate(lines):
            if "pread, parallel" in line:
                pread.append([float(x) for x in lines[i + 1].split()])
            if "mmap + F_RDADVISE" in line:
                mmap_.append([float(x) for x in lines[i + 1].split()])
    return pread, mmap_


def read_p4():
    """P-4's 'advise, wait for residency, then run' split, over the 4 runs."""
    wait, run = [], []
    for path in RUNS:
        section = open(path).read().split(
            "advise, wait for residency, then run")[1].split("--- ")[0]
        wait += [float(x) for x in re.search(r"of which wait\s+(.+)", section)
                 .group(1).replace("ms", "").split("/")]
        run += [float(x) for x in re.search(r"of which run\s+(.+)", section)
                .group(1).replace("ms", "").split("/")]
    return wait, run


def main():
    pread, mmap_ = read_gate()
    flat_p = [x for row in pread for x in row]
    flat_m = [x for row in mmap_ for x in row]

    print("=== §5 the gate, pooled (n=%d per side) ===" % len(flat_p))
    for name, v in (("pread", flat_p), ("mmap ", flat_m)):
        print("  %s median %.3f ms  IQR %.3f-%.3f  min %.2f  max %.2f"
              % (name, st.median(v), quantile(v, .25), quantile(v, .75),
                 min(v), max(v)))
    pooled = st.median(flat_m) / st.median(flat_p)
    print("  ratio of medians      %.4f   (difference %.3f ms)"
          % (pooled, st.median(flat_m) - st.median(flat_p)))
    print("  kernel-deducted       %.4f" % ((st.median(flat_m) - KERNEL_MS)
                                            / st.median(flat_p)))

    paired = []
    for p, m in zip(pread, mmap_):
        for r in range(len(p) // 2):
            paired.append(st.median(m[2 * r:2 * r + 2])
                          / st.median(p[2 * r:2 * r + 2]))
    print("  paired per round      %.4f  (n=%d, %.2f..%.2f)"
          % (st.median(paired), len(paired), min(paired), max(paired)))

    print("\n=== §5a the position effect inside an ABBA round ===")
    slots = {"pread first (after pread)": [x for r in pread for x in r[0::2]],
             "pread second (after mmap)": [x for r in pread for x in r[1::2]],
             "mmap first (after pread)": [x for r in mmap_ for x in r[0::2]],
             "mmap second (after mmap)": [x for r in mmap_ for x in r[1::2]]}
    for name, v in slots.items():
        print("  %-27s n=%d median %.3f ms" % (name, len(v), st.median(v)))
    print("  delta pread %+.3f ms, mmap %+.3f ms"
          % (st.median(slots["pread second (after mmap)"])
             - st.median(slots["pread first (after pread)"]),
             st.median(slots["mmap second (after mmap)"])
             - st.median(slots["mmap first (after pread)"])))
    print("  position-matched ratios: after-pread %.3f   after-mmap %.3f"
          % (st.median(slots["mmap first (after pread)"])
             / st.median(slots["pread first (after pread)"]),
             st.median(slots["mmap second (after mmap)"])
             / st.median(slots["pread second (after mmap)"])))

    print("\n=== §5b bootstrap on the ratio of medians (20,000 resamples) ===")
    random.seed(0)
    boot = []
    for _ in range(20000):
        a = [random.choice(flat_p) for _ in flat_p]
        b = [random.choice(flat_m) for _ in flat_m]
        boot.append(st.median(b) / st.median(a))
    boot.sort()
    over = sum(1 for x in boot if x > 1.20) / len(boot)
    print("  95%% CI %.3f .. %.3f   P(ratio > 1.20 gate) = %.2f"
          % (boot[int(.025 * len(boot))], boot[int(.975 * len(boot))], over))

    print("\n=== §11 where the cold cost goes (P-4 'wait for residency') ===")
    wait, run = read_p4()
    print("  wait for residency   n=%d median %.2f ms" % (len(wait), st.median(wait)))
    print("  run with 1640/1640 pages already in core  n=%d median %.2f ms"
          % (len(run), st.median(run)))
    fault_ms = st.median(run) - KERNEL_MS
    pages = EXPERTS * STRIDE // PAGE
    per_page = fault_ms / pages
    print("  minus the %.2f ms kernel -> %.2f ms over %d pages = %.2f us/page"
          % (KERNEL_MS, fault_ms, pages, per_page * 1e3))

    print("\n=== §12 production extrapolation (導出, serialized in-band) ===")
    for label, experts in (
            ("first touch only (mtp46 trace, n=1)", 21.9),
            ("as many pages as today's misses (45 §4)", 38.45),
            ("every request (240/tok, no mapping survives)", 240.0)):
        pg = experts * (STRIDE / PAGE)
        print("  %-45s %7.0f pg/tok -> %6.2f ms/tok" % (label, pg, pg * per_page))
    print("  %-45s %25.2f ms/tok" % ("today's io (45 §4)", 14.21))


if __name__ == "__main__":
    main()
