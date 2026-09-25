#!/usr/bin/env python3
"""One line of summary-b3.tsv (docs/qfn-prefill/04 Q7) for pair_bench.sh's 本文 3.

  q7_summary.py --header [--pair]
  q7_summary.py <b3-rN dir> <round> <exit> <swapouts> <pageins MB> [m6 int MB, m6 ext MB, m6 swapouts]

From <dir>/turns.jsonl (ToolLoopCheck's checks per turn), <dir>/rounds.jsonl (rounds, generated tokens, MTP drafts
accepted/proposed) and <dir>.rounds.jsonl (Q38_ROUND_LOG: each run's decode steps `step_s`, each round's `pair_begin`,
the checkpoints restored and captured).
"""
import json
import statistics
import sys

COLS = ["round", "turns", "turns_ok", "rounds", "generated", "decode_sum_s", "steps", "step_med_ms", "step_p90_ms",
        "accepted", "proposed", "prefill_sum_s", "begin_n", "begin_sum_s", "begin_max_s", "restores", "captures",
        "pageins_MB", "swapouts", "exit"]


def lines(path):
    try:
        return [json.loads(x) for x in open(path) if x.strip()]
    except FileNotFoundError:
        return []


if sys.argv[1] == "--header":
    extra = ["m6_int_MB", "m6_ext_MB", "m6_swapouts"] if "--pair" in sys.argv else []
    print("\t".join(COLS + extra))
    sys.exit(0)

d, rnd, rc, so, pin, *m6 = sys.argv[1:]
turns = lines(d + "/turns.jsonl")
rounds = lines(d + "/rounds.jsonl")
log = lines(d + ".rounds.jsonl")
runs = [x for x in log if x.get("event") == "run"]
steps = [s for x in runs for s in x.get("step_s", [])]
begins = [x["s"] for x in log if x.get("event") == "pair_begin"]
acc = prop = 0
for x in rounds:
    a, _, p = str(x.get("draft", "0/0")).partition("/")
    acc += int(a or 0)
    prop += int(p or 0)
q = statistics.quantiles(steps, n=10) if len(steps) >= 2 else [0] * 9
row = [rnd, len(turns), sum(1 for t in turns if all(t.get("checks", {}).values())), len(rounds),
       sum(x.get("generated", 0) for x in rounds), f"{sum(x.get('decode_s', 0) for x in runs):.3f}", len(steps),
       f"{statistics.median(steps) * 1000:.1f}" if steps else "", f"{q[8] * 1000:.1f}" if steps else "", acc, prop,
       f"{sum(x.get('prefill_s', 0) for x in runs):.3f}", len(begins), f"{sum(begins):.3f}",
       f"{max(begins):.3f}" if begins else "0", sum(1 for x in log if x.get("event") == "restore"),
       sum(1 for x in log if x.get("event") == "capture"), pin, so, rc]
print("\t".join(str(v) for v in row + m6))
