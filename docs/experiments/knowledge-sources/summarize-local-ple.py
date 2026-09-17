#!/usr/bin/env python3
"""Summarize run-local-ple.sh: summarize-local-ple.py [root] [--answers]

Per arm x condition x question: rounds, tool calls, prompt (uncached) / generated tokens, wall, decode tok/s, draft
acceptance, per repeat. --answers prints every answer with its tool trace for grading.
"""
import json, pathlib, sys
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
root = pathlib.Path(args[0] if args else "scratch/knowledge-sources/qwen-local")
show = "--answers" in sys.argv

cells = defaultdict(list)   # (cond, question, arm) -> [(rep, turn, rounds)]
totals = defaultdict(lambda: defaultdict(float))
for tfile in sorted(root.glob("*/*/rep*/turns.jsonl")):
    arm, cond, rep = tfile.parts[-4], tfile.parts[-3], tfile.parts[-2]
    rounds = defaultdict(list)
    for line in open(tfile.parent / "rounds.jsonl"):
        r = json.loads(line)
        rounds[r["conversation"]].append(r)
    for line in open(tfile):
        t = json.loads(line)
        rs = rounds[t["conversation"]]
        cells[(cond, t["conversation"], arm)].append((rep, t, rs))
        k = totals[(cond, arm, rep)]
        k["turns"] += 1
        k["rounds"] += t["rounds"]
        k["wall"] += t["wall_s"]
        k["read"] += sum(r["prompt"] - r["cached"] for r in rs)
        k["gen"] += sum(r["generated"] for r in rs)
        k["prefill"] += sum(r.get("prefill_s") or 0 for r in rs)
        k["decode"] += sum(r.get("decode_s") or 0 for r in rs)
        k["wiki"] += t.get("wikipedia_steps") or 0
        k["fail"] += 0 if all(t["checks"].values()) else 1

print("## totals per run")
print("| cond | arm | rep | turns | rounds | wiki | read tok | gen tok | prefill s | decode s | wall s | gen tok/s | failed checks |")
print("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for (cond, arm, rep), k in sorted(totals.items()):
    tps = k["gen"] / k["decode"] if k["decode"] else 0
    print(f"| {cond} | {arm} | {rep} | {k['turns']:.0f} | {k['rounds']:.0f} | {k['wiki']:.0f} | {k['read']:.0f} | "
          f"{k['gen']:.0f} | {k['prefill']:.1f} | {k['decode']:.1f} | {k['wall']:.1f} | {tps:.2f} | {k['fail']:.0f} |")

print("\n## per question (rep: rounds / generated / wall)")
print("| cond | question | arm | runs |")
print("|---|---|---|---|")
for (cond, q, arm), runs in sorted(cells.items()):
    cells_txt = ", ".join(f"{rep[3:]}: {t['rounds']}r/{sum(r['generated'] for r in rs)}t/{t['wall_s']:.0f}s"
                          for rep, t, rs in sorted(runs, key=lambda x: x[0]))
    print(f"| {cond} | {q} | {arm} | {cells_txt} |")

if show:
    for (cond, q, arm), runs in sorted(cells.items()):
        for rep, t, rs in sorted(runs, key=lambda x: x[0]):
            print(f"\n=== {cond} {q} {arm} {rep}  ({t['rounds']} rounds, {t['wall_s']:.0f}s)")
            for r in rs:
                for c in r["calls"]:
                    print(f"  · {c}")
            print((t.get("answer") or "").strip())
