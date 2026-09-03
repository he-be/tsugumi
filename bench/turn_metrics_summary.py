#!/usr/bin/env python3
"""Read the app's turn-metrics.jsonl and print it as a table, then the
same rows bucketed by the speedometer so "more memory, faster" can be read
off real use instead of a lab run (docs/MAC_APP.md §4e).

    python3 bench/turn_metrics_summary.py            # the default file
    python3 bench/turn_metrics_summary.py path.jsonl # another file
"""
import json, os, statistics, sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Library/Application Support/Tsugumi/turn-metrics.jsonl")
rows = [json.loads(l) for l in open(path) if l.strip()]
rows = [r for r in rows if r.get("outcome") in ("finished", "tools") and r.get("generatedTokens", 0) >= 16]

def gb(b): return "—" if b is None else f"{b / 2**30:.1f}"
def f(x, d=1): return "—" if x is None else f"{x:.{d}f}"
def ssd_mb_per_tok(r):
    b, n = r.get("decodeDiskReadBytes"), r.get("generatedTokens") or 0
    return None if b is None or n <= 0 else b / 2**20 / n
def prefill_rate(r):
    p, s, c = r.get("promptTokens"), r.get("prefillSeconds"), r.get("cachedPromptTokens") or 0
    return None if not p or not s or s <= 0 else (p - c) / s

print(f"{len(rows)} rounds in {path}\n")
print(f"{'time':19} {'lvl':>4} {'res':>4} {'borrow':>6} {'prompt':>6} {'cached':>6} {'pf tok/s':>8} {'tok/s':>6} {'ssd MB/tok':>10} {'pf ssd MB':>9} {'net':7} ctx")
for r in rows:
    print(f"{r['recordedAt'][:19]:19} {f(r.get('headroomLevel'), 2):>4} {f(r.get('weightsResidentFraction'), 2):>4} "
          f"{gb(r.get('borrowableBytes')):>6} {r.get('promptTokens') or '—':>6} {r.get('cachedPromptTokens') or 0:>6} "
          f"{f(prefill_rate(r), 0):>8} {f(r.get('tokensPerSecond')):>6} {f(ssd_mb_per_tok(r), 1):>10} "
          f"{f(r['prefillDiskReadBytes'] / 2**20 if r.get('prefillDiskReadBytes') is not None else None, 0):>9} "
          f"{r.get('network', ''):7} {r.get('contextTokens')}")

def bucket(level):
    if level is None: return "?"
    return f"{int(level * 4) / 4:.2f}-{min(1, int(level * 4) / 4 + 0.25):.2f}"

groups = {}
for r in rows:
    groups.setdefault(bucket(r.get("headroomLevel")), []).append(r)
print("\nby speedometer (median, n):")
print(f"{'level':11} {'n':>3} {'tok/s':>7} {'pf tok/s':>9} {'residency':>9} {'ssd MB/tok':>10}")
for key in sorted(groups):
    g = groups[key]
    tok = statistics.median(r["tokensPerSecond"] for r in g)
    pf = [prefill_rate(r) for r in g if prefill_rate(r)]
    res = [r["weightsResidentFraction"] for r in g if r.get("weightsResidentFraction") is not None]
    ssd = [ssd_mb_per_tok(r) for r in g if ssd_mb_per_tok(r) is not None]
    print(f"{key:11} {len(g):>3} {tok:>7.1f} {f(statistics.median(pf), 0) if pf else '—':>9} "
          f"{f(statistics.median(res), 2) if res else '—':>9} {f(statistics.median(ssd), 1) if ssd else '—':>10}")
