#!/usr/bin/env python3
"""Compare the branch probes of two PLE tables at the same recorded decision points (docs/qwen38/32).

    ~/LLM/venv/bin/python Scripts/qwen38/replay_branches_report.py scratch/qwen38/branch33 [--draws]

Per point: P(`<tool_call>`) for each table and |ΔP|, the top-1 id of each, an approximate KL(bf16 ‖ q41) over the union
of both top-N lists (the rest of each distribution lumped into one bucket), and for drawn points the decoded draws
grouped by what they do (the call and its arguments, or the first line of text).
"""
import json, math, re, sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

root = Path(sys.argv[1])
show_draws = "--draws" in sys.argv
tok = Tokenizer.from_file(str(Path.home() / "LLM/Qwen3.8-Flash-Next-DS4-IQ2/tokenizer/tokenizer.json"))


def load(arm):
    out = {}
    for f in sorted((root / "probe" / arm).glob("*.log")):
        for line in open(f):
            if line.startswith('{"label"'):
                d = json.loads(line)
                out[d["label"]] = d
    return out


def kl(p_top, q_top):
    p, q = dict(p_top), dict(q_top)
    keys = set(p) | set(q)
    floor = 1e-6
    rp, rq = max(1 - sum(p.values()), floor), max(1 - sum(q.values()), floor)
    total = 0.0
    for k in keys:
        a = p.get(k, 0.0)
        # a token outside q's list has at most q's smallest listed probability
        b = q.get(k, min(min(q.values()), rq))
        if a > 0:
            total += a * math.log(a / max(b, floor))
    total += rp * math.log(rp / rq)
    return total


def action(ids):
    text = tok.decode(ids, skip_special_tokens=False)
    if text.startswith("<tool_call>"):
        calls = re.findall(r"<function=([^>]+)>(.*?)</function>", text, re.S)
        if not calls:
            return "CALL(unfinished) " + text[:60].replace("\n", " ")
        parts = []
        for name, body in calls:
            args = re.findall(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", body, re.S)
            parts.append(name + "(" + ", ".join(f"{k}={v}" for k, v in args) + ")")
        return " + ".join(parts)
    return "TEXT " + text.split("\n")[0][:50]


q41, bf16 = load("q41"), load("bf16")
labels = [l for l in bf16 if l in q41]
rows = []
for label in labels:
    a, b = q41[label], bf16[label]
    rows.append((label, a["p_tool_call"], b["p_tool_call"], abs(a["p_tool_call"] - b["p_tool_call"]),
                 a["top"][0][0] == b["top"][0][0], kl(b["top"], a["top"]), a, b))

print(f"points: {len(rows)} (q41 {len(q41)}, bf16 {len(bf16)})")
print("| point | P q41 | P bf16 | |ΔP| | top-1 same | KL(bf16‖q41) |")
print("|---|--:|--:|--:|:-:|--:|")
for label, pa, pb, d, same, k, a, b in rows:
    print(f"| {label} | {pa:.4f} | {pb:.4f} | {d:.4f} | {'yes' if same else 'NO'} | {k:.4f} |")

print("\n|ΔP| distribution:", " ".join(f">{t}: {sum(r[3] > t for r in rows)}" for t in (0.01, 0.05, 0.1, 0.2, 0.5)))
print("top-1 differs:", sum(not r[4] for r in rows), "/", len(rows))
ks = sorted(r[5] for r in rows)
print("KL median %.4f, max %.4f" % (ks[len(ks) // 2], ks[-1]) if ks else "")

if show_draws:
    for label, pa, pb, d, same, k, a, b in rows:
        if not a["draws"] and not b["draws"]:
            continue
        print(f"\n=== {label}")
        for arm, x in (("q41", a), ("bf16", b)):
            counts = Counter(action(ids) for ids in x["draws"])
            print(f"  {arm} ({len(x['draws'])}):")
            for act, n in counts.most_common():
                print(f"    {n:2d} × {act}")
