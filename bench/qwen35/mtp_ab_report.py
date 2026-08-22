#!/usr/bin/env python3
"""`mtp_ab.sh` が書いたフッターを 1 枚の表にする。"""
from __future__ import annotations
import re, sys, json
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen35/mtp-ab")
rows = defaultdict(dict)
for f in sorted(root.glob("*.footer")):
    task, arm, rep = f.stem.rsplit(".", 2)
    text = f.read_text()
    def num(pat, cast=float):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None
    rows[(task, rep)][arm] = {
        "tok_s": num(r"tok/s=([\d.]+)"),
        "decode_s": num(r"decode=([\d.]+)s"),
        "prefill_s": num(r"prefill=([\d.]+)s"),
        "ttft": num(r"ttft=([\d.]+)s"),
        "new": num(r"new=(\d+)tok", int),
        "P1": num(r"mtpP1=([\d.]+)%"),
        "a": num(r"a=([\d.]+)"),
        "passes": num(r"passes=(\d+)", int),
        "gpu": num(r"\[decode/tok gpu=([\d.]+)ms"),
        "verify": num(r"verify=([\d.]+)ms"),
        "draft": num(r"draft=([\d.]+)ms"),
    }

print(f"{'task':<16} {'rep':<4} {'base tok/s':>10} {'mtp tok/s':>10} {'x':>6} "
      f"{'P1':>7} {'a':>6} {'verify':>8} {'draft':>7}")
best = defaultdict(list)
for (task, rep), arms in sorted(rows.items()):
    b, m = arms.get("base"), arms.get("mtp")
    if not b or not m: continue
    ratio = m["tok_s"] / b["tok_s"]
    best[task].append((b["tok_s"], m["tok_s"], m["P1"], m["a"]))
    print(f"{task:<16} {rep:<4} {b['tok_s']:>10.3f} {m['tok_s']:>10.3f} {ratio:>6.3f} "
          f"{m['P1'] or 0:>6.1f}% {m['a'] or 0:>6.3f} {m['verify'] or 0:>7.1f}ms "
          f"{m['draft'] or 0:>6.1f}ms")

print()
print(f"{'task':<16} {'base':>8} {'mtp':>8} {'x':>7} {'P1':>7} {'a':>6}  (median of reps)")
for task, vals in sorted(best.items()):
    vals.sort()
    mid = len(vals) // 2
    b = sorted(v[0] for v in vals)[mid]
    m = sorted(v[1] for v in vals)[mid]
    p1 = sum(v[2] for v in vals) / len(vals)
    a = sum(v[3] for v in vals) / len(vals)
    print(f"{task:<16} {b:>8.3f} {m:>8.3f} {m/b:>7.3f} {p1:>6.1f}% {a:>6.3f}")

# token identity: mtp vs base
print()
for f in sorted(root.glob("*.base.1.json")):
    task = f.stem.rsplit(".", 2)[0]
    other = root / f"{task}.mtp.1.json"
    if not other.exists(): continue
    a = json.loads(f.read_text())["generated"]
    b = json.loads(other.read_text())["generated"]
    same = sum(1 for x, y in zip(a, b) if x == y)
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print(f"{task:<16} token match {same}/{min(len(a), len(b))}  first diff {first}")
