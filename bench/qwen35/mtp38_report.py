#!/usr/bin/env python3
"""`mtp_ab.sh` の 3 腕 (base / mtp / mtpqb) を 1 枚の表にする。

`mtp_ab_report.py` は base と mtp の 2 腕を前提にしていて、
`38-MTP-VERIFY-PATH.md` は「検証パスの attention を差し替えた腕」を
同じ熱環境の中で並べたいので、腕を任意個取れる形にした。
"""
from __future__ import annotations
import re, sys, json
from pathlib import Path
from collections import defaultdict
from statistics import median

root = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen35/mtp38-ab")
arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base", "mtpqb", "mtp"]
rows = defaultdict(dict)
for f in sorted(root.glob("*.footer")):
    task, arm, rep = f.stem.rsplit(".", 2)
    text = f.read_text()
    def num(pat, cast=float):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None
    rows[(task, rep)][arm] = {
        "tok_s": num(r"tok/s=([\d.]+)"),
        "P1": num(r"mtpP1=([\d.]+)%"),
        "a": num(r"a=([\d.]+)"),
        "gpu": num(r"\[decode/tok gpu=([\d.]+)ms"),
        "verify": num(r"verify=([\d.]+)ms"),
        "draft": num(r"draft=([\d.]+)ms"),
    }

head = f"{'task':<16} {'rep':<4}" + "".join(f"{a:>11}" for a in arms)
head += f"{'x(mtp)':>8}{'x(qb)':>8}{'P1':>7}{'a':>7}{'verify':>9}"
print(head)
agg = defaultdict(lambda: defaultdict(list))
for (task, rep), got in sorted(rows.items()):
    if not all(a in got for a in arms):
        continue
    line = f"{task:<16} {rep:<4}" + "".join(f"{got[a]['tok_s']:>11.3f}" for a in arms)
    base = got["base"]["tok_s"]
    line += f"{got['mtp']['tok_s']/base:>8.3f}"
    line += f"{got['mtpqb']['tok_s']/base:>8.3f}" if "mtpqb" in got else f"{'-':>8}"
    line += f"{got['mtp']['P1'] or 0:>6.1f}%{got['mtp']['a'] or 0:>7.3f}"
    line += f"{got['mtp']['verify'] or 0:>8.1f}ms"
    print(line)
    for a in arms:
        agg[task][a].append(got[a]["tok_s"])
    agg[task]["P1"].append(got["mtp"]["P1"] or 0)
    agg[task]["a"].append(got["mtp"]["a"] or 0)

print()
print(f"{'task':<16}" + "".join(f"{a:>11}" for a in arms)
      + f"{'x(mtp)':>8}{'x(qb)':>8}{'P1':>7}{'a':>7}   (median)")
for task, vals in sorted(agg.items()):
    med = {a: median(vals[a]) for a in arms}
    line = f"{task:<16}" + "".join(f"{med[a]:>11.3f}" for a in arms)
    line += f"{med['mtp']/med['base']:>8.3f}"
    line += f"{med['mtpqb']/med['base']:>8.3f}" if "mtpqb" in med else f"{'-':>8}"
    line += f"{median(vals['P1']):>6.1f}%{median(vals['a']):>7.3f}"
    print(line)

# What the arms emitted, against plain decode and against each other.
print()
for task in sorted({t for t, _ in rows}):
    def toks(arm):
        f = root / f"{task}.{arm}.1.json"
        return json.loads(f.read_text())["generated"] if f.exists() else None
    b = toks("base")
    if b is None:
        continue
    out = f"{task:<16}"
    for arm in arms:
        if arm == "base":
            continue
        t = toks(arm)
        if t is None:
            continue
        same = sum(1 for x, y in zip(b, t) if x == y)
        out += f" {arm} vs base {same}/{len(b)}"
    print(out)
