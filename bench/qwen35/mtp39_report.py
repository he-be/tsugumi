#!/usr/bin/env python3
"""任意個の腕を 1 枚の表にする (`39-VERIFY-PREFETCH.md`)。

`mtp38_report.py` は base / mtp / mtpqb の 3 腕を名前で決め打ちしていて、
本書は「同じ機構を素の decode と MTP の両方に当てる」ので腕が 4 本以上になる。
比は `--ref` の腕に対して取り、トークン列は `--ref` の 1 回目と突き合わせる。
"""
from __future__ import annotations
import re, sys, json, argparse
from pathlib import Path
from collections import defaultdict
from statistics import median

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("arms")
ap.add_argument("--ref", default="base")
a = ap.parse_args()
root, arms, ref = Path(a.root), a.arms.split(","), a.ref

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
        "verify": num(r"verify=([\d.]+)ms"),
        "draft": num(r"draft=([\d.]+)ms"),
        "commit": num(r"commit=([\d.]+)\)"),
    }

agg = defaultdict(lambda: defaultdict(list))
print(f"{'task':<16} {'rep':<4}" + "".join(f"{x:>12}" for x in arms))
for (task, rep), got in sorted(rows.items()):
    if not all(x in got for x in arms):
        continue
    print(f"{task:<16} {rep:<4}" + "".join(f"{got[x]['tok_s']:>12.3f}" for x in arms))
    for x in arms:
        for k in ("tok_s", "P1", "a", "verify", "commit"):
            if got[x][k] is not None:
                agg[task][(x, k)].append(got[x][k])

print()
head = f"{'task':<16}" + "".join(f"{x:>12}" for x in arms)
head += "".join(f"{'x'+x[:7]:>10}" for x in arms if x != ref)
print(head + "   (median)")
for task, vals in sorted(agg.items()):
    med = {x: median(vals[(x, "tok_s")]) for x in arms}
    line = f"{task:<16}" + "".join(f"{med[x]:>12.3f}" for x in arms)
    line += "".join(f"{med[x]/med[ref]:>10.3f}" for x in arms if x != ref)
    print(line)

print()
print(f"{'task':<16}" + "".join(f"{x+' P1/a/verify/commit':>34}" for x in arms))
for task, vals in sorted(agg.items()):
    line = f"{task:<16}"
    for x in arms:
        def m(k, fmt):
            v = vals.get((x, k))
            return format(median(v), fmt) if v else "-"
        line += f"{m('P1','.1f')+'/'+m('a','.3f')+'/'+m('verify','.1f')+'/'+m('commit','.2f'):>34}"
    print(line)

print()
for task in sorted({t for t, _ in rows}):
    def toks(arm):
        f = root / f"{task}.{arm}.1.json"
        return json.loads(f.read_text())["generated"] if f.exists() else None
    b = toks(ref)
    if b is None:
        continue
    out = f"{task:<16} vs {ref}:"
    for arm in arms:
        if arm == ref:
            continue
        t = toks(arm)
        if t is None:
            continue
        same = sum(1 for x, y in zip(b, t) if x == y)
        out += f"  {arm} {same}/{len(b)}"
    print(out)
