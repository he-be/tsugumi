#!/usr/bin/env python3
"""N2 の付録の集計 — verify ブロックの `attn` バケットを 9 つに割る。

入力は `bench/mtp40c/*.err` (`bench/logs/40_n2c/*.err` の写し)。
`TF_PREFILL_GPU_PROFILE=2` の run なので、チャンクごとに
`[prefill gpu ...]` の直後に `[prefill gpu attn embed=.. norm=.. qkv=.. rope=..
kvcopy=.. attn.swa=.. attn.full=.. oproj=.. post=.. total=..]` が出る。

**この run の壁時計は使えない** — 細分はグループごとにコマンドバッファを切るので
GPU がその間遊ぶ (PrefillGPUProfile.swift の doc comment)。取れるのは
**バケットの中の比率**だけである。

**分解能の限界。**ログは 3 桁 (1 ms) で、ブロック 1 個の細目は 0.000〜0.005s
なので**1 個ずつは読めない**。合算 (100 個以上) してはじめて意味を持つ。
比は Σ秒 / Σtotal で取る。

推論は 1 回も起動しない。フィットもしない。
"""

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = [ROOT / "bench/mtp40c", ROOT / "bench/logs/40_n2c"]

DETAIL = re.compile(r"\[prefill gpu attn ((?:[\w.]+=[\d.]+s\(\s*\d+%\) )+)total=([\d.]+)s\]")
HOST = re.compile(r"\[prefill host .*? call=([\d.]+)s \| tiles=(\d+) ahead=(\d+) "
                  r"experts=(\d+) miss=(\d+) io=([\d.]+)s")
KV = re.compile(r"([\w.]+)=([\d.]+)s")
NEW = re.compile(r"new=(\d+)tok")
ROUNDS = re.compile(r"rounds=(\d+)")

DETAILS = ["embed", "norm", "qkv", "rope", "kvcopy", "attn.swa", "attn.full",
           "oproj", "post"]


def parse(path: Path) -> dict:
    text = path.read_text(errors="replace")
    row = {"name": path.stem}
    m = NEW.search(text)
    row["new"] = int(m.group(1)) if m else 0
    m = ROUNDS.search(text)
    row["rounds"] = int(m.group(1)) if m else 0
    chunks, pending = [], None
    for line in text.splitlines():
        m = DETAIL.search(line)
        if m:
            pending = {k: float(v) for k, v in KV.findall(m.group(1))}
            pending["total"] = float(m.group(2))
            continue
        m = HOST.search(line)
        if m:
            if pending is not None:
                pending["experts"] = int(m.group(4))
                chunks.append(pending)
            pending = None
    row["blocks"] = [c for c in chunks if c["experts"] > 0]
    return row


def fmt(x, nd=2):
    return "—" if x != x else f"{x:.{nd}f}"


def main() -> int:
    src = next((d for d in DIRS if d.is_dir()), None)
    if src is None:
        raise SystemExit(f"一次資料が無い: {DIRS}")
    rows = [parse(p) for p in sorted(src.glob("*.err"))]
    rows = [r for r in rows if r["blocks"]]
    print(f"# 一次資料: {src.relative_to(ROOT)}  ({len(rows)} run、帰属専用)\n")

    cells = {}
    for r in rows:
        m = re.match(r"(ja_prose|en_bullets)_r(\d)$", r["name"])
        if m:
            cells.setdefault(m.group(1), []).append(r)

    print("## 1. ブロックの `attn` バケットの中身 (Σ秒の比率、%)\n")
    print("| cell | run | blocks | " + " | ".join(DETAILS) + " | Σtotal(ms/ブロック) |")
    print("|---|---|---|" + "---|" * (len(DETAILS) + 1))
    for reg in sorted(cells):
        for r in cells[reg]:
            b = r["blocks"]
            tot = sum(c["total"] for c in b)
            shares = [sum(c.get(d, 0.0) for c in b) / tot * 100 for d in DETAILS]
            print(f"| {reg} | {r['name'][-2:]} | {len(b)} | "
                  + " | ".join(f"{s:.1f}" for s in shares)
                  + f" | {tot / len(b) * 1000:.2f} |")

    print("\n## 2. セル平均 (3 反復、± は標本 sd)\n")
    print("| cell | n | " + " | ".join(DETAILS) + " |")
    print("|---|---|" + "---|" * len(DETAILS))
    for reg in sorted(cells):
        rs = cells[reg]
        cols = []
        for d in DETAILS:
            vals = []
            for r in rs:
                tot = sum(c["total"] for c in r["blocks"])
                vals.append(sum(c.get(d, 0.0) for c in r["blocks"]) / tot * 100)
            sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            cols.append(f"{statistics.mean(vals):.1f}±{fmt(sd, 1)}")
        print(f"| {reg} | {len(rs)} | " + " | ".join(cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
