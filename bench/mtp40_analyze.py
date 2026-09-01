#!/usr/bin/env python3
"""N2 の集計 — decode の壁時計を GPU busy と待ちに割る (四則と標本 sd のみ)。

入力は `bench/logs/40_n2/*.err` (`bench/mtp40/*.err` にコピーされた同じもの)。
`bench/mtp40_n2_driver.sh` が回した run 1 本 = ファイル 1 本で、拾うのは 3 種類:

  [decode gpuq busy=..s(..%) idle=..s(..%) span=..s buffers=N tok=32]
  [decode gpu attn=..s(..%) shared=..s(..%) moe=..s(..%) head=..s(..%) total=..s]
  footer 行 (tok/s, new=, decode=, decode hit=/io=, decode/tok io=/cb1=/cb2=/head=)

gpuq の窓は decode 32 歩ごとに 1 対で、末尾の端数 (32 歩に満たない分) は
印字されない。したがって窓の合計は run 全体ではなく「窓に入った歩」だけを覆う。
本スクリプトは窓を 2 通りで合算する:
  all  … 印字された全窓
  w2+  … 先頭の窓 (冷たいキャッシュから始まる。36 §14b-2 と同じ理由) を落とす

推論は 1 回も起動しない。フィットもしない。
"""

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = [ROOT / "bench/mtp40", ROOT / "bench/logs/40_n2"]

GPUQ = re.compile(
    r"\[decode gpuq busy=([\d.]+)s\(\s*\d+%\) idle=([\d.]+)s\(\s*\d+%\) "
    r"span=([\d.]+)s buffers=(\d+) tok=(\d+)\]"
)
GPUSTAGE = re.compile(r"\[decode gpu ((?:\w+=[\d.]+s\(\s*\d+%\) )+)total=([\d.]+)s\]")
STAGE_KV = re.compile(r"(\w+)=([\d.]+)s")

FOOTER = {
    "tok_s": r"tok/s=([\d.]+)",
    "new": r"new=(\d+)tok",
    "decode_s": r"decode=([\d.]+)s",
    "hit": r"decode hit=([\d.]+)%",
    "io_s": r"decode hit=[\d.]+% \d+/\d+ io=([\d.]+)s",
    "io_ms": r"decode/tok io=([\d.]+)ms",
    "cb1_ms": r"cb1=([\d.]+)ms",
    "cb2_ms": r"cb2=([\d.]+)ms",
    "head_ms": r"head=([\d.]+)ms",
}

STAGES = ["attn", "shared", "moe", "head"]


def parse(path: Path) -> dict:
    text = path.read_text(errors="replace")
    row = {"name": path.stem}
    for key, pattern in FOOTER.items():
        m = re.search(pattern, text)
        if m:
            row[key] = float(m.group(1))
    windows = [tuple(float(x) for x in m.groups()) for m in GPUQ.finditer(text)]
    stages = []
    for m in GPUSTAGE.finditer(text):
        d = {k: float(v) for k, v in STAGE_KV.findall(m.group(1))}
        d["total"] = float(m.group(2))
        stages.append(d)
    row["windows"] = windows
    row["stages"] = stages
    if row.get("new") and row.get("decode_s"):
        row["ms_per_tok"] = row["decode_s"] / row["new"] * 1000.0
    for label, drop in (("all", 0), ("w2", 1)):
        w = windows[drop:]
        s = stages[drop:]
        toks = sum(x[4] for x in w)
        if not toks:
            continue
        row[f"busy_{label}"] = sum(x[0] for x in w) / toks * 1000.0
        row[f"idle_{label}"] = sum(x[1] for x in w) / toks * 1000.0
        row[f"span_{label}"] = sum(x[2] for x in w) / toks * 1000.0
        row[f"tok_{label}"] = toks
        if s:
            for stage in STAGES:
                row[f"{stage}_{label}"] = sum(d.get(stage, 0.0) for d in s) / toks * 1000.0
            row[f"gsum_{label}"] = sum(d["total"] for d in s) / toks * 1000.0
    return row


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.mean(values), statistics.stdev(values)


def cell_of(name: str):
    m = re.match(r"(ja_prose|en_bullets)_s(\d+)_r(\d)$", name)
    if not m:
        return None
    return (m.group(1), int(m.group(2)))


def fmt(x, nd=2):
    return "—" if x != x else f"{x:.{nd}f}"


def main() -> int:
    src = next((d for d in DIRS if d.is_dir()), None)
    if src is None:
        raise SystemExit(f"一次資料が無い: {DIRS}")
    rows = [parse(p) for p in sorted(src.glob("*.err"))]
    rows = [r for r in rows if r.get("new")]
    by_name = {r["name"]: r for r in rows}

    print(f"# 一次資料: {src.relative_to(ROOT)}  ({len(rows)} run)\n")

    print("## 1. run ごとの生値 (ms/tok。窓は先頭を落とした w2+)\n")
    hdr = ("run", "tok/s", "new", "wall", "span", "busy", "idle", "io", "attn",
           "shared", "moe", "head", "hit%")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join([
            r["name"], fmt(r.get("tok_s", float("nan")), 3), str(int(r.get("new", 0))),
            fmt(r.get("ms_per_tok", float("nan"))), fmt(r.get("span_w2", float("nan"))),
            fmt(r.get("busy_w2", float("nan"))), fmt(r.get("idle_w2", float("nan"))),
            fmt(r.get("io_ms", float("nan"))), fmt(r.get("attn_w2", float("nan"))),
            fmt(r.get("shared_w2", float("nan"))), fmt(r.get("moe_w2", float("nan"))),
            fmt(r.get("head_w2", float("nan"))), fmt(r.get("hit", float("nan")), 1),
        ]) + " |")

    print("\n## 2. セル平均 (3 反復、± は標本 sd)\n")
    cells = {}
    for r in rows:
        c = cell_of(r["name"])
        if c:
            cells.setdefault(c, []).append(r)
    keys = ["ms_per_tok", "span_w2", "busy_w2", "idle_w2", "io_ms",
            "attn_w2", "shared_w2", "moe_w2", "head_w2", "hit", "tok_s"]
    hdr = ("cell", "n", "wall", "span", "busy", "idle", "io(Σpread)",
           "attn", "shared", "moe", "head", "hit%", "busy/span")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    summary = {}
    for c in sorted(cells, key=lambda c: (-c[1], c[0])):
        rs = cells[c]
        agg = {k: mean_sd([r.get(k) for r in rs]) for k in keys}
        summary[c] = agg
        # 占有率は同じ窓の中で取る (span は窓が覆った歩だけの壁時計)。
        share = agg["busy_w2"][0] / agg["span_w2"][0] * 100
        cellname = f"{c[0]} / {c[1]}"
        print("| " + " | ".join([
            cellname, str(len(rs)),
            f"{fmt(agg['ms_per_tok'][0])}±{fmt(agg['ms_per_tok'][1])}",
            f"{fmt(agg['span_w2'][0])}±{fmt(agg['span_w2'][1])}",
            f"**{fmt(agg['busy_w2'][0])}**±{fmt(agg['busy_w2'][1])}",
            f"**{fmt(agg['idle_w2'][0])}**±{fmt(agg['idle_w2'][1])}",
            f"{fmt(agg['io_ms'][0])}±{fmt(agg['io_ms'][1])}",
            fmt(agg["attn_w2"][0]), fmt(agg["shared_w2"][0]),
            fmt(agg["moe_w2"][0]), fmt(agg["head_w2"][0]),
            fmt(agg["hit"][0], 1), f"**{share:.0f}%**",
        ]) + " |")

    print("\n## 3. 検算\n")
    print("(a) 窓の span/tok 対 footer の wall ms/tok (窓は decode の一部しか覆わない)\n")
    print("| run | wall | span(all) | span(w2+) | 覆った歩/new |")
    print("|---|---|---|---|---|")
    for r in rows:
        if "span_all" not in r:
            continue
        print(f"| {r['name']} | {fmt(r['ms_per_tok'])} | {fmt(r['span_all'])} | "
              f"{fmt(r.get('span_w2', float('nan')))} | "
              f"{int(r['tok_all'])}/{int(r['new'])} |")

    print("\n(b) 段の和 (重なりを二重に数える) 対 マージした busy\n")
    print("| run | Σstage | busy | Σstage/busy |")
    print("|---|---|---|---|")
    for r in rows:
        if "gsum_w2" not in r:
            continue
        print(f"| {r['name']} | {fmt(r['gsum_w2'])} | {fmt(r['busy_w2'])} | "
              f"{r['gsum_w2'] / r['busy_w2']:.3f} |")

    print("\n(c) 計器の摂動 (同一 seed・同一セル、env の有無)\n")
    print("| セル | prof on (r1) | prof off | 差 |")
    print("|---|---|---|---|")
    for reg in ("ja_prose", "en_bullets"):
        on = by_name.get(f"{reg}_s48_r1")
        off = by_name.get(f"{reg}_s48_noprof")
        if not on or not off:
            continue
        d = (off["tok_s"] - on["tok_s"]) / on["tok_s"] * 100
        print(f"| {reg} / 48 | {on['tok_s']:.3f} t/s | {off['tok_s']:.3f} t/s | "
              f"{d:+.2f}% |")

    print("\n(d) 熱ドリフト (先頭と末尾の同一条件)\n")
    a = by_name.get("ja_prose_s48_r1")
    b = by_name.get("ja_prose_s48_drift")
    if a and b:
        d = (b["tok_s"] - a["tok_s"]) / a["tok_s"] * 100
        print(f"ja_prose / 48 / seed 1234: {a['tok_s']:.3f} -> {b['tok_s']:.3f} t/s "
              f"({d:+.2f}%)、hit {a['hit']:.1f} -> {b['hit']:.1f}%")

    print("\n## 4. 37 §1a の残差との橋渡し (**導出**)\n")
    print("残差 = wall − io − cb1 − cb2 − head (37 §1a の 4 項の外)。")
    print("`io` は pread の総和で壁時計ではないので、この残差は分割ではない。\n")
    print("| セル | wall | 残差 | busy | busy − 残差 |")
    print("|---|---|---|---|---|")
    for c in sorted(summary, key=lambda c: (-c[1], c[0])):
        rs = cells[c]
        res = mean_sd([
            r["ms_per_tok"] - r["io_ms"] - r["cb1_ms"] - r["cb2_ms"] - r["head_ms"]
            for r in rs if "io_ms" in r
        ])
        busy = summary[c]["busy_w2"][0]
        wall = summary[c]["ms_per_tok"][0]
        print(f"| {c[0]} / {c[1]} | {fmt(wall)} | {fmt(res[0])}±{fmt(res[1])} | "
              f"{fmt(busy)} | {fmt(busy - res[0])} |")

    print("\n## 5. セル間の差分 — 動くのはどちらか (**実測**)\n")
    print("| 対 | Δspan | Δbusy | Δidle | Δio | Δidle/Δio |")
    print("|---|---|---|---|---|---|")
    pairs = [
        ("en_bullets − ja_prose @48", ("en_bullets", 48), ("ja_prose", 48)),
        ("en_bullets − ja_prose @32", ("en_bullets", 32), ("ja_prose", 32)),
        ("32 − 48 @ja_prose", ("ja_prose", 32), ("ja_prose", 48)),
        ("32 − 48 @en_bullets", ("en_bullets", 32), ("en_bullets", 48)),
    ]
    for label, a, b in pairs:
        if a not in summary or b not in summary:
            continue
        d = {k: summary[a][k][0] - summary[b][k][0]
             for k in ("span_w2", "busy_w2", "idle_w2", "io_ms")}
        ratio = d["idle_w2"] / d["io_ms"] if abs(d["io_ms"]) > 1e-9 else float("nan")
        print(f"| {label} | {d['span_w2']:+.2f} | **{d['busy_w2']:+.2f}** | "
              f"{d['idle_w2']:+.2f} | {d['io_ms']:+.2f} | {ratio:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
