#!/usr/bin/env python3
"""N2 の後半の集計 — 投機オン (verify ブロック) の台帳 (四則と標本 sd のみ)。

入力は `bench/mtp40b/*.err` (`bench/logs/40_n2b/*.err` の写し)。
`bench/mtp40_n2b_driver.sh` が `TF_PREFILL_HOST_PROFILE=1` と
`TF_PREFILL_GPU_PROFILE=1` を両方立てて回したので、チャンク 1 個につき
3 行が順に出る:

  [prefill gpu  attn=..s(..%) shared=.. moe=.. tail=.. head=.. total=..s]
  [prefill host enc.front=.. wait.front=.. ... call=..s | tiles=.. ahead=.. experts=.. miss=.. io=..s ..GB/s]
  [prefill gpuq busy=..s(..%) idle=..s(..%) span=..s buffers=..]

プロンプトの prefill チャンクと verify ブロックの分離は **`experts > 0`** で行う
(36 §14a の 5 つの手がかりのうち、§16 が「他の bs / スロット数に持ち出すときは
これを使う」と訂正したもの)。分離の妥当性は §3 の検算で footer と突き合わせる。

推論は 1 回も起動しない。フィットもしない。
"""

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = [[ROOT / "bench/mtp40b", ROOT / "bench/mtp40d"],
        [ROOT / "bench/logs/40_n2b", ROOT / "bench/logs/40_n2d"]]

GPU = re.compile(r"\[prefill gpu ((?:[\w.]+=[\d.]+s\(\s*\d+%\) )+)total=([\d.]+)s\]")
HOST = re.compile(r"\[prefill host (.*?) call=([\d.]+)s \| tiles=(\d+) ahead=(\d+) "
                  r"experts=(\d+) miss=(\d+) io=([\d.]+)s")
GPUQ = re.compile(r"\[prefill gpuq busy=([\d.]+)s\(\s*\d+%\) idle=([\d.]+)s\(\s*\d+%\) "
                  r"span=([\d.]+)s buffers=(\d+)\]")
KV = re.compile(r"([\w.]+)=([\d.]+)s")

FOOTER = {
    "tok_s": r"tok/s=([\d.]+)",
    "new": r"new=(\d+)tok",
    "decode_s": r"decode=([\d.]+)s",
    "hit": r"decode hit=([\d.]+)%",
    "io_s": r"decode hit=[\d.]+% \d+/\d+ io=([\d.]+)s",
    "rounds": r"rounds=(\d+)",
    "accept": r"accept=([\d.]+)/",
    "tok_per_round": r"tok/round=([\d.]+)",
    "draft_s": r"draft=([\d.]+)s",
    "verify_s": r"verify=([\d.]+)s",
}

STAGES = ["attn", "shared", "moe", "tail", "head"]


def parse(path: Path) -> dict:
    text = path.read_text(errors="replace")
    row = {"name": path.stem}
    for key, pattern in FOOTER.items():
        m = re.search(pattern, text)
        if m:
            row[key] = float(m.group(1))
    chunks, pending = [], None
    for line in text.splitlines():
        m = GPU.search(line)
        if m:
            # GPU 側の段名は host 側と `tail` / `head` で衝突するので接頭辞を付ける
            # (付けないと host の値が GPU の値を上書きする)。
            pending = {"g_" + k: float(v) for k, v in KV.findall(m.group(1))}
            pending["gpu_total"] = float(m.group(2))
            continue
        m = HOST.search(line)
        if m:
            c = pending or {}
            pending = None
            c.update({k: float(v) for k, v in KV.findall(m.group(1))})
            c["call"] = float(m.group(2))
            c["tiles"] = int(m.group(3))
            c["ahead"] = int(m.group(4))
            c["experts"] = int(m.group(5))
            c["miss"] = int(m.group(6))
            c["io"] = float(m.group(7))
            chunks.append(c)
            continue
        m = GPUQ.search(line)
        if m and chunks:
            chunks[-1]["busy"] = float(m.group(1))
            chunks[-1]["idle"] = float(m.group(2))
            chunks[-1]["span"] = float(m.group(3))
            chunks[-1]["buffers"] = int(m.group(4))
    row["prompt"] = [c for c in chunks if c["experts"] == 0]
    row["blocks"] = [c for c in chunks if c["experts"] > 0]
    b = row["blocks"]
    new = row.get("new", 0)
    if b and new:
        row["nblocks"] = len(b)
        for key in ("call", "busy", "idle", "io", "fetch", "wait.front", "gpu_total"):
            row["sum_" + key] = sum(c.get(key, 0.0) for c in b)
            row["pt_" + key] = row["sum_" + key] / new * 1000.0
        for st in STAGES:
            row["pt_" + st] = sum(c.get("g_" + st, 0.0) for c in b) / new * 1000.0
            row["pb_" + st] = sum(c.get("g_" + st, 0.0) for c in b) / len(b) * 1000.0
        for key in ("call", "busy", "idle"):
            row["pb_" + key] = row["sum_" + key] / len(b) * 1000.0
        row["sum_experts"] = sum(c["experts"] for c in b)
        row["sum_miss"] = sum(c["miss"] for c in b)
        row["ahead_eq_tiles"] = sum(1 for c in b if c["ahead"] == c["tiles"])
        row["ms_per_tok"] = row["decode_s"] / new * 1000.0
    return row


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.mean(values), statistics.stdev(values)


def cell_of(name: str):
    m = re.match(r"(ja_prose|en_bullets)_bs(\d)_r(\d)$", name)
    return (m.group(1), int(m.group(2))) if m else None


def fmt(x, nd=2):
    return "—" if x != x else f"{x:.{nd}f}"


def pct(values, q):
    s = sorted(values)
    if not s:
        return float("nan")
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main() -> int:
    src = next((g for g in DIRS if g[0].is_dir()), None)
    if src is None:
        raise SystemExit(f"一次資料が無い: {DIRS}")
    files = sorted(p for d in src if d.is_dir() for p in d.glob("*.err"))
    rows = [parse(p) for p in files]
    rows = [r for r in rows if r.get("nblocks")]
    by_name = {r["name"]: r for r in rows}
    print("# 一次資料: " + ", ".join(str(d.relative_to(ROOT)) for d in src if d.is_dir())
          + f"  ({len(rows)} run)\n")

    print("## 1. run ごと (産出トークン 1 本あたり ms。ブロックだけを合算した値)\n")
    hdr = ("run", "tok/s", "new", "blocks", "wall", "Σcall", "busy", "idle", "io",
           "attn", "shared", "moe", "tail", "head", "accept", "hit%")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join([
            r["name"], fmt(r["tok_s"], 3), str(int(r["new"])), str(r["nblocks"]),
            fmt(r["ms_per_tok"]), fmt(r["pt_call"]), fmt(r["pt_busy"]),
            fmt(r["pt_idle"]), fmt(r["pt_io"]), fmt(r["pt_attn"]),
            fmt(r["pt_shared"]), fmt(r["pt_moe"]), fmt(r["pt_tail"]),
            fmt(r["pt_head"]), fmt(r.get("accept", float("nan")), 3),
            fmt(r.get("hit", float("nan")), 1),
        ]) + " |")

    print("\n## 2. セル平均 (3 反復、± は標本 sd。ms/産出トークン)\n")
    cells = {}
    for r in rows:
        c = cell_of(r["name"])
        if c:
            cells.setdefault(c, []).append(r)
    hdr = ("cell", "n", "wall", "Σcall", "busy", "idle", "io", "attn", "shared",
           "moe", "tail", "head", "busy/wall")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for c in sorted(cells, key=lambda c: (c[1], c[0])):
        rs = cells[c]
        g = lambda k: mean_sd([r[k] for r in rs])
        wall = g("ms_per_tok")
        busy = g("pt_busy")
        print("| " + " | ".join([
            f"{c[0]} / bs={c[1]}", str(len(rs)),
            f"{fmt(wall[0])}±{fmt(wall[1])}", f"{fmt(g('pt_call')[0])}",
            f"**{fmt(busy[0])}**±{fmt(busy[1])}",
            f"**{fmt(g('pt_idle')[0])}**±{fmt(g('pt_idle')[1])}",
            f"{fmt(g('pt_io')[0])}±{fmt(g('pt_io')[1])}",
            fmt(g("pt_attn")[0]), fmt(g("pt_shared")[0]), fmt(g("pt_moe")[0]),
            fmt(g("pt_tail")[0]), fmt(g("pt_head")[0]),
            f"**{busy[0] / wall[0] * 100:.0f}%**",
        ]) + " |")

    print("\n## 3. 検算 (36 §14a と同じ 4 本)\n")
    print("| run | ブロック数 対 rounds | Σcall 対 verify= | Σio 対 decode io= | ahead==tiles |")
    print("|---|---|---|---|---|")
    for r in rows:
        rounds = int(r.get("rounds", 0))
        dv = (r["sum_call"] - r["verify_s"]) / r["verify_s"] * 100 if r.get("verify_s") else float("nan")
        di = (r["sum_io"] - r["io_s"]) / r["io_s"] * 100 if r.get("io_s") else float("nan")
        print(f"| {r['name']} | {r['nblocks']} 対 {rounds} | "
              f"{r['sum_call']:.3f} 対 {r.get('verify_s', 0):.3f} ({dv:+.2f}%) | "
              f"{r['sum_io']:.3f} 対 {r.get('io_s', 0):.3f} ({di:+.2f}%) | "
              f"{r['ahead_eq_tiles']}/{r['nblocks']} |")

    print("\n## 4. ブロック 1 個の分布 (p10 / 中央値 / p90、ms)\n")
    print("| cell | call | busy | idle | fetch | io | wait.front |")
    print("|---|---|---|---|---|---|---|")
    for c in sorted(cells, key=lambda c: (c[1], c[0])):
        bl = [b for r in cells[c] for b in r["blocks"]]
        cols = []
        for key in ("call", "busy", "idle", "fetch", "io", "wait.front"):
            v = [b.get(key, 0.0) * 1000 for b in bl]
            cols.append(f"{pct(v, .1):.1f} / **{pct(v, .5):.1f}** / {pct(v, .9):.1f}")
        print(f"| {c[0]} / bs={c[1]} (N={len(bl)}) | " + " | ".join(cols) + " |")

    print("\n## 6. ブロック 1 個あたりの段別費用と、隣接ペアで解く (F, c) (**実測** + **導出**)\n")
    print("`step(bs) = F + c·bs` を段ごとに隣接ペアの傾きで解く (36 §9g の規則)。")
    print("F = ブロック 1 回の固定費、c = 検証位置 1 つの限界費用 (どちらも GPU 秒)。")
    print("2 点なら過不足なく決まるだけだが、3 点あれば傾きが揃うかを見られる。\n")
    widths = sorted({c[1] for c in cells})
    hdr = ["cell", "量"] + [f"bs={w}" for w in widths] \
        + [f"c({widths[i]}→{widths[i+1]})" for i in range(len(widths) - 1)] + ["F"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for reg in ("ja_prose", "en_bullets"):
        for key, label in (("pb_busy", "**busy (合計)**"), ("pb_attn", "attn"),
                           ("pb_shared", "shared"), ("pb_moe", "**moe**"),
                           ("pb_tail", "tail"), ("pb_head", "head"),
                           ("pb_call", "call (壁時計)")):
            vals = []
            for w in widths:
                key_cell = (reg, w)
                vals.append(statistics.mean([r[key] for r in cells[key_cell]])
                            if key_cell in cells else float("nan"))
            slopes = [(vals[i + 1] - vals[i]) / (widths[i + 1] - widths[i])
                      for i in range(len(widths) - 1)]
            ff = vals[0] - slopes[0] * widths[0]
            print("| " + " | ".join(
                [reg, label] + [fmt(v) for v in vals]
                + [f"{s_:+.2f}" for s_ in slopes] + [f"{ff:+.2f}"]) + " |")

    print("\n## 5. 熱ドリフト\n")
    a, b = by_name.get("ja_prose_bs2_r1"), by_name.get("ja_prose_bs2_drift")
    if a and b:
        print(f"ja_prose / bs=2 / seed 1234: {a['tok_s']:.3f} -> {b['tok_s']:.3f} t/s "
              f"({(b['tok_s'] - a['tok_s']) / a['tok_s'] * 100:+.2f}%)、"
              f"busy/tok {a['pt_busy']:.2f} -> {b['pt_busy']:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
