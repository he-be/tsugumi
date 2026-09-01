#!/usr/bin/env python3
"""42 (N3) の集計。四則と標本統計だけ。入力は bench/mtp_goal_ab.py の TSV。

ドライバ自身がスコアと共変量を出すので、ここで足すのは 1 つだけ —
**生成長の食い違いが壁時計比をどれだけ動かしているか**である (§4)。
"""
import statistics, sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/logs/42_n3/goal_v2.tsv")
rows, header = [], None
for line in path.read_text().splitlines():
    if line.startswith("#"):
        continue
    parts = line.split("\t")
    if header is None:
        header = parts
        continue
    rows.append(dict(zip(header, parts)))

pairs = {}
for r in rows:
    key = (r["image"], r["role"])
    pairs.setdefault(key, {})[r["side"]] = r

cells = []
for (image, role), sides in pairs.items():
    off, on = sides["off"], sides["on"]
    cells.append({
        "image": image, "role": role, "order": off["order"],
        "ratio": float(off["wall"]) / float(on["wall"]),
        "len": float(on["new"]) / float(off["new"]),
        "ts": float(on["tok_s"]) / float(off["tok_s"]),
        "io_on": float(on["io_ms_tok"]), "io_off": float(off["io_ms_tok"]),
        "hit_on": float(on["decode_hit"]), "accept": float(on["accept"]),
        "new_off": int(off["new"]), "new_on": int(on["new"]),
    })
cells.sort(key=lambda c: c["ratio"])

print(f"{len(cells)} pairs from {path}\n")
print("image                       role  order   len(on/off)  wall比   t/s比   io_on  hit_on")
for c in cells:
    print(f"{c['image']:27s} {c['role']:5s} {c['order']:7s} "
          f"{c['len']:6.3f} ({c['new_on']:4d}/{c['new_off']:4d}) "
          f"{c['ratio']:6.3f}  {c['ts']:6.3f}  {c['io_on']:6.2f}  {c['hit_on']:5.1f}")

def corr(a, b):
    try:
        return statistics.correlation(a, b)
    except statistics.StatisticsError:
        return float("nan")

R = [c["ratio"] for c in cells]
L = [c["len"] for c in cells]
T = [c["ts"] for c in cells]
IO = [c["io_on"] for c in cells]
H = [c["hit_on"] for c in cells]

print("\n相関 (n=%d ペア)" % len(cells))
print(f"  corr(生成長比 on/off, 壁時計比) = {corr(L, R):+.3f}")
print(f"  corr(io ms/tok,       壁時計比) = {corr(IO, R):+.3f}")
print(f"  corr(hit%%,            壁時計比) = {corr(H, R):+.3f}")
print(f"  corr(io ms/tok,       t/s 比)   = {corr(IO, T):+.3f}")
print(f"  corr(hit%%,            t/s 比)   = {corr(H, T):+.3f}")
print(f"  corr(生成長比,        t/s 比)   = {corr(L, T):+.3f}")

def report(name, sel):
    if not sel:
        print(f"  {name}: 該当なし")
        return
    r = [c["ratio"] for c in sel]
    t = [c["ts"] for c in sel]
    print(f"  {name} (n={len(sel)}): 壁時計比 中央値 {statistics.median(r):.3f}"
          + (f" sd {statistics.stdev(r):.3f}" if len(r) > 1 else "")
          + f" / t/s 比 中央値 {statistics.median(t):.3f}"
          + (f" sd {statistics.stdev(t):.3f}" if len(t) > 1 else ""))

print("\n生成長で切ると")
report("長さが完全に一致", [c for c in cells if c["new_on"] == c["new_off"]])
report("|長さ比-1| <= 5%", [c for c in cells if abs(c["len"] - 1) <= 0.05])
report("|長さ比-1| >  5%", [c for c in cells if abs(c["len"] - 1) > 0.05])
report("全ペア", cells)

ref = [c for c in cells if c["role"] in ("head", "mid", "tail")]
if len(ref) == 3:
    same = [c for c in ref if c["order"] == "off/on"]
    other = [c for c in ref if c["order"] == "on/off"]
    print("\n目盛り run (同一画像 3 本)")
    for c in sorted(ref, key=lambda x: x["role"]):
        print(f"  {c['role']:5s} order={c['order']:7s} 壁時計比 {c['ratio']:.3f}"
              f"  t/s 比 {c['ts']:.3f}")
    if same and other:
        print(f"  同順序 2 本の幅: {abs(same[0]['ratio'] - same[1]['ratio']) / statistics.mean([s['ratio'] for s in same]) * 100:.2f}%"
              f"   順序を変えた 1 本との差: "
              f"{(other[0]['ratio'] - statistics.mean([s['ratio'] for s in same])) / statistics.mean([s['ratio'] for s in same]) * 100:+.2f}%")
