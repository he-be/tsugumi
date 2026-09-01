#!/usr/bin/env python3
"""51 (P-6) — D の試作 (`mmap` + `MTLResidencySet`) を今の `pread` と ABBA で並べる。

49 §10 の P-6:「30 層 × 8 エキスパート、マッピングは張りっぱなし、先読みは
1 層先行、residency set は層ごと。**比ではなく 1 歩の壁時計**を今の
`PreadExpertStreamer` と並べる」。50 §7 のとおり **D の根拠は速度ではなく占有**
なので、速度に求めるのは「負けない」ことだけである。

**2026-08-20 に既定が mmap + advise になった** (52 §8)。このドライバは
`TF_EXPERT_MMAP` を毎 run 明示的に渡すので腕は動かないが、`F_RDADVISE` は
既定で「mmap の腕だけ出す」ようになった — **51 と 52 §1 のログは advise が
どちらの腕にも無かったときのもの**なので、あれを再現するときは `--advise off`
を渡すこと。既定 (`--advise auto`) は**今の製品の形**を測る。

腕は 2 つ、違いは `TF_EXPERT_MMAP` だけ:
  pread  今の production (私有スロット 32 個に `pread` でコピー)
  mmap   層ファイルを `MAP_SHARED` で張り、エキスパート単位の `bytesNoCopy`
         バッファを GPU に直接読ませる。層ごとの `MTLResidencySet` を
         fetch の側で `commit()` + `requestResidency()` する

temp 0 なのでミスは決定論 (46 §5a)。**両腕でミス数と生成文が一致することを
毎 run 検定する** — 一致していればバイトは固定で、動くのは費用だけになる。

使い方:
    ./bench/mtp51_mmap_ab.py --rounds 3                    # 運用点 (math/256/32)
    ./bench/mtp51_mmap_ab.py --rounds 3 --messages bench/story.json
"""
import argparse
import hashlib
import os
import pathlib
import re
import statistics as st
import subprocess
import sys
import time

CLI = "./.build/release/TurboFieldfareCLI"
MODEL = os.environ.get("TF_MODEL", "scratch/gemma4-qat-sym.gturbo")
MESSAGES = os.environ.get("TF_MESSAGES", "bench/math.json")
COOL_S = int(os.environ.get("COOL_S", "10"))
EXPERT_MB = 3.35872          # expertStride 3,358,720 B (48 §1)

RE_EXPERT = re.compile(r"decode hit=([\d.]+)%\s+(\d+)/(\d+)\s+io=([\d.]+)s")
RE_PREFILL_IO = re.compile(r"expert prefill hit=[\d.]+%\s+\d+/(\d+)\s+io=([\d.]+)s")
RE_PERTOK = re.compile(r"decode/tok io=([\d.]+)ms")
RE_RUN = re.compile(r"new=(\d+)tok decode=([\d.]+)s tok/s=([\d.]+)")
RE_LOAD = re.compile(r"prefill=([\d.]+)s ttft=([\d.]+)s peak=([\d.]+)GB rss=([\d.]+)GB")
# [expert mmap layers=30 mapped=12.90GB sync=712(+2730/-1770) noop=476
#  residency=2.570s /tok=80.33ms]
RE_MMAP = re.compile(
    r"expert mmap layers=(\d+) mapped=([\d.]+)GB sync=(\d+)\(\+(\d+)/-(\d+)\) "
    r"noop=(\d+) residency=([\d.]+)s /tok=([\d.]+)ms")


def run_once(mmap_on, args):
    env = dict(os.environ)
    env["TF_EXPERT_MMAP"] = "1" if mmap_on else "0"
    # auto = 製品の既定 (mmap の腕だけ advise)。on/off は両腕に固定する。
    if args.advise != "auto":
        env["TF_EXPERT_MMAP_ADVISE"] = "1" if args.advise == "on" else "0"
    else:
        env.pop("TF_EXPERT_MMAP_ADVISE", None)
    cmd = [CLI, "--model", args.model, "--messages-file", args.messages,
           "--temperature", "0", "--max-new", str(args.max_new),
           "--expert-cache-slots", str(args.slots),
           "--verification", "trusted-install"]
    if args.draft_block_size:
        cmd += ["--draft-block-size", str(args.draft_block_size)]
    started = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    blob = proc.stdout + proc.stderr
    if proc.returncode != 0:
        sys.stderr.write(blob[-2000:])
        raise SystemExit(f"CLI failed ({proc.returncode})")
    expert, pertok, run = (RE_EXPERT.search(blob), RE_PERTOK.search(blob),
                           RE_RUN.search(blob))
    load, prefill_io = RE_LOAD.search(blob), RE_PREFILL_IO.search(blob)
    if not (expert and pertok and run and load and prefill_io):
        sys.stderr.write(blob[-2000:])
        raise SystemExit("footer をパースできない")
    hits, requests = int(expert.group(2)), int(expert.group(3))
    misses = requests - hits
    steps = int(run.group(1)) - 1          # decode 歩数 (1 歩目は prefill 由来)
    io_ms = float(pertok.group(1))
    # MTP (bs>0) はブロック経路なので `[decode/tok]` の台帳が動かない。
    # そのときは expert telemetry の decode io を歩数で割る。
    if io_ms == 0.0:
        io_ms = float(expert.group(4)) * 1e3 / max(steps, 1)
    mb = misses * EXPERT_MB / max(steps, 1)
    mm = RE_MMAP.search(blob)
    # 生成文だけを取る (footer は腕で違う)。temp 0 なので両腕で同一のはず。
    text = blob.split("[stop=")[0]
    row = {"mmap": mmap_on, "misses": misses, "requests": requests,
           "digest": hashlib.sha256(text.encode()).hexdigest()[:12],
           "steps": steps, "io_ms": io_ms, "mb_tok": mb,
           "gbs": mb / 1e3 / (io_ms / 1e3) if io_ms else 0.0,
           "tok_s": float(run.group(3)),
           "decode_s": float(run.group(2)),
           "prefill_s": float(load.group(1)),
           "ttft_s": float(load.group(2)),
           "peak_gb": float(load.group(3)),
           "rss_gb": float(load.group(4)),
           "prefill_experts": int(prefill_io.group(1)),
           "prefill_io_s": float(prefill_io.group(2)),
           "wall": time.time() - started, "text": blob}
    row["residency_ms_tok"] = float(mm.group(8)) if mm else 0.0
    row["syncs"] = int(mm.group(3)) if mm else 0
    return row


def median(values):
    return st.median(values) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--messages", default=MESSAGES)
    # MTP を入れた条件。**別条件なので他の表と数字を並べないこと** (40 §4-11)。
    ap.add_argument("--draft-block-size", type=int, default=0)
    ap.add_argument("--advise", choices=("auto", "on", "off"), default="auto",
                    help="F_RDADVISE: auto = 製品の既定 (mmap の腕だけ)、"
                         "on/off は両腕に固定。51 と 52 §1 のログは off で取られている")
    ap.add_argument("--out", default="bench/mtp51/mmap_ab.log")
    args = ap.parse_args()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    log = open(args.out, "w")
    log.write(f"# {' '.join(sys.argv)}\n# model={args.model} "
              f"messages={args.messages} slots={args.slots} "
              f"max_new={args.max_new} bs={args.draft_block_size} "
              f"cool={COOL_S}s\n")

    results = {False: [], True: []}
    # ABBA: pread mmap mmap pread。位置効果を腕の両側から食わせる (48 §5a)。
    for round_index in range(args.rounds):
        for mmap_on in (False, True, True, False):
            result = run_once(mmap_on, args)
            results[mmap_on].append(result)
            arm = "mmap " if mmap_on else "pread"
            line = (f"round {round_index + 1}  {arm}  "
                    f"io {result['io_ms']:7.2f} ms/tok  "
                    f"tok/s {result['tok_s']:7.3f}  "
                    f"prefill {result['prefill_s']:6.3f}s  "
                    f"peak {result['peak_gb']:5.2f}GB  "
                    f"miss {result['misses']}/{result['requests']}  "
                    f"out {result['digest']}")
            print(line, flush=True)
            log.write(line + "\n")
            log.write(result["text"] + "\n")
            log.flush()
            time.sleep(COOL_S)

    pread, mm = results[False], results[True]
    print()
    print("== バイトの検定 (temp 0 なら両腕で同一のはず、46 §5a)")
    miss_p = {r["misses"] for r in pread}
    miss_m = {r["misses"] for r in mm}
    print(f"   pread ミス {sorted(miss_p)}")
    print(f"   mmap  ミス {sorted(miss_m)}")
    print("   " + ("ミス数は固定 ⇒ 動いたのは費用だけ" if len(miss_p | miss_m) == 1
                   else "*** ミス数が動いた。この比較は成り立たない ***"))
    digests = {r["digest"] for r in pread} | {r["digest"] for r in mm}
    print(f"   生成文の sha256 (先頭 12): {sorted(digests)}")
    print("   " + ("両腕で同一 ⇒ 48 §9 の受け入れ (バイト一致) を満たす"
                   if len(digests) == 1
                   else "*** 出力が違う。D の受け入れ条件を満たしていない ***"))

    print()
    cols = (("io ms/tok", "io_ms", "{:>12.2f}"),
            ("GB/s", "gbs", "{:>8.2f}"),
            ("tok/s", "tok_s", "{:>9.3f}"),
            ("prefill s", "prefill_s", "{:>11.3f}"),
            ("ttft s", "ttft_s", "{:>9.3f}"),
            ("peak GB", "peak_gb", "{:>9.2f}"))
    header = f"{'腕':<8}{'n':>3}" + "".join(
        f"{name:>{int(fmt.split(':>')[1].split('.')[0])}}" for name, _, fmt in cols)
    print(header)
    for label, group in (("pread", pread), ("mmap", mm)):
        row = f"{label:<8}{len(group):>3}"
        for _, key, fmt in cols:
            row += fmt.format(median([r[key] for r in group]))
        print(row)

    def delta(key, unit="", better_low=True):
        a, b = median([r[key] for r in pread]), median([r[key] for r in mm])
        sign = "-" if b < a else "+"
        ratio = b / a if a else float("nan")
        verdict = ""
        if (b < a) == better_low:
            verdict = "mmap の勝ち" if b != a else ""
        else:
            verdict = "mmap の負け"
        return (f"   {key:<12} {a:8.3f}{unit} -> {b:8.3f}{unit}  "
                f"({sign}{abs(b - a):.3f}, x{ratio:.3f})  {verdict}")

    print()
    print("== 1 歩の壁時計 (P-6 が問うているもの)")
    print(delta("io_ms", " ms/tok"))
    print(delta("tok_s", " tok/s", better_low=False))
    print()
    print("== 占有 (50 §7 で D に残っている根拠)")
    print(delta("peak_gb", " GB"))
    print(delta("rss_gb", " GB"))
    print()
    print("== プロンプト prefill (冷たい側。ここは mmap が払う側になりうる)")
    print(delta("prefill_s", " s"))
    print(delta("ttft_s", " s"))
    if any(r["residency_ms_tok"] for r in mm):
        print()
        print(f"== residency set の更新 (49 §3 の prep が production でいくらか)")
        print(f"   {median([r['residency_ms_tok'] for r in mm]):.2f} ms/tok "
              f"(sync {median([r['syncs'] for r in mm]):.0f} 回。prefill を含む)")
    print(f"\n   一次ログ: {args.out}")
    log.close()


if __name__ == "__main__":
    main()
