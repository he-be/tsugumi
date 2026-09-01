#!/usr/bin/env python3
"""49 §12 — 今の `io` の何割がページキャッシュ由来かを ABBA で測る。

D (`mmap` + `bytesNoCopy`) が消せるのは「ページキャッシュ → 私有スロットの
memcpy」だけで、**ディスク → ページキャッシュは消せない** (`MAP_SHARED` も
同じキャッシュを読む)。だから D の賞金の上限は「今の `io` のうちコピーの分」
であって `io` 全体ではない。46 §6 が置いたまま未着手だった 1 手で決める。

腕は 2 つ。違いは `TF_EXPERT_NOCACHE` だけ:
  off  今の production (`PreadExpertStreamer.swift:109` は F_NOCACHE を立てない)
  on   同じ fd に `fcntl(F_NOCACHE, 1)` を立てて読みをキャッシュから外す

temp 0 なのでミスは決定論 (46 §5a)。**両腕でミス数が一致することを検定し、
一致しない run は捨てる** — 一致していればバイトは固定で、動くのは
バイトあたりの費用だけになる。

使い方:
    ./bench/mtp49_nocache_ab.py --rounds 3
    ./bench/mtp49_nocache_ab.py --rounds 3 --slots 32 --max-new 256
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics as st
import subprocess
import sys
import time

CLI = "./.build/release/TurboFieldfareCLI"
MODEL = os.environ.get("TF_MODEL", "scratch/gemma4-qat-sym.gturbo")
MESSAGES = os.environ.get("TF_MESSAGES", "bench/story.json")
COOL_S = int(os.environ.get("COOL_S", "10"))
EXPERT_MB = 3.35872          # expertStride 3,358,720 B (48 §1)

# [expert prefill hit=0.0% 0/998 io=0.270s | decode hit=85.8% 4734/5520 io=0.689s]
RE_EXPERT = re.compile(
    r"decode hit=([\d.]+)%\s+(\d+)/(\d+)\s+io=([\d.]+)s")
# [decode/tok io=28.76ms cb1=0.82ms cb2=0.42ms head=3.39ms]
RE_PERTOK = re.compile(r"decode/tok io=([\d.]+)ms")
# [stop=maxTokens prefill=11tok new=24tok decode=1.45s tok/s=16.588]
RE_RUN = re.compile(r"new=(\d+)tok decode=([\d.]+)s tok/s=([\d.]+)")


def run_once(nocache, args):
    env = dict(os.environ)
    env["TF_EXPERT_NOCACHE"] = "1" if nocache else "0"
    cmd = [CLI, "--model", args.model, "--messages-file", args.messages,
           "--temperature", "0", "--max-new", str(args.max_new),
           "--expert-cache-slots", str(args.slots),
           "--verification", "trusted-install"]
    started = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    blob = proc.stdout + proc.stderr
    if proc.returncode != 0:
        sys.stderr.write(blob[-2000:])
        raise SystemExit(f"CLI failed ({proc.returncode})")
    expert, pertok, run = (RE_EXPERT.search(blob), RE_PERTOK.search(blob),
                           RE_RUN.search(blob))
    if not (expert and pertok and run):
        sys.stderr.write(blob[-2000:])
        raise SystemExit("footer をパースできない")
    hits, requests = int(expert.group(2)), int(expert.group(3))
    misses = requests - hits
    steps = int(run.group(1)) - 1          # decode 歩数 (1 歩目は prefill 由来)
    io_ms = float(pertok.group(1))
    mb = misses * EXPERT_MB / max(steps, 1)
    # temp 0 なので生成文は腕によらず同一のはず。違えば速度比較は成り立たない。
    digest = hashlib.sha256(proc.stdout.encode()).hexdigest()[:12]
    return {"nocache": nocache, "misses": misses, "requests": requests,
            "digest": digest,
            "steps": steps, "io_ms": io_ms, "mb_tok": mb,
            "gbs": mb / 1e3 / (io_ms / 1e3) if io_ms else 0.0,
            "tok_s": float(run.group(3)), "wall": time.time() - started,
            "text": blob}


def median(values):
    return st.median(values) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--messages", default=MESSAGES)
    ap.add_argument("--out", default="bench/mtp49/nocache_ab.log")
    args = ap.parse_args()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    log = open(args.out, "w")
    log.write(f"# {' '.join(sys.argv)}\n# model={args.model} "
              f"messages={args.messages} slots={args.slots} "
              f"max_new={args.max_new} cool={COOL_S}s\n")

    results = {False: [], True: []}
    # ABBA: off on on off。位置効果を腕の両側から食わせる (48 §5a)。
    for round_index in range(args.rounds):
        for nocache in (False, True, True, False):
            result = run_once(nocache, args)
            results[nocache].append(result)
            arm = "NOCACHE" if nocache else "cached "
            line = (f"round {round_index + 1}  {arm}  "
                    f"io {result['io_ms']:7.2f} ms/tok  "
                    f"{result['mb_tok']:7.1f} MB/tok  "
                    f"{result['gbs']:5.2f} GB/s  "
                    f"miss {result['misses']}/{result['requests']}  "
                    f"out {result['digest']}  "
                    f"tok/s {result['tok_s']:.3f}")
            print(line, flush=True)
            log.write(line + "\n")
            log.write(result["text"] + "\n")
            log.flush()
            time.sleep(COOL_S)

    off, on = results[False], results[True]
    miss_off = {r["misses"] for r in off}
    miss_on = {r["misses"] for r in on}
    print()
    print("== ミス数の検定 (temp 0 なら両腕で同一のはず、46 §5a)")
    print(f"   cached  {sorted(miss_off)}")
    print(f"   NOCACHE {sorted(miss_on)}")
    fixed = len(miss_off | miss_on) == 1
    print("   " + ("バイトは固定されている ⇒ 動いたのは費用だけ" if fixed
                   else "*** ミス数が動いた。この比較は成り立たない ***"))
    digests = {r["digest"] for r in off} | {r["digest"] for r in on}
    print(f"   生成文の sha256 (先頭 12): {sorted(digests)}")
    print("   " + ("両腕で同一 ⇒ 同じ 256 トークンを出している" if len(digests) == 1
                   else "*** 出力が違う。速度比較は成り立たない ***"))

    io_off, io_on = median([r["io_ms"] for r in off]), median([r["io_ms"] for r in on])
    gb_off, gb_on = median([r["gbs"] for r in off]), median([r["gbs"] for r in on])
    print()
    print(f"{'腕':<10}{'n':>3}{'io ms/tok':>12}{'GB/s':>8}{'tok/s':>9}")
    for label, group in (("cached", off), ("NOCACHE", on)):
        print(f"{label:<10}{len(group):>3}{median([r['io_ms'] for r in group]):>12.2f}"
              f"{median([r['gbs'] for r in group]):>8.2f}"
              f"{median([r['tok_s'] for r in group]):>9.3f}")
    print(f"\n   キャッシュが今どれだけ効いているか: "
          f"io {io_off:.2f} -> {io_on:.2f} ms/tok "
          f"({io_on - io_off:+.2f}, x{io_on / io_off:.2f})")

    # D の賞金の上限。暖かい pread の実効帯域 (48 §3 の対照、8 エキスパート
    # 26.87 MB を 0.84〜1.26 ms = 21.3〜32.0 GB/s) で全バイトをコピーしたら
    # 何 ms かかるか = 全部キャッシュに載っていた場合のコピー費用。
    mb = median([r["mb_tok"] for r in off])
    copy_lo, copy_hi = mb / 32.0, mb / 21.3      # ms/tok
    print(f"\n== D の賞金 (導出。48 §3 の暖かい pread 21.3〜32.0 GB/s を単価に)")
    print(f"   {mb:.1f} MB/tok を全部コピーで運ぶと {copy_lo:.2f}〜{copy_hi:.2f} ms/tok")
    if io_on > io_off:
        # io = f*(コピー) + (1-f)*(ディスク)。NOCACHE 側を全ディスクとみなす。
        for label, copy in (("下端", copy_lo), ("上端", copy_hi)):
            frac = (io_on - io_off) / max(io_on - copy, 1e-9)
            frac = min(max(frac, 0.0), 1.0)
            print(f"   {label}: キャッシュ由来 {frac * 100:.0f}% "
                  f"⇒ D が消せるのは約 {frac * copy:.2f} ms/tok "
                  f"(今の io {io_off:.2f} の {frac * copy / io_off * 100:.0f}%)")
    else:
        # 予測と逆。F_NOCACHE は「ディスク -> ページキャッシュ -> ユーザ緩衝」の
        # 2 段を「ディスク -> ユーザ緩衝」の 1 段にするので、キャッシュを外すと
        # 速くなりうる。その場合キャッシュは助けではなく費用である。
        print(f"   **NOCACHE のほうが速い** ({io_off - io_on:.2f} ms/tok 減、"
              f"x{io_off / io_on:.2f})。")
        print(f"   ⇒ 今の `io` のうち少なくとも {io_off - io_on:.2f} ms/tok は")
        print(f"     ページキャッシュを経由する費用であって、ディスクではない。")
        print(f"   ⇒ D の賞金の見積もりの前に、この 1 行のほうが先に効く。")
    print(f"\n   一次ログ: {args.out}")
    log.close()


if __name__ == "__main__":
    main()
