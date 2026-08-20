#!/usr/bin/env python3
"""48 §14 (P-5) の数字を bench/mtp48/p5_*.log から出し直す (導出のみ)。

新しい測定はしない。入力は
  bench/mtp48/p5_palindrome_run{1,2}.log  -- 交互順 A B B* C D D C B* B A
  bench/mtp48/p5_rotate_run{1,2}.log      -- 毎ラウンド 1 つずつ回す
  bench/mtp48/p5_polluted_run1.log        -- 汚染を先に入れた版 (対照の追試)
  bench/mtp48/p5_experts_{2,4,8,16}.log   -- ページ数を 8 倍振った傾き

出すもの:
  1. 腕ごとの 1st pass / warm / premium の中央値と、ブートストラップ 95% CI
  2. premium から出した 1 ページあたりの費用 (48 §11 の 2.44 us/page の位置)
  3. その ページ単価で引き直した 48 §12 の表
"""
import glob
import random
import re
import statistics as st

CLEAN = sorted(glob.glob("bench/mtp48/p5_palindrome_run*.log")
               + glob.glob("bench/mtp48/p5_rotate_run*.log"))
POLLUTED = sorted(glob.glob("bench/mtp48/p5_polluted_run*.log"))
SWEEP = [(n, f"bench/mtp48/p5_experts_{n}.log") for n in (2, 4, 8, 16)]
PAGES = 1640              # 8 experts x 205 pages, 48 §1
ARMS = ["A", "B", "B*", "C", "D"]
# 48 §12 の前提。ページ/tok はあちらの表から動かしていない。
STEP_PAGES = {"初回タッチだけ (46 trace, n=1)": 4490,
              "今のミスと同数 (45 §4)": 7882,
              "全要求 (240/tok)": 49200}
IO_MS_PER_TOK = 14.21     # 45 §4、対照


def read(path):
    """`--- raw, in trial order` の生値。腕 -> {1st, warm, prep}。"""
    text = open(path).read()
    block = text.split("--- raw, in trial order")[1]
    out, arm = {}, None
    for line in block.splitlines():
        head = re.match(r"^  ([AB*CD]+) ", line)
        if head:
            arm = head.group(1)
            out[arm] = {}
            continue
        for key, tag in (("1st pass", "1st"), ("warm", "warm"), ("prep", "prep")):
            if arm and line.strip().startswith(key):
                out[arm][tag] = [float(x) for x in line.split()[1 if key != "1st pass" else 2:]]
    return out


def pool(paths):
    out = {}
    for path in paths:
        for arm, series in read(path).items():
            for tag, values in series.items():
                out.setdefault(arm, {}).setdefault(tag, []).extend(values)
    return out


def boot(values, reps=20000, seed=48):
    rng = random.Random(seed)
    medians = sorted(st.median(rng.choices(values, k=len(values))) for _ in range(reps))
    return medians[int(0.025 * reps)], medians[int(0.975 * reps)]


def main():
    clean = pool(CLEAN)
    print(f"== 1. 清浄なプロセス、{len(CLEAN)} invocation を pool "
          f"(交互順 palindrome 2 本 + rotate 2 本)")
    print(f"{'腕':<4} {'n':>4} {'1st pass':>10} {'warm':>8} {'premium':>9}"
          f" {'premium 95% CI':>18} {'us/page':>9}")
    per_page = {}
    for arm in ARMS:
        first, warm = clean[arm]["1st"], clean[arm]["warm"]
        prem = [a - b for a, b in zip(first, warm)]
        lo, hi = boot(prem)
        per_page[arm] = st.median(prem) * 1e3 / PAGES
        print(f"{arm:<4} {len(first):>4} {st.median(first):>9.2f}m"
              f" {st.median(warm):>7.2f}m {st.median(prem):>8.2f}m"
              f" {lo:>8.2f}..{hi:<8.2f} {per_page[arm]:>8.2f}")
    print(f"\n   48 §11 は同じ量を 2.44 us/page と置いた (対照 4.25 ms、n=20、")
    print(f"   P-1〜P-4 の全景を走らせたプロセスの中)。")

    if POLLUTED:
        dirty = pool(POLLUTED)
        print(f"\n== 2. 汚染を先に入れた版 ({len(POLLUTED)} invocation) — 対照の追試")
        print(f"{'腕':<4} {'1st pass':>10} {'clean との差':>14}")
        for arm in ARMS:
            d = st.median(dirty[arm]["1st"])
            c = st.median(clean[arm]["1st"])
            print(f"{arm:<4} {d:>9.2f}m {d - c:>+13.2f}m")

    print(f"\n== 3. premium はページ数に比例するか (--mmap-p5-experts の掃引)")
    print(f"   §11 も §12 も「1 ページあたり」を仮定して外挿している。振ってみる。")
    print(f"{'エキスパート':>10} {'ページ':>7} " + "".join(f"{a:>18}" for a in ARMS))
    flat = {}
    for count, path in SWEEP:
        try:
            data = pool([path])
        except (IOError, IndexError):
            continue
        pages = count * 205
        cells = ""
        for arm in ARMS:
            prem = st.median([a - b for a, b in
                              zip(data[arm]["1st"], data[arm]["warm"])])
            flat.setdefault(arm, []).append(prem)
            cells += f"{prem:>9.2f}m{prem * 1e3 / pages:>8.2f}u"
        print(f"{count:>10} {pages:>7} " + cells)
    print(f"   (各セル: premium ms / それをページ数で割った us)")
    for arm in ARMS:
        if len(flat.get(arm, [])) == len(SWEEP):
            lo, hi = min(flat[arm]), max(flat[arm])
            series = flat[arm]
            rising = all(b >= a for a, b in zip(series, series[1:]))
            verdict = ("固定費 — 8 倍振っても平ら" if hi <= 2.2 * lo
                       else ("ページ数とともに単調に増える" if rising
                             else "動くが単調でない (ページ数では説明できない)"))
            print(f"   arm {arm:<3} premium {lo:.2f}..{hi:.2f} ms  -> {verdict}")

    print(f"\n== 4. 48 §12 の表を P-5 のページ単価で引き直す (直列に積んだ場合)")
    print(f"{'前提':<34} {'ページ/tok':>10} " + "".join(f"{'arm ' + a:>10}" for a in ARMS))
    for label, pages in STEP_PAGES.items():
        cells = "".join(f"{pages * per_page[a] / 1e3:>9.2f}m" for a in ARMS)
        print(f"{label:<34} {pages:>10}" + cells)
    print(f"{'対照: 今の io (45 §4)':<34} {'':>10}" + f"{IO_MS_PER_TOK:>9.2f}m")
    print("\n   48 §12 の同じ表は 10.95 / 19.23 / 120.0 ms/tok だった。")
    print("   ただし §3 が示すとおり arm B の premium はページ数で動かないので、")
    print("   この表の立て方 (ページ/tok × 単価) 自体が arm B には当たらない。")
    print("   コマンドバッファあたりの固定費として積むなら 30 層 x "
          f"{st.median(flat['B']):.2f} ms = {30 * st.median(flat['B']):.2f} ms/tok。")


if __name__ == "__main__":
    main()
