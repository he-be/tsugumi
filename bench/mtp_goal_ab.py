#!/usr/bin/env python3
"""採点基準 v2 (`docs/mtp/42-N3-SCORING-V2.md`) を実行する。

ゴールタスク **16 枚**を **MTP on / off を交互に**走らせ、画像ごとの
end-to-end 壁時計比を出し、その**中央値**を 1 本のスコアにする。

v1 (22 §4-§5) との違いは 3 つだけである。**目標 1.33 / 中止 1.10 は書き換えない。**

  1. **画像。**v1 の 8 枚は失われた (38 §6)。v2 は `bench/mtp_goal_v2_images.tsv` が
     凍結する 16 枚で、**v1 との絶対値の連続性は無い** (40 §3)。
  2. **スロット 80 → 32。**`Scripts/demo/serve.py` が実際に使う設定である
     (27 §2c が v1 をこの条件で走らせている)。48 は c(slots) のトレンドを
     見るために振っただけで、運用点ではない。
  3. **衛生と共変量。**クールダウン / 暖機 / ドリフト検定 / 目盛り run を
     ドライバが持ち (P6)、footer の `decode hit%` と `io` を列にする (N3 設計 2-3)。

条件 (**変えるときは v3 を切って旧数字を測り直すこと**):

  - 16 枚 × 固定プロンプト `bench/mtp_goal_prompt.json` / Reasoning ON / temp 0 /
    **32 スロット** / bs=4 / 画像ごとにプロセスを分ける
  - 生成長に上限を置かない。`--max-new 2048` は保険で、**到達したら FAIL**
  - wall = load + ttft + decode (footer から拾う。プロセス全体の実時間は
    Metal のティアダウンを含むので使わない)
  - **temp 0 は運用点ではない** (運用点は temp 1.0。40 §4-4)。採点で 0 を使うのは
    on/off の生成長を揃えるためで、v1 から引き継いだ 1 点である。

**同じ画像を反復してもスコアの足元は増えない** — temp 0 では出力が決定論的で、
動くのはマシンの状態だけだからである。反復は 1 本にし、代わりに画像を増やした。
その代わり **1 枚目 (`REFERENCE`) だけを先頭・中盤・末尾で 3 回**測り、

  - 先頭 対 末尾 = **熱ドリフト検定** (|差| > 5% ならスイープごと疑う。39 §3 の 4 つ目)
  - 3 本の幅 = **目盛り**。これ以下の差を本文で読んではいけない (36 §9g)

として使う。

  ./bench/mtp_goal_ab.py                       # v2 のスコア (18 ペア、約 40 分)
  ./bench/mtp_goal_ab.py --cooldown 20         # 衛生の既定に戻す
  ./bench/mtp_goal_ab.py --images a.png --max-new 256   # 動作確認 (スコアにならない)
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 画像はリポジトリ外に置く (追跡もプッシュもしない)。既定は ~/Pictures/sample_imgs で、
# TF_SAMPLE_IMGS で差し替える。採点に使う 16 枚は下の台帳が凍結している。
IMAGES_DIR = pathlib.Path(
    os.environ.get("TF_SAMPLE_IMGS", os.path.expanduser("~/Pictures/sample_imgs"))
)
LEDGER = ROOT / "bench/mtp_goal_v2_images.tsv"

# 採点基準 v2 が凍結する測定条件。--images で上書きすると採点ではなくなる。
V2_SLOTS = 32
V2_BLOCK_SIZE = 4
# 20 秒はスイープの衛生 (36 §6)。v2 は 10 秒で凍結する (2026-08-20 のユーザー判断) —
# 熱に余裕があるという読みで、その読みが正しかったかは末尾のドリフト検定が示す。
V2_COOLDOWN_S = 10
DRIFT_LIMIT_PCT = 5.0


def load_ledger() -> list:
    """凍結された 16 枚を読み、md5 を突き合わせる。1 つでも違えば走らない。"""
    if not LEDGER.is_file():
        raise SystemExit(f"台帳が無い: {LEDGER}")
    rows = []
    for line in LEDGER.read_text().splitlines():
        if line.startswith("#") or line.startswith("pool_index"):
            continue
        pool_index, md5, size, name = line.split("\t")
        rows.append({"pool_index": int(pool_index), "md5": md5,
                     "bytes": int(size), "name": name})
    if not IMAGES_DIR.is_dir():
        raise SystemExit(f"画像ディレクトリが無い: {IMAGES_DIR} (TF_SAMPLE_IMGS で指定する)")
    bad = []
    for row in rows:
        path = IMAGES_DIR / row["name"]
        if not path.is_file():
            bad.append(f"  無い: {row['name']}")
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != row["md5"]:
            bad.append(f"  md5 不一致: {row['name']} {digest} != {row['md5']}")
    if bad:
        raise SystemExit("採点画像が台帳と合わない (v2 のスコアにならない):\n"
                         + "\n".join(bad) + f"\n台帳: {LEDGER}")
    return rows


FOOTER_PATTERNS = {
    "new": r"new=(\d+)tok",
    "decode": r"decode=([\d.]+)s",
    "tok_s": r"tok/s=([\d.]+)",
    "load": r"load=([\d.]+)s",
    "ttft": r"ttft=([\d.]+)s",
    "peak": r"peak=([\d.]+)GB",
    "stop": r"stop=(\w+)",
    "accept": r"accept=([\d.]+)/",
    "rounds": r"rounds=(\d+)",
    "draft": r"draft=([\d.]+)s",
    "verify": r"verify=([\d.]+)s",
    # decode 側の expert 統計 = N3 の共変量。ブロックは 1 回の verify で k トークンぶんの
    # ルーティング和集合を畳むので、要求数とミス数が別々に動く。
    # 41 §1a: `2^H` (と代理ラベルの register) は入力ではなく出力なので、
    # 「どの register で測るか」は選べない。**選べないものは共変量として記録する。**
    "decode_hit": r"decode hit=([\d.]+)%",
    "decode_experts": r"decode hit=[\d.]+% \d+/(\d+)",
    "decode_io": r"decode hit=[\d.]+% \d+/\d+ io=([\d.]+)s",
}


def parse_footer(stderr: str) -> dict:
    out = {}
    for key, pattern in FOOTER_PATTERNS.items():
        match = re.search(pattern, stderr)
        if match:
            value = match.group(1)
            out[key] = value if key == "stop" else float(value)
    return out


def run_once(args, image: str, block_size: int) -> dict:
    cmd = [
        str(ROOT / ".build/release/TurboFieldfareCLI"),
        "--model", args.model,
        "--messages-file", str(ROOT / "bench/mtp_goal_prompt.json"),
        "--image", str(IMAGES_DIR / image),
        "--image-tokens", str(args.image_tokens),
        "--thinking", "on",
        "--temperature", "0",
        "--max-new", str(args.max_new),
        "--max-context", str(args.max_context),
        "--expert-cache-slots", str(args.slots),
        "--verification", "trusted-install",
        "--draft-block-size", str(block_size),
    ]
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        raise SystemExit(f"run failed ({image}, bs={block_size}): {proc.stderr[-800:]}")
    row = parse_footer(proc.stderr)
    row["image"] = image
    row["block_size"] = block_size
    row["process_seconds"] = elapsed
    row["text"] = proc.stdout
    row["wall"] = row["load"] + row["ttft"] + row["decode"]
    # 共変量は 1 産出トークンあたりに正規化する。footer の io は run 全体の秒で、
    # 生成長が on/off で揃わないので秒のままでは比較できない (§3)。
    row["io_ms_tok"] = (row.get("decode_io", 0.0) * 1000.0 / row["new"]) if row.get("new") else 0.0
    return row


def cooldown(seconds: float, note: str = "") -> None:
    if seconds > 0:
        print(f"# cooldown {seconds:g}s {note}", flush=True)
        time.sleep(seconds)


def run_pair(args, image: str, index: int, rep: int, seq: int) -> dict:
    """1 枚の on/off ペア。順序は交互 (マシンの温まりが片側に乗るのを防ぐ)。"""
    order = ["off", "on"] if index % 2 == 0 else ["on", "off"]
    result = {}
    for side in order:
        result[side] = run_once(args, image, args.block_size if side == "on" else 0)
        cooldown(args.cooldown)
    off, on = result["off"], result["on"]
    return {"off": off, "on": on, "order": "/".join(order),
            "ratio": off["wall"] / on["wall"], "image": image,
            "rep": rep, "seq": seq}


def spread_pct(values: list) -> float:
    return (max(values) - min(values)) / statistics.mean(values) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="scratch/gemma4-qat.gturbo")
    parser.add_argument("--images", nargs="*", default=None,
                        help="台帳を無視して任意の画像で走る (採点にはならない)")
    parser.add_argument("--block-size", type=int, default=V2_BLOCK_SIZE)
    parser.add_argument("--slots", type=int, default=V2_SLOTS)
    parser.add_argument("--image-tokens", type=int, default=280)
    parser.add_argument("--max-new", type=int, default=2048)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--cooldown", type=float, default=V2_COOLDOWN_S)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--out", default="bench/logs/mtp_goal_ab.tsv")
    args = parser.parse_args()

    scoring = args.images is None
    if scoring:
        ledger = load_ledger()
        images = [row["name"] for row in ledger]
        digests = [f"{row['md5']}  {row['name']}" for row in ledger]
    else:
        images = args.images
        missing = [i for i in images if not (IMAGES_DIR / i).is_file()]
        if missing:
            raise SystemExit(f"画像が見つからない: {missing} (探した先: {IMAGES_DIR})")
        digests = [f"{hashlib.md5((IMAGES_DIR / n).read_bytes()).hexdigest()}  {n}"
                   for n in images]

    # 冷温プロトコル (P6): 他の推論プロセスが居ないことを確認し、機械の状態を記録する。
    others = subprocess.run(["pgrep", "-f", "TurboFieldfare"],
                            capture_output=True, text=True).stdout.split()
    others = [p for p in others if p != str(os.getpid())]
    if others:
        raise SystemExit(f"他の TurboFieldfare プロセスが居る (pid {others})。GPU は 1 個で使う。")
    wired = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                           capture_output=True, text=True).stdout.strip()

    reference = images[0]
    # 走らせる順: REFERENCE を先頭・中盤・末尾に置き、間に残り 15 枚を挟む。
    # REFERENCE の 3 本が (a) 熱ドリフト検定 (b) 分解能の目盛り を兼ねる。
    rest = images[1:]
    half = len(rest) // 2
    plan = ([(reference, "head")]
            + [(n, "body") for n in rest[:half]]
            + [(reference, "mid")]
            + [(n, "body") for n in rest[half:]]
            + [(reference, "tail")])

    print(f"# scoring basis: {'v2' if scoring else 'ad-hoc (NOT a score)'}")
    print(f"# bs={args.block_size} slots={args.slots} temp=0 thinking=on "
          f"cooldown={args.cooldown:g}s images={len(images)} pairs={len(plan)}")
    print(f"# images from {IMAGES_DIR}   iogpu.wired_limit_mb={wired}")

    if not args.no_warmup:
        # 暖機 1 本 (39 §3)。スロット切り替え直後のページイン費用を測定に載せない。
        print(f"# warmup (discarded): {reference} bs=0", flush=True)
        run_once(args, reference, 0)
        cooldown(args.cooldown, "(after warmup)")

    rows = []
    print("image\trole\torder\twall_off\twall_on\tnew_off\tnew_on\tratio"
          "\thit_on\tio_on\taccept\tstop")
    for seq, (image, role) in enumerate(plan):
        row = run_pair(args, image, seq, rep=role, seq=seq)
        row["role"] = role
        rows.append(row)
        off, on = row["off"], row["on"]
        flag = ""
        if off.get("stop") == "maxTokens" or on.get("stop") == "maxTokens":
            flag = "  FAIL(maxTokens)"
        print(f"{image}\t{role}\t{row['order']}\t{off['wall']:.1f}\t{on['wall']:.1f}"
              f"\t{int(off['new'])}\t{int(on['new'])}\t{row['ratio']:.3f}"
              f"\t{on.get('decode_hit', float('nan')):.1f}\t{on['io_ms_tok']:.2f}"
              f"\t{on.get('accept', float('nan')):.3f}"
              f"\t{off.get('stop')}/{on.get('stop')}{flag}")
        sys.stdout.flush()

    # --- スコア -------------------------------------------------------------
    # temp 0 は greedy なのでターゲットがループに入ることがある。投機は出力を変えない
    # ので on/off の両方に同じように起きる (原理上は比を壊さない) が、**採点には使えない**:
    #   (a) 繰り返し列は低エントロピーなので受理長と decode hit% が跳ね、MTP に有利に偏る
    #   (b) 25 §4 の argmax 反転で on/off が分岐すると、片側だけがループしうる
    # 22 §4 が「`--max-new` に到達したら FAIL、データ点にしない」と書いたのはこの穴で、
    # v1 のドライバは印を付けるだけだった。ここで実際に落とす。
    dropped = [r for r in rows
               if r["off"].get("stop") == "maxTokens" or r["on"].get("stop") == "maxTokens"]
    kept = [r for r in rows if r not in dropped]
    if not kept:
        raise SystemExit("全ペアが maxTokens に達した。スコアにならない。")
    by_image = {}
    for row in kept:
        by_image.setdefault(row["image"], []).append(row)
    cells = {name: statistics.median([r["ratio"] for r in rs])
             for name, rs in by_image.items()}
    ratios = [cells[n] for n in images if n in cells]
    median = statistics.median(ratios)

    print()
    if dropped:
        print(f"dropped {len(dropped)}/{len(rows)} pairs (stop=maxTokens, 22 §4):")
        for r in dropped:
            print(f"  {r['image']} ({r['role']}) off={r['off'].get('stop')} "
                  f"new={int(r['off']['new'])} / on={r['on'].get('stop')} "
                  f"new={int(r['on']['new'])}")
        print()
    # 長さが揃わないセルは壁時計比の一部が「長さの比」になる。落とさないが印を付ける。
    skewed = [r for r in kept if abs(r["on"]["new"] / r["off"]["new"] - 1.0) > 0.25]
    if skewed:
        names = ", ".join(f"{r['image']}({r['on']['new']:.0f}/{r['off']['new']:.0f})"
                          for r in skewed)
        print(f"length-skewed cells (|on/off - 1| > 25%): {names}")
        print()
    # 36 §9g: 中央値・平均・分布を並べる (どれか 1 つだけを載せない)。
    print(f"end-to-end wall ratio (off / on), n={len(ratios)} images")
    print(f"  median {median:.3f}   mean {statistics.mean(ratios):.3f}"
          + (f"   sd {statistics.stdev(ratios):.3f}" if len(ratios) > 1 else "")
          + f"   min {min(ratios):.3f}   max {max(ratios):.3f}")
    if len(ratios) > 3:
        q = statistics.quantiles(ratios, n=4)
        print(f"  quartiles {q[0]:.3f} / {median:.3f} / {q[2]:.3f}")
    print(f"  target 1.33 / abort 1.10 (22-GOAL-RESET §5) -> "
          f"{'target met' if median >= 1.33 else ('above abort' if median >= 1.10 else 'below abort')}")

    speed = [r["on"]["tok_s"] / r["off"]["tok_s"] for r in kept]
    lengths = [r["on"]["new"] / r["off"]["new"] for r in kept]
    print(f"  median decode t/s ratio (on/off):     {statistics.median(speed):.3f}")
    print(f"  median generated-length ratio (on/off): {statistics.median(lengths):.3f}")
    identical = sum(1 for r in kept if r["off"]["text"] == r["on"]["text"])
    print(f"  identical text off vs on: {identical}/{len(kept)} "
          f"(04-PHASES §3 gate 1 records where it diverges, it does not require equality)")

    # --- 共変量 (N3 設計 3): スコアは共変量と一緒に報告する ------------------
    print()
    print("covariates (on side, per produced token)")
    hits = [r["on"]["decode_hit"] for r in kept if "decode_hit" in r["on"]]
    ios = [r["on"]["io_ms_tok"] for r in kept]
    if hits:
        print(f"  decode hit%: median {statistics.median(hits):.1f}"
              f"  min {min(hits):.1f}  max {max(hits):.1f}")
    print(f"  expert io ms/tok: median {statistics.median(ios):.2f}"
          f"  min {min(ios):.2f}  max {max(ios):.2f}")
    # 比 と io の相関。GPU busy は register でもスロット数でも動かない (41 §1) ので、
    # セル間のばらつきがこの 1 変数で説明できるなら、v2 は別の日の数字と比較できる。
    if len(kept) > 2:
        xs = [r["on"]["io_ms_tok"] for r in kept]
        ys = [r["ratio"] for r in kept]
        try:
            r_io = statistics.correlation(xs, ys)
            print(f"  corr(io ms/tok, ratio) = {r_io:+.3f}  (n={len(kept)} pairs)")
        except statistics.StatisticsError:
            pass
        if hits and len(hits) == len(ys):
            try:
                print(f"  corr(hit%,      ratio) = "
                      f"{statistics.correlation(hits, ys):+.3f}")
            except statistics.StatisticsError:
                pass

    # --- ドリフト検定と目盛り (39 §3 の 4 つ目 / 36 §9g) ---------------------
    ref_rows = [r for r in kept if r["image"] == reference]
    if len(ref_rows) >= 2:
        head, tail = ref_rows[0], ref_rows[-1]
        drift = (tail["off"]["tok_s"] - head["off"]["tok_s"]) / head["off"]["tok_s"] * 100.0
        ref_ratios = [r["ratio"] for r in ref_rows]
        print()
        print(f"drift / resolution ({reference}, {len(ref_rows)} reps)")
        print(f"  off t/s head->tail: {head['off']['tok_s']:.2f} -> "
              f"{tail['off']['tok_s']:.2f}  ({drift:+.2f}%)"
              f"  {'OK' if abs(drift) <= DRIFT_LIMIT_PCT else 'SUSPECT (>5%)'}")
        print(f"  ratio reps: {' / '.join(f'{x:.3f}' for x in ref_ratios)}"
              f"   spread {spread_pct(ref_ratios):.2f}%"
              f"  <- 本文でこれ以下の差を読まないこと")

    # --- 出力 ---------------------------------------------------------------
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        handle.write(f"# basis\t{'v2' if scoring else 'ad-hoc'}\n")
        handle.write(f"# conditions\tslots={args.slots}\tbs={args.block_size}\ttemp=0"
                     f"\tthinking=on\tcooldown={args.cooldown:g}s"
                     f"\tmax_new={args.max_new}\timage_tokens={args.image_tokens}\n")
        handle.write(f"# images_dir\t{IMAGES_DIR}\tiogpu.wired_limit_mb={wired}\n")
        handle.write(f"# ledger\t{LEDGER.relative_to(ROOT)}\n")
        for line in digests:
            handle.write(f"# md5\t{line}\n")
        handle.write(f"# score\tmedian={median:.4f}\tn={len(ratios)}\n")
        handle.write("image\trole\torder\tside\tblock_size\twall\tload\tttft\tdecode\tnew\ttok_s"
                     "\tpeak\tstop\trounds\taccept\tdraft\tverify"
                     "\tdecode_hit\tdecode_experts\tdecode_io\tio_ms_tok\tratio\n")
        for row in rows:
            for side in ("off", "on"):
                r = row[side]
                handle.write("\t".join(str(x) for x in [
                    r["image"], row["role"], row["order"], side, r["block_size"],
                    f"{r['wall']:.3f}", r["load"], r["ttft"], r["decode"],
                    int(r["new"]), r["tok_s"], r.get("peak", ""), r.get("stop", ""),
                    int(r.get("rounds", 0)), r.get("accept", ""),
                    r.get("draft", ""), r.get("verify", ""),
                    r.get("decode_hit", ""), int(r.get("decode_experts", 0)),
                    r.get("decode_io", ""), f"{r['io_ms_tok']:.3f}",
                    f"{row['ratio']:.4f}",
                ]) + "\n")
    print(f"\n  wrote {args.out}")

    texts = {f"{r['image']}#{r['role']}": {"off": r["off"]["text"], "on": r["on"]["text"]}
             for r in rows}
    text_path = out_path.with_suffix(".texts.json")
    text_path.write_text(json.dumps(texts, ensure_ascii=False, indent=1))
    try:
        shown = text_path.relative_to(ROOT)
    except ValueError:      # --out にリポジトリ外の絶対パスを渡した場合
        shown = text_path
    print(f"  wrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
