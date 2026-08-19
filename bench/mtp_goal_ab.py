#!/usr/bin/env python3
"""採点基準 v1 (`docs/mtp/22-GOAL-RESET.md` §4-§5) をそのまま実行する。

ゴールタスク 8 枚を **MTP on / off を交互に**走らせ、画像ごとの
end-to-end 壁時計比を出し、その**中央値**を 1 本のスコアにする。
22 §5 が凍結した目標 1.33 / 中止 1.10 と比べるのはこの数字だけである。

条件 (22 §4、変えるときは v2 を切って旧数字を測り直すこと):

  - 画像 × 固定プロンプト `bench/mtp_goal_prompt.json`
    (画像は `$TF_SAMPLE_IMGS`、既定 `~/Pictures/sample_imgs`。**リポジトリ外**)
  - Reasoning ON / temp 0 / 80 スロット / 画像ごとにプロセスを分ける
  - 生成長に上限を置かない。`--max-new 2048` は保険で、**到達したら FAIL**
  - wall = load + ttft + decode (footer から拾う。プロセス全体の実時間は
    Metal のティアダウンを含むので使わない)

on/off の順序は画像ごとに入れ替える。マシンの温まりが片側だけに乗るのを
防ぐためで、22 §5 の「比は同一セッションの交互 A/B で取る」の実装である。

  ./bench/mtp_goal_ab.py                     # bs=4、ディレクトリ内の全画像
  ./bench/mtp_goal_ab.py --block-size 3      # bs を変える
  ./bench/mtp_goal_ab.py --images a.png b.png --max-new 256    # 動作確認
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
# TF_SAMPLE_IMGS で差し替える。--images を省くとディレクトリ内の全画像を名前順に使う。
# **どの枚数を採点に使うかは未凍結である** — 採点基準 v2 が切れるまで、この既定は
# 「そこに在るもの全部」であって基準ではない (docs/mtp/22 §5-5)。
IMAGES_DIR = pathlib.Path(
    os.environ.get("TF_SAMPLE_IMGS", os.path.expanduser("~/Pictures/sample_imgs"))
)


def available_images() -> list:
    if not IMAGES_DIR.is_dir():
        raise SystemExit(f"画像ディレクトリが無い: {IMAGES_DIR} (TF_SAMPLE_IMGS で指定する)")
    return sorted(f.name for f in IMAGES_DIR.iterdir()
                  if not f.name.startswith(".") and f.is_file())

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
    # decode 側の expert 統計。ブロックは 1 回の verify で k トークンぶんの
    # ルーティング和集合を畳むので、要求数とミス数が別々に動く (§4)。
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
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="scratch/gemma4-qat.gturbo")
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--slots", type=int, default=80)
    parser.add_argument("--image-tokens", type=int, default=280)
    parser.add_argument("--max-new", type=int, default=2048)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--out", default="bench/logs/mtp_goal_ab.tsv")
    args = parser.parse_args()

    if args.images is None:
        args.images = available_images()
    missing = [i for i in args.images if not (IMAGES_DIR / i).is_file()]
    if missing:
        raise SystemExit(f"画像が見つからない: {missing} (探した先: {IMAGES_DIR})")

    # 34 §1-L6 / P6: 使った画像の md5 を run ごとに記録する。凍結ではなく、
    # 「この run が何を見たか」を後から検証できるようにするためのもの。
    digests = []
    for name in args.images:
        h = hashlib.md5((IMAGES_DIR / name).read_bytes()).hexdigest()
        digests.append(f"{h}  {name}")

    rows = []
    ratios = []
    print(f"# goal-condition A/B, bs={args.block_size}, {len(args.images)} images")
    print(f"# images from {IMAGES_DIR}")
    print("image\torder\twall_off\twall_on\tnew_off\tnew_on\tratio\taccept\tstop")
    for index, image in enumerate(args.images):
        # 交互: 偶数枚目は off を先に、奇数枚目は on を先に。
        order = ["off", "on"] if index % 2 == 0 else ["on", "off"]
        result = {}
        for side in order:
            result[side] = run_once(args, image,
                                    args.block_size if side == "on" else 0)
        off, on = result["off"], result["on"]
        ratio = off["wall"] / on["wall"]
        ratios.append(ratio)
        rows.append({"off": off, "on": on, "order": "/".join(order), "ratio": ratio})
        flag = ""
        if off.get("stop") == "maxTokens" or on.get("stop") == "maxTokens":
            flag = "  FAIL(maxTokens)"
        print(f"{image}\t{'/'.join(order)}\t{off['wall']:.1f}\t{on['wall']:.1f}"
              f"\t{int(off['new'])}\t{int(on['new'])}\t{ratio:.3f}"
              f"\t{on.get('accept', float('nan')):.3f}"
              f"\t{off.get('stop')}/{on.get('stop')}{flag}")
        sys.stdout.flush()

    median = statistics.median(ratios)
    # 生成長は on/off で揃わない (§2 の分岐で答えの長さが変わる) ので、
    # 壁時計比とは別に「1 トークンあたりの速さ」も出す。スコアは前者だが、
    # 前者が動いた理由が長さなのか速さなのかは後者でしか読めない。
    speed = [r["on"]["tok_s"] / r["off"]["tok_s"] for r in rows]
    lengths = [r["on"]["new"] / r["off"]["new"] for r in rows]
    print()
    print(f"median decode t/s ratio (on / off):      {statistics.median(speed):.3f}")
    print(f"median generated-length ratio (on/off):  {statistics.median(lengths):.3f}")
    print(f"median end-to-end wall ratio (off / on): {median:.3f}")
    print(f"  target 1.33 / abort 1.10 (22-GOAL-RESET §5) -> "
          f"{'target met' if median >= 1.33 else ('above abort' if median >= 1.10 else 'below abort')}")
    identical = sum(1 for r in rows if r["off"]["text"] == r["on"]["text"])
    print(f"  identical text off vs on: {identical}/{len(rows)} "
          f"(04-PHASES §3 gate 1 records where it diverges, it does not require equality)")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        handle.write(f"# images_dir\t{IMAGES_DIR}\n")
        for line in digests:
            handle.write(f"# md5\t{line}\n")
        handle.write("image\torder\tside\tblock_size\twall\tload\tttft\tdecode\tnew\ttok_s"
                     "\tpeak\tstop\trounds\taccept\tdraft\tverify"
                     "\tdecode_hit\tdecode_experts\tdecode_io\tratio\n")
        for row in rows:
            for side in ("off", "on"):
                r = row[side]
                handle.write("\t".join(str(x) for x in [
                    r["image"], row["order"], side, r["block_size"],
                    f"{r['wall']:.3f}", r["load"], r["ttft"], r["decode"],
                    int(r["new"]), r["tok_s"], r.get("peak", ""), r.get("stop", ""),
                    int(r.get("rounds", 0)), r.get("accept", ""),
                    r.get("draft", ""), r.get("verify", ""),
                    r.get("decode_hit", ""), int(r.get("decode_experts", 0)),
                    r.get("decode_io", ""),
                    f"{row['ratio']:.4f}",
                ]) + "\n")
    print(f"  wrote {args.out}")

    texts = {r["off"]["image"]: {"off": r["off"]["text"], "on": r["on"]["text"]}
             for r in rows}
    text_path = out_path.with_suffix(".texts.json")
    text_path.write_text(json.dumps(texts, ensure_ascii=False, indent=1))
    print(f"  wrote {text_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
