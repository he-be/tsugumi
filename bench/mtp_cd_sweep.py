#!/usr/bin/env python3
"""turbo 本番 CLI から c・d・ラウンド外費用を単離する計器 (bs × task スイープ)。

出典:
  - `docs/mtp/35-LLAMACPP-CALIBRATION.md` §4 — 参照実装 (llama.cpp) の実測で
    「ブロックの壁時計は `step(k) = base + Σc_i + d·k` に乗る」ことを示した。
    c (verify 位置 1 つの限界費用) と d (ドラフト 1 本の費用) の 2 値だけで
    投機の損益が計算できる、という結論をそのまま turbo 本番バイナリに適用する。
  - 同 §6(d) — 「turbo 側で d を単離しないと次の判断ができない」
    「bs=2 (ドラフト 1 本) が実験行列に無いのは穴」「行列はタスク軸と
    直積で回さないと最適 k は決まらない」の 3 点がこのドライバの理由。
  - `docs/mtp/34-M9-PROPOSAL.md` §2c — 「ラウンドの外の費用」
    (ドラフト 3 歩、受理判定の logits 読み戻し、KV 巻き戻し、detokenize) は
    「誰も測っていない」と名指しされている。このドライバは
    `outside_s = decode - draft - verify` としてこれを毎行出す。

このドライバはモデルを起動する以外の一切を担当する計器であり、
turbo 本番 CLI (`TurboFieldfareCLI`) を子プロセスとして 1 実行 = 1 プロセスで
呼ぶだけである。`Sources/` はもちろん、他の `bench/*.py` にも触れない。

軸:
  - タスク 3 本 (`--tasks` で選ぶ。既定は 3 本全部):
      prose = --messages-file bench/story.json
      math  = --messages-file bench/math.json
      image = --messages-file bench/mtp_goal_prompt.json
              --image sample_imgs/IMG_2113.JPG --image-tokens 280
  - ブロック幅 `--bs` (既定 0,2,3,4,6,8。0 = 投機オフ)。
    各タスクについて、この並びの末尾にもう一度 bs=0 を足して測る
    (熱ドリフトの検定。先頭 base と末尾 base の t/s 差を最後に報告する)。

固定条件 (すべて CLI フラグで明示的に渡す。既定値に頼らない):
  --temperature 1.0 --top-k 64 --top-p 0.95 --seed 1234 --max-new 192
  --max-context 4096 --expert-cache-slots 32 --thinking off
  --verification trusted-install --prefill on
  (--slots / --max-new / --seed はこのドライバの引数で上書きできる)

実行の作法:
  - 1 実行 = 1 プロセス (`subprocess.run`, capture_output=True, cwd=リポジトリルート)。
  - 各実行の直前に `pgrep -fl 'TurboFieldfare|llama-server|llama-bench|mlx'` を
    確認し、自分以外の推論プロセスが居たら実行せず異常終了する。
  - 実行と実行の間に `COOL_S` (環境変数、既定 20) 秒スリープする
    (`bench/llamacpp_spec_sweep.py` の作法を流用。最後の実行の後はスリープしない)。
  - 1 実行ごとに結果をその場で標準出力に 1 行印字して flush する。
  - 個々の実行が失敗しても全体を止めず、その行を FAIL として記録して次へ進む
    (ただし pgrep 違反は即停止する)。

使い方の例:
  ./bench/mtp_cd_sweep.py                       # 3 タスク × bs 既定列、本番
  ./bench/mtp_cd_sweep.py --tasks math           # math だけ
  ./bench/mtp_cd_sweep.py --bs 0,2,4 --max-new 32 --dry-run   # 動作確認のみ
  COOL_S=5 ./bench/mtp_cd_sweep.py --bs 0,2      # クールダウンを短縮

`--dry-run` は実際には何も起動せず、組み立てたコマンド列を順に印字して終わる。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `bench/mtp_goal_ab.py` の FOOTER_PATTERNS をそのまま流用する。
# 出典 (実物の footer 出力行): Sources/TurboFieldfareCLI/Run.swift
#   L242 stop=/prefill=/new=/decode=/tok/s=
#   L281 load=
#   L284 ttft=
#   L285 peak=
#   L292-293 decode hit=.../io=
#   L311-315 mtp bs=/rounds=/accept=/draft=/verify=
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
    "decode_hit": r"decode hit=([\d.]+)%",
    "decode_experts": r"decode hit=[\d.]+% \d+/(\d+)",
    "decode_io": r"decode hit=[\d.]+% \d+/\d+ io=([\d.]+)s",
    "tok_per_round": r"tok/round=([\d.]+)",
    # 受理長ヒストグラム (mtp.acceptedHistogram)。カンマ区切りの生文字列の
    # まま保持する (他の列と違い float に変換しない — STRING_FOOTER_KEYS 参照)。
    "accepted_hist": r"accepted=([\d,]+)",
}

# parse_footer で float に変換せず文字列のまま row に入れる列。
STRING_FOOTER_KEYS = {"stop", "accepted_hist"}

TASKS = {
    "prose": {"messages_file": "bench/story.json"},
    "math": {"messages_file": "bench/math.json"},
    "image": {
        "messages_file": "bench/mtp_goal_prompt.json",
        "image": "sample_imgs/IMG_2113.JPG",
        "image_tokens": 280,
    },
}

COOL_S = int(os.environ.get("COOL_S", "20"))

INFERENCE_PROC_PATTERN = "TurboFieldfare|llama-server|llama-bench|mlx"


def parse_footer(stderr: str) -> dict:
    """footer から値を拾う。bs=0 では mtp 行そのものが出ないため、
    accept/rounds/draft/verify は欠けうる。呼び出し側は .get() で読むこと。"""
    out = {}
    for key, pattern in FOOTER_PATTERNS.items():
        match = re.search(pattern, stderr)
        if match:
            value = match.group(1)
            out[key] = value if key in STRING_FOOTER_KEYS else float(value)
    return out


def check_no_other_inference_proc() -> None:
    proc = subprocess.run(
        ["pgrep", "-fl", INFERENCE_PROC_PATTERN],
        capture_output=True, text=True,
    )
    # 監視シェル (`until pgrep -f 'TurboFieldfare…'` の類) は自分のコマンド
    # ライン自体がパターンにマッチする自己言及的な偽陽性を出す。2026-08-19 に
    # これでスイープが 1 度止まったので、シェルと pgrep 自身を除く。
    # 残すのは実際に GPU を掴む実行ファイルだけ。
    shellish = ("/bin/zsh", "/bin/sh", "/bin/bash", "pgrep", "grep", "until ", "while ")
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.strip() and not any(token in ln for token in shellish)]
    if lines:
        raise SystemExit(
            "他の推論プロセスが居る:\n" + "\n".join(lines)
        )


def build_cmd(args, task: str, bs: int) -> list:
    task_cfg = TASKS[task]
    cmd = [
        str(ROOT / ".build/release/TurboFieldfareCLI"),
        "--model", args.model,
        "--messages-file", str(ROOT / task_cfg["messages_file"]),
    ]
    if "image" in task_cfg:
        cmd += [
            "--image", str(ROOT / task_cfg["image"]),
            "--image-tokens", str(task_cfg["image_tokens"]),
        ]
    cmd += [
        "--temperature", str(args.temperature),
        "--top-k", "64",
        "--top-p", "0.95",
        "--seed", str(args.seed),
        "--max-new", str(args.max_new),
        "--max-context", "4096",
        "--expert-cache-slots", str(args.slots),
        "--thinking", "off",
        "--verification", "trusted-install",
        "--prefill", "on",
        "--draft-block-size", str(bs),
    ]
    return cmd


def run_once(args, task: str, bs: int, slot: str) -> dict:
    """slot は同じ (task, bs) の中での位置ラベル (先頭/末尾の base 判別用)。"""
    check_no_other_inference_proc()
    cmd = build_cmd(args, task, bs)
    row = {"task": task, "bs": bs, "slot": slot, "temperature": args.temperature,
           "cmd": " ".join(cmd)}
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    except Exception as exc:  # プロセス起動自体が失敗した場合も止めずに続行
        row["status"] = "FAIL"
        row["error"] = str(exc)
        return row
    elapsed = time.monotonic() - started
    row["process_seconds"] = elapsed
    if proc.returncode != 0:
        row["status"] = "FAIL"
        row["error"] = proc.stderr[-800:]
        return row
    footer = parse_footer(proc.stderr)
    row.update(footer)
    row["status"] = "OK"
    row["stderr"] = proc.stderr
    row["text"] = proc.stdout
    return row


def add_derived(row: dict) -> None:
    """生値から導出値を計算し row に書き足す。欠損は空文字のまま残す。"""
    if row.get("status") != "OK":
        return
    new = row.get("new")
    decode = row.get("decode")
    verify = row.get("verify")
    draft = row.get("draft")
    rounds = row.get("rounds")
    bs = row["bs"]

    if new and decode is not None and new > 0:
        row["ms_per_tok"] = decode / new * 1000.0

    if rounds and rounds > 0:
        if verify is not None:
            row["step_ms"] = verify / rounds * 1000.0
        if draft is not None:
            row["draft_per_round_ms"] = draft / rounds * 1000.0
        # drafts_total: ラウンド数 * (bs-1) の近似値。ヒストグラムには
        # 「各ラウンドが何本受理したか」しか残らず「何本ドラフトを生成したか」
        # は残らない (原理上は常に bs-1 本出すはずだが、実装が途中で打ち切る
        # 経路があるかどうかまではこの計器から検証できない) ため、正確な値
        # ではなく上限に近い近似であることを明示するために列名に _total を残す。
        drafts = rounds * (bs - 1) if bs > 1 else 0
        row["drafts"] = drafts
        row["drafts_total"] = drafts
        if draft is not None and drafts > 0:
            row["d_ms"] = draft / drafts * 1000.0
        if decode is not None and draft is not None and verify is not None:
            outside_s = decode - draft - verify
            row["outside_s"] = outside_s
            row["outside_per_round_ms"] = outside_s / rounds * 1000.0
    else:
        # bs=0 (投機オフ) では draft/verify/rounds が footer に出ない。
        # decode 全体がラウンド外費用にあたるものは無い (投機ループ自体が無い)
        # ので outside_s は計算しない — 欠損のままにして呼び出し側が判別する。
        pass

    # 受理長ヒストグラム (mtp.acceptedHistogram, h[i] = 受理長ちょうど i の
    # ラウンド数, i=0..bs-1) から位置別の条件付き受理率を導出する。
    # 35 §3c は「k をまたいだ別 run の差分」でしか位置別受理率を出せなかったが、
    # このヒストグラムなら 1 回の run の中で直接取れる。
    #   rounds_from_hist = Σ h[i]  (footer の rounds と検算。ズレても止めず
    #                                 両方を TSV に残して呼び出し側が判別する)
    #   a_i = P(位置 i を受理 | 位置 i-1 まで受理された)
    #       = (Σ_{j>=i} h[j]) / (Σ_{j>=i-1} h[j])      for i = 1..bs-1
    hist_str = row.get("accepted_hist")
    if hist_str:
        try:
            hist = [int(x) for x in hist_str.split(",") if x.strip() != ""]
        except ValueError:
            hist = []
        if hist:
            row["rounds_from_hist"] = sum(hist)
            for i in range(1, bs):
                numer = sum(hist[i:])
                denom = sum(hist[i - 1:])
                if denom > 0:
                    row[f"a{i}"] = numer / denom


def format_dry_run(cmd: list) -> str:
    return " ".join(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="scratch/gemma4-qat.gturbo")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS.keys()),
                        choices=list(TASKS.keys()),
                        help="測るタスク (既定: prose math image の 3 本全部)")
    parser.add_argument("--bs", default="0,2,3,4,6,8",
                        help="ブロック幅のカンマ区切り列 (既定 0,2,3,4,6,8。0=投機オフ)")
    parser.add_argument("--slots", type=int, default=32,
                        help="--expert-cache-slots に渡す値 (既定 32)")
    parser.add_argument("--max-new", type=int, default=192,
                        help="--max-new に渡す値 (既定 192)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="--seed に渡す値 (既定 1234)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="--temperature に渡す値 (既定 1.0)")
    parser.add_argument("--out", default="bench/logs/36_cd_sweep.tsv")
    parser.add_argument("--dry-run", action="store_true",
                        help="起動せず、組み立てたコマンド列を順に印字して終了する")
    args = parser.parse_args()

    try:
        bs_list = [int(x) for x in args.bs.split(",") if x.strip() != ""]
    except ValueError:
        raise SystemExit(f"--bs の値が不正: {args.bs!r}")

    # 各タスクについて、bs 列の末尾にもう一度 bs=0 を足す (熱ドリフト検定)。
    plan = []  # [(task, bs, slot), ...]  slot は "head_base" / "tail_base" / "mid"
    for task in args.tasks:
        for i, bs in enumerate(bs_list):
            slot = "head_base" if (bs == 0 and i == 0) else "mid"
            plan.append((task, bs, slot))
        plan.append((task, 0, "tail_base"))

    if args.dry_run:
        print(f"# dry-run: {len(plan)} runs planned "
              f"(tasks={args.tasks}, bs={bs_list}, COOL_S={COOL_S})")
        for task, bs, slot in plan:
            cmd = build_cmd(args, task, bs)
            print(format_dry_run(cmd))
        return 0

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(".jsonl")

    # a1..a{n} 列の本数は、この実行で登場しうる最大の bs から決まる
    # (bs 本のブロックは a_1..a_{bs-1} までしか意味を持たない)。
    max_bs = max(bs_list) if bs_list else 0
    a_cols = [f"a{i}" for i in range(1, max(max_bs - 1, 0) + 1)]

    tsv_cols = [
        "task", "bs", "slot", "temperature", "status", "new", "decode", "tok_s", "load", "ttft",
        "peak", "stop", "accept", "rounds", "draft", "verify", "tok_per_round",
        "decode_hit", "decode_experts", "decode_io",
        "ms_per_tok", "step_ms", "draft_per_round_ms", "drafts", "drafts_total",
        "d_ms", "outside_s", "outside_per_round_ms",
        "accepted_hist", "rounds_from_hist", *a_cols,
        "speedup", "process_seconds",
    ]

    rows = []
    head_base_ts = {}  # task -> tok_s (先頭 bs=0)
    tail_base_ts = {}  # task -> tok_s (末尾 bs=0、最後に見つかったもの)

    print(f"# mtp_cd_sweep: {len(plan)} runs planned "
          f"(tasks={args.tasks}, bs={bs_list}, COOL_S={COOL_S}s)")
    sys.stdout.flush()

    with out_path.open("w") as tsv_handle, jsonl_path.open("w") as jsonl_handle:
        tsv_handle.write("\t".join(tsv_cols) + "\n")
        tsv_handle.flush()

        for idx, (task, bs, slot) in enumerate(plan):
            row = run_once(args, task, bs, slot)
            add_derived(row)

            if row["status"] == "OK" and bs == 0:
                if slot == "head_base":
                    head_base_ts[task] = row.get("tok_s")
                if slot == "tail_base":
                    tail_base_ts[task] = row.get("tok_s")

            base = head_base_ts.get(task)
            if row["status"] == "OK" and base:
                row["speedup"] = row.get("tok_s", 0) / base if base else ""

            rows.append(row)

            jsonl_handle.write(json.dumps(
                {k: v for k, v in row.items() if k != "cmd"},
                ensure_ascii=False) + "\n")
            jsonl_handle.flush()

            tsv_handle.write("\t".join(
                str(row.get(c, "")) for c in tsv_cols) + "\n")
            tsv_handle.flush()

            if row["status"] == "OK":
                print(f"[{idx+1}/{len(plan)}] task={task} bs={bs} slot={slot} "
                      f"OK tok/s={row.get('tok_s', float('nan')):.3f} "
                      f"new={int(row.get('new', 0))} "
                      f"accept={row.get('accept', 'n/a')} "
                      f"rounds={int(row.get('rounds', 0)) if row.get('rounds') is not None else 'n/a'} "
                      f"d_ms={row.get('d_ms', 'n/a')} "
                      f"outside/round_ms={row.get('outside_per_round_ms', 'n/a')} "
                      f"stop={row.get('stop', '')}")
            else:
                print(f"[{idx+1}/{len(plan)}] task={task} bs={bs} slot={slot} "
                      f"FAIL {row.get('error', '')[:200]!r}")
            sys.stdout.flush()

            is_last = idx == len(plan) - 1
            if not is_last:
                print(f"# cooldown {COOL_S}s", flush=True)
                time.sleep(COOL_S)

    print(f"\n# wrote {args.out}")
    print(f"# wrote {jsonl_path.relative_to(ROOT)}")

    # タスクごとのまとめ表
    for task in args.tasks:
        print(f"\n=== {task} ===")
        print("bs\tt/s\tbase比\t受理長\trounds\tstep_ms\td_ms\toutside/round_ms\tms_per_tok\tstop")
        for row in rows:
            if row["task"] != task or row["status"] != "OK":
                continue

            def fmt(key, spec="{:.3f}"):
                v = row.get(key)
                return spec.format(v) if isinstance(v, (int, float)) else "n/a"

            print(f"{row['bs']}\t{fmt('tok_s')}\t{fmt('speedup')}\t"
                  f"{fmt('accept')}\t{fmt('rounds', '{:.0f}')}\t"
                  f"{fmt('step_ms', '{:.2f}')}\t{fmt('d_ms', '{:.2f}')}\t"
                  f"{fmt('outside_per_round_ms', '{:.2f}')}\t"
                  f"{fmt('ms_per_tok', '{:.2f}')}\t{row.get('stop', '')}")

    # 熱ドリフト行
    print("\n=== 熱ドリフト (先頭 base vs 末尾 base) ===")
    print("task\thead t/s\ttail t/s\t差(%)")
    for task in args.tasks:
        h = head_base_ts.get(task)
        t = tail_base_ts.get(task)
        if h and t:
            diff_pct = (t - h) / h * 100.0
            print(f"{task}\t{h:.3f}\t{t:.3f}\t{diff_pct:+.2f}%")
        else:
            print(f"{task}\t{h if h else 'n/a'}\t{t if t else 'n/a'}\tn/a")

    return 0


if __name__ == "__main__":
    sys.exit(main())
