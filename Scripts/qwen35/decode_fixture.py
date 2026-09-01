#!/usr/bin/env python3
"""reference_forward.py の生成ログを `--qwen-decode-fixture` の JSON に変える。

Phase 3 の出口条件は「固定プロンプトから N トークン、CPU float32 参照と完全一致」
(`docs/qwen35moe/04-PHASES.md`)。参照の側は 47.5 s/トークンなので、生成は 1 度だけ
回してログに残し、以後はこの JSON を比較の相手にする。

    python3 Scripts/qwen35/decode_fixture.py \\
        --log scratch/qwen35/ref-64-oq4e-g64.log \\
        --tokenizer ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 \\
        --text "日本の首都はどこですか。一文で答えてください。" \\
        --out scratch/qwen35/decode-fixture-64.json

プロンプトはログに切り詰めた形でしか残らない (先頭 16 個 + "…") ので、
**同じ文をトークナイザに通し直して復元し、ログに載っている先頭 16 個と
突き合わせてから**書く。食い違ったらそれは別のプロンプトのログである。
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

PROMPT_LINE = re.compile(r"入力 (\d+) トークン: (\[[^\]]*\])")
STEP_LINE = re.compile(r"\[\s*\d+\]\s+token=\s*(-?\d+)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="reference_forward.py の出力")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json のあるディレクトリ")
    ap.add_argument("--text", required=True, help="ログを作ったときの --text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 ならログにある全部")
    args = ap.parse_args()

    log = Path(args.log).read_text(encoding="utf-8")
    header = PROMPT_LINE.search(log)
    if header is None:
        raise SystemExit(f"{args.log} に「入力 N トークン」の行が無い")
    prompt_length = int(header.group(1))
    logged_head = ast.literal_eval(header.group(2))

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(args.tokenizer).expanduser() / "tokenizer.json"))
    prompt = tok.encode(
        f"<|im_start|>user\n{args.text}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False).ids

    if len(prompt) != prompt_length:
        raise SystemExit(f"再現したプロンプトは {len(prompt)} トークン、"
                         f"ログは {prompt_length} — --text が違う")
    if prompt[:len(logged_head)] != logged_head:
        raise SystemExit("ログの先頭トークンと一致しない — 別のプロンプトのログ")

    expected = [int(m.group(1)) for m in STEP_LINE.finditer(log)]
    if not expected:
        raise SystemExit(f"{args.log} に生成トークンの行が無い")
    if args.max_tokens:
        expected = expected[:args.max_tokens]

    Path(args.out).write_text(
        json.dumps({"prompt": prompt, "expected": expected}, indent=1) + "\n",
        encoding="utf-8")
    print(f"{args.out}: プロンプト {len(prompt)} / 期待 {len(expected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
