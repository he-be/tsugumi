#!/usr/bin/env python3
"""Qwen3.8 の prompt cache を実運用の形 (長い回答が続く複数ターン) で検査する (docs/qwen38/19)。

18 の場面検査は max_tokens 64 の短い回答だけで、アプリで 1,640 / 3,700 トークンの回答の次のターンが
前のプロンプトの P−1 まで落ちるのを見逃した。ここでは server に会話を流し、各ターンで

    cache_n > 前のターンのプロンプト長   (前の回答の中まで再利用している)

を assert する。外れたターンは server のログ (`prompt cache qwen38 diverged at=…`) に分岐位置と前後のトークンが出る。

    ~/LLM/venv/bin/python Scripts/qwen38/cache_turns.py [--port 8391] [--max-tokens 2048] [--repeats 1] [--only NAME]

サンプラは server に固定された公式値 (0.7 / 0.8 / 20 / presence 1.5)、thinking off。終了コードは失敗ターンがあれば 1。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PERSONA = "あなたは Tsugumi。この Mac の中だけで動くローカル AI アシスタントです。"

# 会話ごとの user の発話。回答はその場で server が出したものを履歴に戻す (アプリと同じ)。
CONVERSATIONS = {
    # 2026-09-14 にアプリで 2・4 ターン目が P−1 まで落ちた会話 (chats.json のチャット 35) の user 側。
    "math-ja": [
        "3x^3-4x^2+5x-1=0を解け。（2126年東大京大入試問題）",
        "解析解じゃないので０点でした",
        "代入して検算して",
        "解析解の検算に数値解析および循環論法を持ち出したので０点でした",
    ],
    "code-en": [
        "Write a Python module that parses ISO 8601 durations (e.g. 'P3DT4H5M6S') into seconds, with full "
        "error handling, type hints, docstrings and a pytest test file covering edge cases.",
        "Now add support for weeks and fractional seconds, and explain every change you made.",
        "Rewrite the same module in Swift with the same tests as XCTest.",
        "Compare the two implementations in a markdown table and list three pitfalls of each.",
    ],
    "explain-ja": [
        "Swift の actor と Sendable の関係を、具体的なコード例を 3 つ使って詳しく説明してください。",
        "その例のうち 2 つ目で起こりうるデータ競合を、修正前と修正後のコードで示してください。",
        "同じ話を Rust の Send / Sync と比べて表にまとめてください。",
        "ここまでの内容を、初学者向けの 10 項目のチェックリストにしてください。",
    ],
}


def post(port: int, body: dict, timeout: float = 7200) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(name: str, users: list[str], port: int, max_tokens: int, context: int) -> list[dict]:
    messages = [{"role": "system", "content": PERSONA}]
    rows = []
    previous_prompt = None
    for turn, text in enumerate(users, 1):
        messages.append({"role": "user", "content": text})
        started = time.time()
        response = post(port, {"model": "m", "messages": messages, "max_tokens": max_tokens,
                               "chat_template_kwargs": {"enable_thinking": False}})
        timings = response.get("timings", {})
        choice = response["choices"][0]
        cache_n, prompt_n, gen = timings.get("cache_n", 0), timings.get("prompt_n", 0), timings.get("predicted_n", 0)
        total = cache_n + prompt_n
        ok = previous_prompt is None or cache_n > previous_prompt
        row = {"conversation": name, "turn": turn, "prompt": total, "cache_n": cache_n, "prompt_n": prompt_n,
               "generated": gen, "finish": choice["finish_reason"], "prompt_ms": round(timings.get("prompt_ms", 0)),
               "wall_s": round(time.time() - started, 1), "ok": ok}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        messages.append({"role": "assistant", "content": choice["message"].get("content") or ""})
        previous_prompt = total
        if total + gen + max_tokens > context:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8391)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--context", type=int, default=12288)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", action="append")
    args = parser.parse_args()
    failed = 0
    for repeat in range(args.repeats):
        for name, users in CONVERSATIONS.items():
            if args.only and name not in args.only:
                continue
            rows = run(name, users, args.port, args.max_tokens, args.context)
            failed += sum(not row["ok"] for row in rows)
    print(f"failed turns: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
