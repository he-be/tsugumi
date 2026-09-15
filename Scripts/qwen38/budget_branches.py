#!/usr/bin/env python3
"""ツール結果の末尾に足す行 (残り回数・文脈) だけを変えたプロンプトを分岐点ごとに書く (docs/qwen38/25)。

アプリの会話 (chats.json の 1 本、outputContinuationTurns) を、`Tests/.../qwen38-tool-loop/spec.json` の宣言と
システムプロンプトで上流の chat_template.jinja に描き、tokenizer.json で符号化する。分岐点 k は「モデルの呼び出し
k 回目の結果まで」を読んだ直後の生成開始位置。最後の結果の末尾だけを変える。`--qwen38-branch-probe` の manifest を書く。

    ~/LLM/venv/bin/python Scripts/qwen38/budget_branches.py OUT --chat 37 --points 2,5,6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tool_loop_fixture import FIXTURE, TOK_DIR, message, render  # noqa: E402

CHATS = Path.home() / "Library/Application Support/Tsugumi/chats.json"
LIMIT = "(ツール呼び出しは合計 6 回までで、上限に達しました。これ以上ツールは呼べません。ここまでの結果で答えます。)"


def rounds(used, limit=6):
    return f"ツール呼び出し: {limit} 回中 {used} 回使用、残り {limit - used} 回。"


def context(used, total=16384):
    return f"文脈: {total:,} トークン中、この結果の前までで {used:,} 使用、残り {max(total - used, 0):,}。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--chat", type=int, required=True)
    ap.add_argument("--points", default="2,5,6")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--sample-tokens", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((FIXTURE / "spec.json").read_text())
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": json.loads(t["parametersJSON"])}}
             for t in spec["tools"]]
    chat = json.loads(CHATS.read_text())["chats"][a.chat]
    turns = [dict(t) for t in chat["outputContinuationTurns"]]
    for t in turns:
        if t["role"] == "tool" and t["text"].endswith(LIMIT):
            t["text"] = t["text"][: -len(LIMIT)].rstrip("\n")
    tok = Tokenizer.from_file(str(TOK_DIR / "tokenizer.json"))

    def encode(conts):
        messages = ([{"role": "system", "content": spec["system"]},
                     {"role": "user", "content": chat["outputPromptText"]}] + [message(t) for t in conts])
        return tok.encode(render(messages, tools), add_special_tokens=False).ids

    # Tool results in order; the first is the app's Wikipedia lookup, not a model call.
    tool_ends = [i for i, t in enumerate(turns) if t["role"] == "tool"]
    calls = len(tool_ends) - 1
    variants, trunk = [], None
    for k in sorted(int(p) for p in a.points.split(",")):
        end = tool_ends[k]
        base = turns[: end + 1]
        used = len(encode(base))

        def with_note(note):
            conts = [dict(t) for t in base]
            if note:
                conts[-1]["text"] += "\n\n" + note
            return conts

        cases = {"none": None, "limit": LIMIT}
        if k < 6:
            cases[f"r{6 - k}"] = f"({rounds(k)})"
            cases[f"r{6 - k}-ctx-ample"] = f"({rounds(k)}{context(used)})"
            cases[f"r{6 - k}-ctx-tight"] = f"({rounds(k)}{context(15_900)})"
        for name, note in cases.items():
            ids = encode(with_note(note))
            label = f"k{k}-{name}"
            (out / f"{label}.tokens").write_text(",".join(map(str, ids)))
            variants.append({"label": label, "tokens": f"{label}.tokens"})
            print(f"{label}: {len(ids)} tokens")
        if k == max(int(p) for p in a.points.split(",")):
            trunk = f"k{k}-none.tokens"
    json.dump({"trunk": trunk, "samples": a.samples, "sampleTokens": a.sample_tokens, "variants": variants,
               "chat": a.chat, "calls": calls}, open(out / "manifest.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
