#!/usr/bin/env python3
"""Qwen3.8 の Web ツールループの要求を上流の描画で書く (docs/qwen38/20)。

`Tests/TsugumiApp/Core/Fixtures/qwen38-tool-loop/spec.json` (Swift のテスト `Qwen38ToolLoopPromptTests` が
アプリの宣言・システムプロンプト・会話から書いたもの) を読み、各要求をアプリのターンから OpenAI 形式の
メッセージに組み直して、チェックポイントの chat_template.jinja を HF と同じ jinja (trim_blocks / lstrip_blocks、
tojson = json.dumps(ensure_ascii=False)) で描き、`<label>.txt` を同じディレクトリに書く。

    ~/LLM/venv/bin/python Scripts/qwen38/tool_loop_fixture.py

メッセージの組み方は Swift 側 (`RealInferenceSession.validatedChatRequest`) を写したものではなく、
HF にツールループを渡すクライアントの普通の形 (arguments は JSON をパースした dict、reasoning_content は空なら無し)。
"""
from __future__ import annotations

import json
from pathlib import Path

import jinja2

TOK_DIR = Path.home() / "LLM/Qwen3.8-Flash-Next-tokenizer"
FIXTURE = Path(__file__).resolve().parents[2] / "Tests/TsugumiApp/Core/Fixtures/qwen38-tool-loop"


def render(messages, tools):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, **_: json.dumps(v, ensure_ascii=False)

    def raise_exception(msg):
        raise jinja2.TemplateError(msg)

    template = env.from_string((TOK_DIR / "chat_template.jinja").read_text())
    return template.render(messages=messages, tools=tools, add_generation_prompt=True, enable_thinking=False,
                           raise_exception=raise_exception)


def message(turn):
    out = {"role": turn["role"], "content": turn["text"]}
    if turn.get("reasoningText"):
        out["reasoning_content"] = turn["reasoningText"]
    if turn.get("toolCalls"):
        out["tool_calls"] = [{"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": json.loads(c["argumentsJSON"])}}
                             for c in turn["toolCalls"]]
    if turn.get("toolCallID"):
        out["tool_call_id"] = turn["toolCallID"]
    if turn.get("toolName"):
        out["name"] = turn["toolName"]
    return out


def main():
    spec = json.loads((FIXTURE / "spec.json").read_text())
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": json.loads(t["parametersJSON"])}}
             for t in spec["tools"]]
    for step in spec["steps"]:
        messages = ([{"role": "system", "content": spec["system"]}]
                    + [message(t) for t in step["history"]]
                    + [{"role": "user", "content": step["prompt"]}]
                    + [message(t) for t in step["continuation"]])
        text = render(messages, tools)
        (FIXTURE / f"{step['label']}.txt").write_text(text)
        print(f"{step['label']}: {len(text)} chars")


if __name__ == "__main__":
    main()
