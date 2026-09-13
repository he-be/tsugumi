#!/usr/bin/env python3
"""Qwen3.8 の生成・MTP 実測に使うプロンプトをトークン列にする (docs/qwen38/10)。

上流の chat_template.jinja を jinja2 でそのまま描き、thinking off (`enable_thinking=False`) で
`<think>\n\n</think>\n\n` まで入れて tokenizer.json で符号化する。運用点 (メモリ `qwen38-operating-point`) に
合わせて英語・即答・ツール呼び出しを含む。

    ~/LLM/venv/bin/python Scripts/qwen38/chat_prompts.py build scratch/qwen38/prompts
    ~/LLM/venv/bin/python Scripts/qwen38/chat_prompts.py decode <ids ファイル>

build はディレクトリに `<name>.tokens` (カンマ区切り) を書く。long は Swift ソースを `<file path=…>` で包み、
全体が `--long-tokens` (既定 11,700) に収まるように末尾のファイルを切る。
decode は `--qwen38-generate` の出力 (1 行目 `ids: a,b,c`) か、カンマ区切りの id を読んで文字列を出す。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jinja2
from tokenizers import Tokenizer

TOK_DIR = Path.home() / "LLM/Qwen3.8-Flash-Next-tokenizer"
REPO = Path(__file__).resolve().parents[2]

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the repository and return its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "start_line": {"type": "integer", "description": "First line to return (1-based)."},
            "end_line": {"type": "integer", "description": "Last line to return (inclusive)."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a shell command in the repository root and return stdout and stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command line to run."}},
            "required": ["command"]}}},
]

SHORT = {
    "code": [{"role": "user", "content":
        "Write a Python function `parse_duration(s: str) -> int` that converts an ISO 8601 duration such as "
        "'P3DT4H5M6S' into total seconds. Support days, hours, minutes and seconds, raise ValueError on bad "
        "input, and add a short docstring plus three doctest examples."}],
    "tool": [{"role": "system", "content": "You are a coding agent working in a Swift repository. Use the tools to inspect files before answering."},
             {"role": "user", "content":
        "The build fails with `error: cannot find 'allocateBatch' in scope` in "
        "Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift. Find out why."}],
    "explain": [{"role": "user", "content":
        "Explain the difference between a mutex, a semaphore and a condition variable. Give one concrete "
        "bug each one prevents, as a short list, then a two-sentence summary."}],
}

# The files are read at the repository state the prompt is built from (docs/qwen38/10 records the commit).
LONG = {
    "long": ([
        "Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift",
        "Sources/TsugumiKernelCheck/Qwen38DecodeCheck.swift",
        "Sources/Tsugumi/Runtime/Qwen38/Qwen38GDNChunk.swift",
        "Sources/Tsugumi/Metal/MoE/moe_ggml.metal",
    ], "Above are the source files of a Metal inference runner. In `Qwen38Runner.forwardBody`, the batch scratch "
       "is reallocated depending on the batch size. Explain when this happens, what state is preserved across the "
       "reallocation, and what could go wrong if a caller keeps the returned logits pointer. Then propose a minimal "
       "code change that makes the pointer misuse impossible, as a unified diff."),
    "long2": ([
        "Sources/Tsugumi/Metal/Qwen/qwen38.metal",
        "Sources/Tsugumi/Infrastructure/ModelIO/GGUFFile.swift",
        "Sources/Tsugumi/Kernels/Qwen38/GGMLDenseGEMV.swift",
        "Sources/Tsugumi/Infrastructure/Metal/MetalContext.swift",
    ], "Above are Metal kernels and the Swift code that loads model files and compiles shaders. List every kernel "
       "that reads a sum over a long axis with SIMD lanes, and for one of them explain step by step how the lanes "
       "and `simd_sum` produce the result. Then write a unit test in Swift (XCTest) that checks `q38_group_rms_scale` "
       "against a plain CPU loop for a random input."),
    "long3": ([
        "Sources/TsugumiKernelCheck/Q2GemmBench.swift",
        "Sources/TsugumiKernelCheck/Qwen38SelectCheck.swift",
        "Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift",
    ], "You are reviewing this code before a release. Find three concrete bugs or risky patterns (with file and "
       "function names), explain the failure each could cause, and give a fix for each as a short code snippet. "
       "Finish with one sentence on what to test first."),
}


def render(messages, tools=None):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, **_: json.dumps(v, ensure_ascii=False)

    def raise_exception(msg):
        raise jinja2.TemplateError(msg)

    tmpl = env.from_string((TOK_DIR / "chat_template.jinja").read_text())
    return tmpl.render(messages=messages, tools=tools, add_generation_prompt=True, enable_thinking=False,
                       raise_exception=raise_exception)


def build(out: Path, long_tokens: int):
    tok = Tokenizer.from_file(str(TOK_DIR / "tokenizer.json"))
    out.mkdir(parents=True, exist_ok=True)

    def write(name, text):
        ids = tok.encode(text, add_special_tokens=False).ids
        (out / f"{name}.tokens").write_text(",".join(map(str, ids)) + "\n")
        (out / f"{name}.txt").write_text(text)
        print(f"{name}: {len(ids)} tokens")
        return ids

    for name, messages in SHORT.items():
        write(name, render(messages, TOOLS if name == "tool" else None))

    for name, (files, question) in LONG.items():
        write_long(tok, write, name, files, question, long_tokens)


def write_long(tok, write, name, files, question, long_tokens):
    """Files in order, the last one cut so the whole prompt fits."""
    q_ids = len(tok.encode(render([{"role": "user", "content": question}]), add_special_tokens=False).ids)
    budget = long_tokens - q_ids - 16
    parts = []
    used = 0
    for rel in files:
        body = (REPO / rel).read_text()
        block = f'<file path="{rel}">\n{body}\n</file>\n'
        n = len(tok.encode(block, add_special_tokens=False).ids)
        if used + n > budget:
            lines = body.split("\n")
            lo, hi = 0, len(lines)
            while lo < hi:  # most lines that fit
                mid = (lo + hi + 1) // 2
                b = f'<file path="{rel}" truncated="true">\n' + "\n".join(lines[:mid]) + "\n</file>\n"
                if used + len(tok.encode(b, add_special_tokens=False).ids) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            parts.append(f'<file path="{rel}" truncated="true">\n' + "\n".join(lines[:lo]) + "\n</file>\n")
            break
        parts.append(block)
        used += n
    write(name, render([{"role": "user", "content": "".join(parts) + "\n" + question}]))


def decode(path: Path):
    tok = Tokenizer.from_file(str(TOK_DIR / "tokenizer.json"))
    text = path.read_text()
    line = next((l for l in text.splitlines() if l.startswith("ids:")), text)
    ids = [int(t) for t in line.removeprefix("ids:").split(",") if t.strip()]
    sys.stdout.write(tok.decode(ids, skip_special_tokens=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out", type=Path)
    b.add_argument("--long-tokens", type=int, default=11_700)
    d = sub.add_parser("decode")
    d.add_argument("path", type=Path)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.out, args.long_tokens)
    else:
        decode(args.path)


if __name__ == "__main__":
    main()
