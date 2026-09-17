#!/usr/bin/env python3
"""Qwen3.8-27B GGUF の dense テンソルで、Metal GEMV の突き合わせ用 fixture を書く (docs/qwen38-27b/01 §3-1)。

指定した型ごとに、(行数, 列数) の形ごとに最初のテンソルを 1 本選び、乱数の入力 x [T, n] と
正解 y = x @ W.T [T, m] を書く。W は gguf-py (`quants.py`) で逆量子化し、積は float64 で取る。
行は 4,096 行ずつ逆量子化するので、LM head (248,320 行) でも常駐は数百 MB。

    ~/LLM/venv/bin/python Scripts/qwen38_27b/dense_kernel_fixture.py --types IQ3_S \\
        --out scratch/qwen38_27b/dense-fixture

出力 (リトルエンディアン):
    meta.json            {"gguf": ..., "cases": [{"name", "type", "m", "n", "tokens", "x", "y"}]}
    <i>.x.f32            [T, n]
    <i>.y.f64            [T, m]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "LLM/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

DEFAULT_GGUF = Path.home() / "LLM/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
ROW_CHUNK = 4096


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--types", required=True, help="comma-separated GGML type names (IQ3_S,Q4_K,...)")
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    types = set(args.types.split(","))
    r = GGUFReader(args.gguf, "r")
    picked = {}
    for t in r.tensors:
        ty = t.tensor_type.name
        if ty not in types or len(t.shape) != 2:
            continue
        key = (ty, int(t.shape[1]), int(t.shape[0]))
        picked.setdefault(key, t)

    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"gguf": str(args.gguf), "cases": []}
    meta["cases"] = [c for c in meta["cases"] if c["type"] not in types]
    rng = np.random.default_rng(args.seed)
    for (ty, m, n), t in sorted(picked.items()):
        x = rng.uniform(-1, 1, size=(args.tokens, n)).astype(np.float32)
        y = np.empty((args.tokens, m), np.float64)
        data = t.data
        for r0 in range(0, m, ROW_CHUNK):
            r1 = min(m, r0 + ROW_CHUNK)
            w = dequantize(np.ascontiguousarray(data[r0:r1]), t.tensor_type).reshape(r1 - r0, n).astype(np.float64)
            y[:, r0:r1] = x.astype(np.float64) @ w.T
        stem = f"{ty}-{m}x{n}"
        x.tofile(args.out / f"{stem}.x.f32")
        y.tofile(args.out / f"{stem}.y.f64")
        meta["cases"].append({"name": t.name, "type": ty, "m": m, "n": n, "tokens": args.tokens,
                              "x": f"{stem}.x.f32", "y": f"{stem}.y.f64"})
        print(f"{ty:8s} {t.name:28s} [{m} x {n}]  max|y| {np.abs(y).max():.4g}", flush=True)
    meta_path.write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
