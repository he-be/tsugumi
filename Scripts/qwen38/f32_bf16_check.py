#!/usr/bin/env python3
"""DS4-IQ2 GGUF の F32 テンソルが BF16 でビット単位に表せるかを数える (docs/qwen38/15 §2 W-4、検定だけ)。

BF16 は符号 1・指数 8・仮数 7 ビットで、float32 と指数の幅が同じ。float32 の値 v が BF16 で損失なく表せるのは、
仮数 23 ビットの下位 16 ビットが 0 のとき (0・非正規化数・inf も同じ条件)。上流が bf16 から F32 に広げただけなら、
全要素でこれが成り立つ。数えるだけで、何も書かない。

    ~/LLM/venv/bin/python Scripts/qwen38/f32_bf16_check.py
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "LLM/llama.cpp/gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402

DEFAULT_SRC = Path.home() / "LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf"
F_NOCACHE = 48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = ap.parse_args()
    r = GGUFReader(args.src, "r")
    fd = os.open(args.src, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    # テンソル名の層番号を落とした族ごとに集計する (blk.N.ssm_alpha.weight -> blk.*.ssm_alpha.weight)。
    fam = collections.OrderedDict()
    for t in r.tensors:
        if t.tensor_type != GGMLQuantizationType.F32:
            continue
        buf = bytearray(t.n_bytes)
        mv = memoryview(buf)
        got = 0
        while got < t.n_bytes:
            k = os.preadv(fd, [mv[got:]], t.data_offset + got)
            assert k > 0
            got += k
        bits = np.frombuffer(buf, np.uint32)
        low = (bits & 0xFFFF) != 0
        v = bits.view(np.float32)
        key = re.sub(r"^blk\.\d+\.", "blk.*.", t.name)
        f = fam.setdefault(key, dict(tensors=0, n=0, bad=0, bytes=0, maxabs=0.0, nonfinite=0, sub=0))
        f["tensors"] += 1
        f["n"] += bits.size
        f["bad"] += int(low.sum())
        f["bytes"] += t.n_bytes
        f["maxabs"] = max(f["maxabs"], float(np.abs(v[np.isfinite(v)]).max()) if np.isfinite(v).any() else 0.0)
        f["nonfinite"] += int((~np.isfinite(v)).sum())
        f["sub"] += int(((bits & 0x7F800000) == 0).sum() - (bits & 0x7FFFFFFF == 0).sum())
    total = dict(n=0, bad=0, bytes=0)
    print("| 族 | テンソル数 | 要素 | MiB | 下位 16 ビットが 0 でない要素 | max |v| | 非有限 | 非正規化 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, f in fam.items():
        print(f"| `{key}` | {f['tensors']} | {f['n']:,} | {f['bytes'] / 2**20:.1f} | {f['bad']:,} ({f['bad'] / f['n']:.4f}) | "
              f"{f['maxabs']:.4g} | {f['nonfinite']} | {f['sub']} |")
        for k in total:
            total[k] += f[k]
    print(f"| 計 | {sum(f['tensors'] for f in fam.values())} | {total['n']:,} | {total['bytes'] / 2**20:.1f} | "
          f"{total['bad']:,} ({total['bad'] / total['n']:.4f}) | | | |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
