#!/usr/bin/env python3
"""`ggml_iq.metal` の表が llama.cpp の `ggml-common.h` と一致するか確かめる (docs/qwen38-27b/03)。

表は ggml-common.h から生成して写した。llama.cpp を更新したときや表を触ったときに回す。

    python3 Scripts/qwen38_27b/check_ggml_iq_tables.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

COMMON = Path.home() / "LLM/llama.cpp/ggml/src/ggml-common.h"
METAL = Path(__file__).resolve().parents[2] / "Sources/Tsugumi/Metal/Quant/ggml_iq.metal"
TABLES = ["kmask_iq2xs", "ksigns_iq2xs", "iq2xxs_grid", "iq2xs_grid", "iq2s_grid",
          "iq3xxs_grid", "iq3s_grid", "kvalues_iq4nl", "iq1s_grid"]


def values(text: str) -> list[int]:
    text = re.sub(r"//.*", "", text)
    return [int(v, 0) for v in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", text)]


def main() -> int:
    common = COMMON.read_text()
    metal = METAL.read_text()
    ok = True
    for name in TABLES:
        c = re.search(r"GGML_TABLE_BEGIN\(\w+, %s, \w+\)(.*?)GGML_TABLE_END" % name, common, re.S)
        m = re.search(r"ggml_%s\[\d+\] = \{(.*?)\};" % name, metal, re.S)
        a, b = values(c.group(1)), values(m.group(1))
        same = a == b
        ok = ok and same
        print(f"{name:14s} {len(a):5d} {'一致' if same else '不一致'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
