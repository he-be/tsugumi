#!/usr/bin/env python3
"""How well the PixelML KVA projector predicts our runner's late-layer inputs (docs/qwen38/30).

Reads a `TsugumiKernelCheck --qwen38-llkv-dump` directory (the boundary residual [T][4][2560] and layers 24...'s input
projections: whole = `exact-*`, filled from the copies' mean = `mean-*`, from the layer's own hc mix = `hcmix-*`) and
`projector.safetensors`. The engine code is not public; the reading used here is the one the dumps support
(docs/qwen38/30 §2):
  - the head's input is the boundary residual as it is (copy-major [4][2560] flattened, no norm);
  - its base is the layer's own projection of the copies' mean (`mean-*`, the "block-average" w0), also un-normed;
  - the GDN head's v / b / a outputs are in HF's grouped value-head order (key group g, r = 0..2 → HF head 3g + r),
    llama.cpp's GGUF puts them tiled (16r + g), so they are moved; the q part of the qkv head is 0;
  - the FA head is up(act(down(x))), the activation is not recorded: each of none / relu / gelu / silu is scored.
Printed per layer and target: the per-row cosine against the exact values of the base, `hcmix`, base + head, and
a * base + b * head with a, b least squares over all rows (how far the head could go at any scale).

    ~/LLM/venv/bin/python Scripts/qwen38/llkv_projector_fit.py scratch/qwen38/llkv28/dump512 \
        ~/LLM/Qwen3.8-KVA-Projector/projector.safetensors [layers]
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

HC, E = 4, 2560
HK, HV, DL = 16, 48, 128
HKV, D = 2, 256
GDN = [24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46]
FA = [27, 31, 35, 39, 43, 47]
# HF value head 3g + r (key group g) sits at GGUF head 16r + g.
HF_TO_GGUF_V = np.array([(h % 3) * HK + h // 3 for h in range(HV)])


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    header.pop("__metadata__", None)
    out = {}
    for name, v in header.items():
        a, b = v["data_offsets"]
        assert v["dtype"] == "BF16", v["dtype"]
        u16 = np.frombuffer(raw, dtype="<u2", count=(b - a) // 2, offset=8 + n + a)
        # bf16 and f32 share the sign and 8-bit exponent: the 7 mantissa bits go on top of f32's 23.
        out[name] = (u16.astype(np.uint32) << 16).view(np.float32).reshape(v["shape"]).astype(np.float64)
    return out


def rows(dump: Path, name: str, width: int) -> np.ndarray:
    return np.fromfile(dump / f"{name}.f32", dtype=np.float32).reshape(-1, width).astype(np.float64)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30
    return float((num / den).mean())


def to_gguf_v(d: np.ndarray, head_dim: int) -> np.ndarray:
    v = d.reshape(len(d), HV, head_dim)
    out = np.empty_like(v)
    out[:, HF_TO_GGUF_V] = v
    return out.reshape(len(d), -1)


ACTS = {
    "none": lambda h: h,
    "relu": lambda h: np.maximum(h, 0),
    "gelu": lambda h: 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.071355 * h ** 3))),
    "silu": lambda h: h / (1 + np.exp(-np.clip(h, -60, 60))),
}


def line(label: str, t: np.ndarray, base: np.ndarray, hcmix: np.ndarray, head: np.ndarray) -> str:
    A = np.stack([base.ravel(), head.ravel()], 1)
    (a, b), *_ = np.linalg.lstsq(A, t.ravel(), rcond=None)
    return (f"{label:16s} base {cos(t, base):.3f}  hcmix {cos(t, hcmix):.3f}  base+head {cos(t, base + head):.3f}"
            f"  fit {cos(t, a * base + b * head):.3f} (a {a:6.2f} b {b:5.2f})")


def main() -> None:
    dump, proj = Path(sys.argv[1]), Path(sys.argv[2])
    layers = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else sorted(GDN + FA)
    w = load_safetensors(proj)
    x = rows(dump, "exact-boundary", HC * E)
    print(f"T {len(x)}")
    for il in layers:
        load = lambda tag, part, width: rows(dump, f"{tag}-L{il}-{part}", width)
        if il in GDN:
            for part, width in [("qkv", 2 * HK * DL + HV * DL), ("b", HV), ("a", HV)]:
                t, base, hcm = (load(tag, part, width) for tag in ("exact", "mean", "hcmix"))
                head = (x @ w[f"gdn_heads.{il}.{part}.lora_down.weight"].T) @ w[f"gdn_heads.{il}.{part}.lora_up.weight"].T
                if part == "qkv":
                    k, v = slice(HK * DL, 2 * HK * DL), slice(2 * HK * DL, width)
                    print(line(f"L{il} k", t[:, k], base[:, k], hcm[:, k], head[:, k]))
                    print(line(f"L{il} v", t[:, v], base[:, v], hcm[:, v], to_gguf_v(head[:, v], DL)))
                else:
                    print(line(f"L{il} {part}", t, base, hcm, to_gguf_v(head, 1)))
        else:
            for part, width in [("k", HKV * D), ("v", HKV * D), ("ik", 128)]:
                t, base, hcm = (load(tag, part, width) for tag in ("exact", "mean", "hcmix"))
                h = x @ w[f"fa_heads.{il}.{part}.down.weight"].T
                for name, act in ACTS.items():
                    head = act(h) @ w[f"fa_heads.{il}.{part}.up.weight"].T
                    print(line(f"L{il} {part} {name}", t, base, hcm, head))


if __name__ == "__main__":
    main()
