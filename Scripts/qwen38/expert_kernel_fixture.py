#!/usr/bin/env python3
"""DS4-IQ2 GGUF の routed expert を 1 層ぶん切り出し、Metal カーネルの突き合わせ用 fixture を書く。

Q2 検証の 1 本目 (IQ2_XXS gate/up + Q2_K down の expert 行列積を M3 Pro で affine int4 と比べる) の
正解側。逆量子化は llama.cpp の gguf-py (`quants.py`) に任せる。CPU で数 MB を読むだけで、
llama.cpp の推論は回さない。

    ~/LLM/venv/bin/python Scripts/qwen38/expert_kernel_fixture.py \\
        --gguf ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf \\
        --layer 24 --out scratch/qwen38/expert-fixture-l24

出力 (すべてリトルエンディアン):
    meta.json        形状、expert 番号、blob 内オフセット
    blobs.bin        expert ごとに gate (IQ2_XXS) | up (IQ2_XXS) | down (Q2_K、入力 768 列に pad 済み) を連結
    x.f16            入力 hidden [D]
    residual.f16     残差 [D] (カーネルは y = residual + Σ w_i down_i(act_i) を返す)
    weights.f16      routing weight [k]
    ref_acts.f32     silu(gate_i x) * (up_i x) [k, F]
    ref_y.f32        y [D]
    ref_gate.f32 / ref_up.f32   gate_i x / up_i x [k, F] (カーネルの切り分け用)

expert は x を実際の router (`ffn_gate_inp`) に通した top-k。重みは softmax → top-k → 和 1 に正規化。
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    r = GGUFReader(args.gguf, "r")
    tensors = {t.name: t for t in r.tensors}
    pre = f"blk.{args.layer}."
    gate_t = tensors[pre + "ffn_gate_exps.weight"]
    up_t = tensors[pre + "ffn_up_exps.weight"]
    down_t = tensors[pre + "ffn_down_exps.weight"]
    router_t = tensors[pre + "ffn_gate_inp.weight"]

    d = int(gate_t.shape[0])          # 2560 (入力列)
    f = int(gate_t.shape[1])          # 640 (行)
    down_in = int(down_t.shape[0])    # 768 (pad 後の入力列)
    field = r.fields.get("ds4.qwen4.down.logical_input")
    f_logical = int(field.contents()) if field is not None else f
    assert f_logical == f, (f_logical, f)
    assert gate_t.tensor_type.name == "IQ2_XXS" and up_t.tensor_type.name == "IQ2_XXS"
    assert down_t.tensor_type.name == "Q2_K"

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal(d).astype(np.float16)
    residual = (0.1 * rng.standard_normal(d)).astype(np.float16)

    router = np.asarray(router_t.data, dtype=np.float32)  # [512, D]
    logits = router @ x.astype(np.float32)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    experts = np.argsort(-probs)[: args.top_k]
    weights = probs[experts] / probs[experts].sum()
    weights16 = weights.astype(np.float16)

    gate_raw = np.asarray(gate_t.data)   # [512, F, bytes/row]
    up_raw = np.asarray(up_t.data)
    down_raw = np.asarray(down_t.data)   # [512, D, bytes/row]

    xf = x.astype(np.float64)
    blobs = bytearray()
    offsets = []
    ref_gate = np.zeros((args.top_k, f), np.float64)
    ref_up = np.zeros((args.top_k, f), np.float64)
    ref_acts = np.zeros((args.top_k, f), np.float64)
    y = residual.astype(np.float64).copy()
    for i, e in enumerate(experts):
        g_bytes = np.ascontiguousarray(gate_raw[e]).tobytes()
        u_bytes = np.ascontiguousarray(up_raw[e]).tobytes()
        dn_bytes = np.ascontiguousarray(down_raw[e]).tobytes()
        base = len(blobs)
        offsets.append({"expert": int(e), "base": base,
                        "gate": 0, "up": len(g_bytes), "down": len(g_bytes) + len(u_bytes),
                        "stride": len(g_bytes) + len(u_bytes) + len(dn_bytes)})
        blobs += g_bytes + u_bytes + dn_bytes

        gw = dequantize(gate_raw[e], gate_t.tensor_type).reshape(f, d).astype(np.float64)
        uw = dequantize(up_raw[e], up_t.tensor_type).reshape(f, d).astype(np.float64)
        dw = dequantize(down_raw[e], down_t.tensor_type).reshape(d, down_in).astype(np.float64)
        g = gw @ xf
        u = uw @ xf
        a = g / (1.0 + np.exp(-g)) * u
        ref_gate[i], ref_up[i], ref_acts[i] = g, u, a
        a_pad = np.zeros(down_in)
        a_pad[:f] = a.astype(np.float32)  # カーネルは float32 の act を読む
        y += float(weights16[i]) * (dw @ a_pad)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "blobs.bin").write_bytes(bytes(blobs))
    x.tofile(args.out / "x.f16")
    residual.tofile(args.out / "residual.f16")
    weights16.tofile(args.out / "weights.f16")
    ref_acts.astype(np.float32).tofile(args.out / "ref_acts.f32")
    ref_gate.astype(np.float32).tofile(args.out / "ref_gate.f32")
    ref_up.astype(np.float32).tofile(args.out / "ref_up.f32")
    y.astype(np.float32).tofile(args.out / "ref_y.f32")
    meta = {
        "gguf": str(args.gguf), "layer": args.layer, "D": d, "F": f, "down_in": down_in,
        "top_k": args.top_k, "experts": [int(e) for e in experts],
        "weights": [float(w) for w in weights16],
        "gate_bytes": gate_raw.shape[1] * gate_raw.shape[2],
        "down_bytes": down_raw.shape[1] * down_raw.shape[2],
        "offsets": offsets,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"layer {args.layer}: experts {meta['experts']}")
    print(f"expert stride {offsets[0]['stride']} B, |y - residual| = {np.linalg.norm(y - residual):.4f}, "
          f"acts rms = {np.sqrt((ref_acts ** 2).mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
