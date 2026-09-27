#!/usr/bin/env python3
"""Bonsai 2 27B の MLX 2-bit パックに、MTP 同梱 GGUF の MTP ヘッド (blk.64、15 本) を足す (docs/qwen38-27b/16 §14)。

MTP 同梱版 (`ProCreations/Ternary-Bonsai-2-27B-MTP`、Scripts/qwen38_27b/bonsai_mtp_fetch.py) の blk.64 は
Q8_0 (行列 7 本) と F32 (norm 8 本)。行列は Hadamard 回転なし (llama.cpp が逆変換するのは埋め込みの参照だけ)。
norm は GGUF もパックも掛ける形 (1 + w を畳んだ値) で、本体の層どうしで値が一致する。

  # MBP (標準ライブラリだけ): GGUF から blk.64 を抜き出す (451 MB)
  python3 bench/m6/bonsai_mtp_pack.py extract \
      ~/LLM/Ternary-Bonsai-2-27B-MTP/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf scratch/bonsai27b/mtp-blk64.bin
  # M6 (mlx-venv): 8-bit / group 32 の affine に量子化し、元のパックをリンクで指す新しいディレクトリを作る
  ~/LLM/ref/mlx-venv/bin/python bonsai_mtp_pack.py pack mtp-blk64.bin \
      ~/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit ~/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit-mtp

新しいディレクトリは本体の 8.6 GB を複製しない: `base.safetensors` は元の model.safetensors へのリンクで、
`model.safetensors.index.json` が本体と `mtp.safetensors` の両方を指す (ishizuki の WeightStore は
model.safetensors が無いときに index を読む)。config.json だけ書き換える:
components.mtp true・text_config.mtp_num_hidden_layers 1・modules に MTP の線形 7 本 (block 0 = 回転なし)・
quantization に同じ 7 本の上書き (bits 8、group_size 32)。
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Scripts" / "qwen38_27b"))

Q8_0, F32 = 8, 0
BITS, GROUP = 8, 32

HEAD = {
    "nextn.eh_proj.weight": "language_model.mtp.fc.weight",
    "nextn.enorm.weight": "language_model.mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "language_model.mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "language_model.mtp.norm.weight",
}
BLOCK = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}


def pack_name(gguf_name: str, layer: int) -> str:
    tail = gguf_name[len(f"blk.{layer}."):]
    if tail in HEAD:
        return HEAD[tail]
    return f"language_model.mtp.layers.0.{BLOCK[tail]}"


def extract(gguf: str, out: str) -> None:
    from bonsai_mtp_fetch import read_header  # noqa: E402  (the GGUF header reader already in the repo)

    with open(gguf, "rb") as f:
        n = 1 << 24
        while True:
            try:
                header = read_header(f.read(n))
                break
            except Exception:
                n *= 2
                f.seek(0)
        layers = header["kv"]["qwen35.block_count"] - header["kv"]["qwen35.nextn_predict_layers"]
        chosen = [t for t in header["tensors"] if t["name"].startswith(f"blk.{layers}.")]
        assert len(chosen) == 15, [t["name"] for t in chosen]
        entries, blobs, at = [], [], 0
        for t in chosen:
            ne0, ne1 = (t["dims"] + [1])[:2]
            if t["type"] == Q8_0:
                size = ne0 * ne1 // 32 * 34
            elif t["type"] == F32:
                size = ne0 * ne1 * 4
            else:
                raise SystemExit(f"{t['name']}: type {t['type']} is neither Q8_0 nor F32")
            f.seek(header["data_start"] + t["offset"])
            blob = f.read(size)
            assert len(blob) == size
            entries.append({"name": pack_name(t["name"], layers), "gguf": t["name"], "type": t["type"],
                            "shape": [ne1, ne0] if len(t["dims"]) == 2 else [ne0], "offset": at, "size": size})
            blobs.append(blob)
            at += size
    meta = json.dumps({"source": os.path.basename(gguf), "tensors": entries}).encode()
    with open(out, "wb") as w:
        w.write(struct.pack("<Q", len(meta)))
        w.write(meta)
        for b in blobs:
            w.write(b)
    print(f"{len(entries)} tensors, {at:,} bytes -> {out}")


def pack(blob: str, base: str, out: str) -> None:
    import mlx.core as mx
    import numpy as np

    # Mapped, not read: M6 runs this beside a resident server, so only one tensor is held at a time.
    raw = np.memmap(blob, dtype=np.uint8, mode="r")
    (n,) = struct.unpack("<Q", raw[:8].tobytes())
    meta = json.loads(raw[8:8 + n].tobytes())
    data = raw[8 + n:]
    arrays: dict[str, mx.array] = {}
    linears: list[str] = []
    for t in meta["tensors"]:
        chunk = data[t["offset"]:t["offset"] + t["size"]]
        if t["type"] == F32:
            arrays[t["name"]] = mx.array(np.frombuffer(chunk, dtype="<f4").reshape(t["shape"]).copy())
            continue
        blocks = np.frombuffer(chunk, dtype=np.dtype([("d", "<f2"), ("q", "i1", 32)]))
        w = (blocks["q"].astype(np.float16) * blocks["d"][:, None]).reshape(t["shape"])
        wq, scales, biases = mx.quantize(mx.array(w), group_size=GROUP, bits=BITS)
        back = mx.dequantize(wq, scales, biases, group_size=GROUP, bits=BITS)
        err = float(mx.max(mx.abs(back.astype(mx.float32) - mx.array(w).astype(mx.float32))).item())
        del back, w
        mx.eval(wq, scales, biases)
        step = float(np.abs(blocks["d"].astype(np.float32)).max())
        print(f"{t['name']:60s} {t['shape']}  max |requant - q8_0| {err:.2e} (largest q8_0 step {step:.2e})")
        stem = t["name"][: -len(".weight")]
        arrays[stem + ".weight"], arrays[stem + ".scales"], arrays[stem + ".biases"] = wq, scales, biases
        linears.append(stem)

    base_dir, out_dir = Path(base).expanduser(), Path(out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "mtp.safetensors"), arrays, metadata={"format": "mlx"})

    for entry in base_dir.iterdir():
        if entry.name in ("config.json", "model.safetensors"):
            continue
        link = out_dir / entry.name
        if not link.exists():
            link.symlink_to(entry)
    base_link = out_dir / "base.safetensors"
    if not base_link.exists():
        base_link.symlink_to(base_dir / "model.safetensors")

    with open(base_dir / "model.safetensors", "rb") as f:
        (hn,) = struct.unpack("<Q", f.read(8))
        names = [k for k in json.loads(f.read(hn)) if k != "__metadata__"]
    weight_map = {k: "base.safetensors" for k in names}
    weight_map.update({k: "mtp.safetensors" for k in arrays})
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"mtp_source": meta["source"]}, "weight_map": weight_map}, indent=1))

    config = json.loads((base_dir / "config.json").read_text())
    config["components"]["mtp"] = True
    config["text_config"]["mtp_num_hidden_layers"] = 1
    prefix = "language_model."
    known = {m["path"] for m in config["modules"]}
    for stem in linears:
        path = stem[len(prefix):]
        if path not in known:
            config["modules"].append({"path": path, "block": 0, "embedding": False, "dtype": "float16"})
        config["quantization"][stem] = {"bits": BITS, "group_size": GROUP}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    total = sum(a.nbytes for a in arrays.values())
    print(f"{len(arrays)} arrays, {total / 1e6:.1f} MB -> {out_dir}/mtp.safetensors; config + index written")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "pack":
        pack(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
