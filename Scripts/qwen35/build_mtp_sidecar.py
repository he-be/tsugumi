#!/usr/bin/env python3
"""差し替え済み MTP ヘッドを、ランタイムが mmap できる sidecar に書き出す。

**`.moepack` には MTP が 1 バイトも入っていない** — `RepackPlanner.classify` が
`language_model.mtp.` を `.excludedDraft` に落とすからで、これは
[30 §6-6](../../docs/qwen35moe/30-MTP-HEAD-GRAFT.md) が「pack を作り直す理由が
無い」と書いたとおりである。ヘッドを実機で回すのに 20 GB の repack をやり直す
必要は無く、**503 MB の sidecar を 1 枚足せばよい。**

出るもの (既定 `~/LLM/ornith-mtp-head/`):

    mtp_head.json     形と量子化と offset の目録
    mtp_core.bin      平文 9 + 8bit 8 本 (weights/scales/biases) = 33 本
    mtp_experts.bin   256 エキスパートの blob。**本体の packed_experts と同じ
                      並び** (gate w/s/b, up w/s/b, down w/s/b、ページ境界まで
                      padding) なので `Model.routedExpertOffsets` と 1:1 で合う

`--moepack` を渡すと、その manifest の `expertStride` と本体 layer 0 の blob 内
offset に**一致するか**を検算する (合わなければ書かずに落ちる)。

    scratch/mtp-venv/bin/python Scripts/qwen35/build_mtp_sidecar.py \\
        ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa-baked \\
        --out ~/LLM/ornith-mtp-head \\
        --moepack scratch/ornith-oq4e-g64.moepack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_io import Checkpoint  # noqa: E402

PREFIX = "language_model.mtp."
PAGE = 16384
ALIGN = 256

# 平文 (BF16) 9 本。左が sidecar の短い名、右がチェックポイントの名。
PLAIN = {
    "fc": "fc.weight",
    "router": "layers.0.mlp.gate.weight",
    "pre_fc_norm_embedding": "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden": "pre_fc_norm_hidden.weight",
    "input_layernorm": "layers.0.input_layernorm.weight",
    "post_attention_layernorm": "layers.0.post_attention_layernorm.weight",
    "q_norm": "layers.0.self_attn.q_norm.weight",
    "k_norm": "layers.0.self_attn.k_norm.weight",
    "final_norm": "norm.weight",
}

# 量子化 8 本 (すべて 8bit/g64 affine)。
QUANT = {
    "q_proj": "layers.0.self_attn.q_proj",
    "k_proj": "layers.0.self_attn.k_proj",
    "v_proj": "layers.0.self_attn.v_proj",
    "o_proj": "layers.0.self_attn.o_proj",
    "shared_gate_proj": "layers.0.mlp.shared_expert.gate_proj",
    "shared_up_proj": "layers.0.mlp.shared_expert.up_proj",
    "shared_down_proj": "layers.0.mlp.shared_expert.down_proj",
    "shared_expert_gate": "layers.0.mlp.shared_expert_gate",
}

# 4bit/g64 のエキスパート 3 役。blob 内はこの順で weights, scales, biases。
EXPERT_ROLES = [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]


def align(value: int, to: int = ALIGN) -> int:
    return ((value + to - 1) // to) * to


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="差し替え済み・焼き込み済み (…-shisa-baked)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--moepack", help="expertStride と blob 内 offset を照合する")
    args = ap.parse_args()

    root = Path(args.checkpoint).expanduser()
    if not (root / "mtp_graft_manifest.json").exists():
        raise SystemExit("差し替え済みではない (mtp_graft_manifest.json が無い)")
    if not (root / "bake_manifest.json").exists():
        raise SystemExit("焼き込み前は渡さない — ランタイムは q_norm に "
                         "head_dim**-0.5 が畳まれている前提で scale 1.0 を使う")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    ck = Checkpoint(root)
    cfg = ck.config
    text = cfg.get("text_config", cfg)

    def ref(short: str):
        name = PREFIX + short
        if name not in ck:
            raise SystemExit(f"{name} がチェックポイントに無い")
        return ck.tensors[name]

    # ---- core --------------------------------------------------------------
    core_index: dict[str, dict] = {}
    blocks: list[tuple[int, str, str]] = []   # (offset, tensor name, component)
    cursor = 0
    for short, tail in PLAIN.items():
        t = ref(tail)
        if t.dtype != "BF16":
            raise SystemExit(f"{short} は BF16 のはずが {t.dtype}")
        core_index[short] = {"dtype": "bf16", "shape": list(t.shape),
                             "offset": cursor, "size": t.nbytes,
                             "scaleOffset": 0, "scaleSize": 0,
                             "biasOffset": 0, "biasSize": 0, "bits": 16}
        blocks.append((cursor, PREFIX + tail, "raw"))
        cursor = align(cursor + t.nbytes)

    for short, tail in QUANT.items():
        w, s, b = ref(tail + ".weight"), ref(tail + ".scales"), ref(tail + ".biases")
        if w.dtype != "U32" or s.dtype != "BF16" or b.dtype != "BF16":
            raise SystemExit(f"{short} の dtype が想定外: {w.dtype}/{s.dtype}/{b.dtype}")
        rows, packed = w.shape
        cols = packed * 4          # 8bit: uint32 1 個に 4 重み
        groups = s.shape[1]
        if cols // groups != 64:
            raise SystemExit(f"{short} の group が 64 でない ({cols}/{groups})")
        w_off = cursor
        blocks.append((w_off, PREFIX + tail + ".weight", "raw"))
        s_off = align(w_off + w.nbytes)
        blocks.append((s_off, PREFIX + tail + ".scales", "raw"))
        b_off = align(s_off + s.nbytes)
        blocks.append((b_off, PREFIX + tail + ".biases", "raw"))
        cursor = align(b_off + b.nbytes)
        core_index[short] = {"dtype": "u32", "shape": [rows, cols], "bits": 8,
                             "groupSize": 64,
                             "offset": w_off, "size": w.nbytes,
                             "scaleOffset": s_off, "scaleSize": s.nbytes,
                             "biasOffset": b_off, "biasSize": b.nbytes}

    # ランタイムは `bytesNoCopy` でこのファイルを丸ごと 1 本の MTLBuffer に
    # するので、長さはページの倍数でなければならない。
    core_size = ((cursor + PAGE - 1) // PAGE) * PAGE
    core_path = out / "mtp_core.bin"
    digest = hashlib.sha256()
    with core_path.open("wb") as fh:
        for offset, name, _ in blocks:
            fh.seek(0, 2)
            pad = offset - fh.tell()
            if pad < 0:
                raise SystemExit("core の並びが崩れた")
            fh.write(b"\0" * pad)
            fh.write(ck.raw(name).tobytes())
        fh.write(b"\0" * (core_size - fh.tell()))
    with core_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    core_sha = digest.hexdigest()
    print(f"  mtp_core.bin    {core_size:,} B  {len(core_index)} 本", flush=True)

    # ---- experts -----------------------------------------------------------
    sub: dict[str, dict] = {}
    blob_cursor = 0
    arrays = {}
    for role, tail in EXPERT_ROLES:
        w = ref(f"layers.0.mlp.switch_mlp.{tail}.weight")
        s = ref(f"layers.0.mlp.switch_mlp.{tail}.scales")
        b = ref(f"layers.0.mlp.switch_mlp.{tail}.biases")
        experts = w.shape[0]
        per_w, per_s, per_b = w.nbytes // experts, s.nbytes // experts, b.nbytes // experts
        sub[role] = {"weightOffset": blob_cursor, "weightSize": per_w}
        blob_cursor += per_w
        sub[role]["scaleOffset"] = blob_cursor
        sub[role]["scaleSize"] = per_s
        blob_cursor += per_s
        sub[role]["biasOffset"] = blob_cursor
        sub[role]["biasSize"] = per_b
        blob_cursor += per_b
        arrays[role] = (ck.raw(w.name).reshape(experts, -1).view("<u1"),
                        ck.raw(s.name).reshape(experts, -1).view("<u1"),
                        ck.raw(b.name).reshape(experts, -1).view("<u1"))
    stride = ((blob_cursor + PAGE - 1) // PAGE) * PAGE
    num_experts = arrays["gate"][0].shape[0]

    if args.moepack:
        manifest = json.loads((Path(args.moepack).expanduser()
                               / "manifest.json").read_text())
        if manifest["expertStride"] != stride:
            raise SystemExit(f"expertStride が本体 {manifest['expertStride']} と "
                             f"違う ({stride})")
        if manifest["expertsPerLayer"] != num_experts:
            raise SystemExit("expertsPerLayer が本体と違う")
        print(f"  照合: expertStride {stride:,} B / {num_experts} 本 — 本体と一致",
              flush=True)

    experts_path = out / "mtp_experts.bin"
    digest = hashlib.sha256()
    pad = bytearray(stride - blob_cursor)
    with experts_path.open("wb") as fh:
        for e in range(num_experts):
            for role, _ in EXPERT_ROLES:
                w, s, b = arrays[role]
                fh.write(w[e].tobytes())
                fh.write(s[e].tobytes())
                fh.write(b[e].tobytes())
            fh.write(pad)
    with experts_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    experts_sha = digest.hexdigest()
    print(f"  mtp_experts.bin {stride * num_experts:,} B  {num_experts} blob", flush=True)

    head = {
        "format": "tsugumi-mtp-head-v1",
        "source": {
            "checkpoint": str(root),
            "graft": json.loads((root / "mtp_graft_manifest.json").read_text())
                        .get("source", {}),
        },
        "arch": {
            "hiddenSize": text["hidden_size"],
            "numHeads": text["num_attention_heads"],
            "numKVHeads": text["num_key_value_heads"],
            "headDim": text.get("head_dim", 256),
            "numExperts": num_experts,
            "topK": text["num_experts_per_tok"],
            "moeIntermediateSize": text["moe_intermediate_size"],
            "sharedIntermediateSize": text.get("shared_expert_intermediate_size",
                                               text["moe_intermediate_size"]),
            "ropeTheta": text.get("rope_parameters", text).get("rope_theta",
                                                              text.get("rope_theta")),
            "partialRotaryFactor": text.get("partial_rotary_factor", 0.25),
            "rmsNormEps": text["rms_norm_eps"],
        },
        "core": {"file": "mtp_core.bin", "size": core_size, "sha256": core_sha,
                 "tensors": core_index},
        "experts": {"file": "mtp_experts.bin", "count": num_experts,
                    "stride": stride, "bits": 4, "groupSize": 64,
                    "size": stride * num_experts, "sha256": experts_sha,
                    "subTensors": sub},
    }
    (out / "mtp_head.json").write_text(json.dumps(head, indent=2, ensure_ascii=False))
    print(f"  → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
