#!/usr/bin/env python3
"""M2 (docs/mtp/04-PHASES.md §5): dump reference intermediates for the Gemma 4
MTP drafter alone.

Loads the 236 MB int4 assistant checkpoint fetched by
`fetch_draft_weights.py`, dequantizes it to float32 (the nibble/group layout
was verified bit-exact against `mx.dequantize`), and runs upstream's own
`Gemma4AssistantForCausalLM` (transformers 5.10.4 — the version whose
`Gemma4TextAttention` can actually construct the all-layers-shared assistant;
5.6.x crashes, see docs/mtp/12-M2-RESULTS.md) on synthetic inputs.

The drafter's inputs are `(token, target last hidden, shared KV from target
layers 28/29)`, so everything is random and the 26B target is never loaded —
the whole point of the synthetic-input plan is that both sides can be fed the
same numbers without running the target in Python (`04-PHASES.md` §5).

Reference runs in **float32** (same policy as the vision fixtures: separate
kernel bugs from precision).

Output: scratch/mtp-fixtures/<case>/*.bin + manifest.json
"""

import json
import os
import struct
import sys

import numpy as np
import torch

from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM

WEIGHTS_PATH = "scratch/mtp-weights/draft.safetensors"
CONFIG_PATH = "scratch/mtp-weights/config.json"
OUT_DIR = "scratch/mtp-fixtures"

FIXTURE_MAGIC = b"TFVFIX01"
DTYPE_F32 = 0
DTYPE_I32 = 1

# Target-side geometry the drafter is pinned against (docs/mtp/01 §4). The
# shared-KV head layouts equal the target's last sliding (28) / full (29)
# layers.
NUM_KV_HEADS_SWA = 8
HEAD_DIM_SWA = 256
NUM_KV_HEADS_FULL = 2
HEAD_DIM_FULL = 512
# The drafter's two inputs are both 2816-wide vectors from the *target*:
# its embedding of the last token (pre-multiplied by the target's
# embed_scale; the drafter's own 1024-wide table is only the tied lm head —
# that is what mlx-vlm's `bind()` reading the target's `embed_tokens` means,
# 01 §5 Q6). Synthetic fixtures therefore generate the embedding vector
# itself; the token id never enters the forward.
BACKBONE_HIDDEN = 2816
EMBED_SCALE = float(BACKBONE_HIDDEN) ** 0.5
VOCAB = 262144

# (name, kv_len, seed). kv_len 1024 is the window boundary (mask short-
# circuits); 1500 puts the query past the window so the bidirectional SWA
# mask is live — the case that distinguishes "read the last 1024 KV" from
# "read the first 1024 KV".
CASES = [
    ("short-37", 37, 0x4D32_0001),
    ("window-1024", 1024, 0x4D32_0002),
    ("past-window-1500", 1500, 0x4D32_0003),
]


# MARK: - safetensors + int4 dequant


def read_safetensors(path: str) -> dict:
    raw = open(path, "rb").read()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    base = 8 + header_len
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        lo, hi = meta["data_offsets"]
        buf = np.frombuffer(raw[base + lo : base + hi], dtype=np.uint8)
        if meta["dtype"] == "BF16":
            out[name] = buf.view("<u2").reshape(meta["shape"]).copy()
        elif meta["dtype"] == "U32":
            out[name] = buf.view("<u4").reshape(meta["shape"]).copy()
        else:
            raise SystemExit(f"unexpected dtype {meta['dtype']} for {name}")
    return out


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def dequant_int4(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                 group: int = 64) -> np.ndarray:
    """MLX affine int4 layout: little-endian U32 words, even element in the low
    nibble, one BF16 scale/bias per row group. Bit-exact against
    `mx.dequantize` at f32 scales (see docs/mtp/12-M2-RESULTS.md)."""
    rows, words = packed.shape
    n = words * 8
    bytes_ = packed.view(np.uint8).reshape(rows, words * 4)
    vals = np.empty((rows, n), dtype=np.float32)
    vals[:, 0::2] = (bytes_ & 0x0F).astype(np.float32)
    vals[:, 1::2] = (bytes_ >> 4).astype(np.float32)
    s = bf16_bits_to_f32(scales)
    b = bf16_bits_to_f32(biases)
    out = vals.reshape(rows, n // group, group)
    return (out * s[..., None] + b[..., None]).reshape(rows, n)


# MARK: - fixtures


def write_tensor(path: str, array: np.ndarray) -> dict:
    if array.dtype == np.int64 or array.dtype == np.int32:
        array = array.astype(np.int32)
        dtype_tag = DTYPE_I32
    else:
        array = array.astype(np.float32)
        dtype_tag = DTYPE_F32
    array = np.ascontiguousarray(array)
    with open(path, "wb") as f:
        f.write(FIXTURE_MAGIC)
        f.write(struct.pack("<II", dtype_tag, array.ndim))
        for d in array.shape:
            f.write(struct.pack("<I", d))
        f.write(array.tobytes(order="C"))
    return {
        "file": os.path.basename(path),
        "dtype": "f32" if dtype_tag == DTYPE_F32 else "i32",
        "shape": list(array.shape),
        "bytes": os.path.getsize(path),
    }


def build_model() -> Gemma4AssistantForCausalLM:
    raw = json.load(open(CONFIG_PATH))
    text_kw = {k: v for k, v in raw["text_config"].items() if v is not None}
    cfg = Gemma4AssistantConfig(
        text_config=text_kw,
        **{k: raw[k] for k in ("backbone_hidden_size", "use_ordered_embeddings",
                               "tie_word_embeddings", "num_centroids",
                               "centroid_intermediate_top_k")})
    model = Gemma4AssistantForCausalLM(cfg)

    tensors = read_safetensors(WEIGHTS_PATH)
    quant_group = raw["quantization"]["group_size"]
    state = {}
    for name, value in tensors.items():
        base, _, leaf = name.rpartition(".")
        if leaf == "scales" or leaf == "biases":
            continue  # folded into the weight below
        if name.endswith(".weight") and (base + ".scales") in tensors:
            w = dequant_int4(value, tensors[base + ".scales"],
                             tensors[base + ".biases"], group=quant_group)
            state[name] = torch.from_numpy(w)
        elif value.dtype == np.uint16:
            state[name] = torch.from_numpy(bf16_bits_to_f32(value))
        else:
            raise SystemExit(f"unexpected tensor {name}")

    # lm_head is tied to the embedding; upstream's tie logic fills it in.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected}")
    if [k for k in missing if k != "lm_head.weight"]:
        raise SystemExit(f"missing keys: {missing}")
    model.tie_weights()

    # Buffers: layer_scalar lives in the checkpoint but not in state_dict.
    for layer_idx, layer in enumerate(model.model.layers):
        key = f"model.layers.{layer_idx}.layer_scalar"
        if key in state:
            layer.layer_scalar.copy_(state[key])
        else:
            raise SystemExit(f"missing {key}")

    return model.to(torch.float32).eval()


def main() -> int:
    torch.manual_seed(0)
    model = build_model()
    embed = model.model.embed_tokens.weight  # [V, 1024] fp32, tied lm head

    manifest = {
        "source_repo": "mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit",
        "source_revision": "bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c",
        "reference": "transformers 5.10.4 Gemma4AssistantForCausalLM, float32",
        "fixture_magic": FIXTURE_MAGIC.decode(),
        "embed_scale": EMBED_SCALE,
        "shared_kv": {
            "sliding": {"target_layer": 28, "num_kv_heads": NUM_KV_HEADS_SWA,
                        "head_dim": HEAD_DIM_SWA},
            "full": {"target_layer": 29, "num_kv_heads": NUM_KV_HEADS_FULL,
                     "head_dim": HEAD_DIM_FULL},
        },
        "cases": [],
    }

    captured: dict[str, torch.Tensor] = {}

    def capture(key):
        def hook(_module, _inputs, output):
            captured[key] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = [
        model.pre_projection.register_forward_hook(capture("h_pre")),
        model.model.norm.register_forward_hook(capture("h_norm")),
        model.post_projection.register_forward_hook(capture("last_hidden")),
        model.lm_head.register_forward_hook(capture("logits")),
    ]
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(capture(f"layer{i}")))

    for name, kv_len, seed in CASES:
        case_dir = os.path.join(OUT_DIR, name)
        os.makedirs(case_dir, exist_ok=True)
        rng = np.random.default_rng(seed)
        position = kv_len - 1

        target_embed = (rng.standard_normal(BACKBONE_HIDDEN) * 0.5).astype(np.float32)
        hidden = rng.standard_normal(BACKBONE_HIDDEN).astype(np.float32)
        k_swa = (rng.standard_normal((1, NUM_KV_HEADS_SWA, kv_len, HEAD_DIM_SWA)) * 0.5).astype(np.float32)
        v_swa = (rng.standard_normal((1, NUM_KV_HEADS_SWA, kv_len, HEAD_DIM_SWA)) * 0.5).astype(np.float32)
        k_full = (rng.standard_normal((1, NUM_KV_HEADS_FULL, kv_len, HEAD_DIM_FULL)) * 0.5).astype(np.float32)
        v_full = (rng.standard_normal((1, NUM_KV_HEADS_FULL, kv_len, HEAD_DIM_FULL)) * 0.5).astype(np.float32)

        tok_embed = target_embed * EMBED_SCALE
        inputs_embeds = torch.from_numpy(
            np.concatenate([tok_embed, hidden])[None, None, :])
        shared_kv = {
            "sliding_attention": (torch.from_numpy(k_swa), torch.from_numpy(v_swa)),
            "full_attention": (torch.from_numpy(k_full), torch.from_numpy(v_full)),
        }
        position_ids = torch.tensor([[position]], dtype=torch.long)

        captured.clear()
        with torch.no_grad():
            out = model(inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        attention_mask=None,
                        shared_kv_states=shared_kv)

        logits = captured["logits"][0, 0].numpy()
        for key in ["h_pre", "layer0", "layer1", "layer2", "layer3",
                    "h_norm", "last_hidden", "logits"]:
            if not np.isfinite(captured[key].numpy()).all():
                raise SystemExit(f"{name}: {key} is not finite")
        argmax = int(logits.argmax())

        entries = {
            "target_embed_in": write_tensor(os.path.join(case_dir, "target_embed_in.bin"),
                                             target_embed),
            "last_hidden_in": write_tensor(os.path.join(case_dir, "last_hidden_in.bin"), hidden),
            "k_swa": write_tensor(os.path.join(case_dir, "k_swa.bin"), k_swa),
            "v_swa": write_tensor(os.path.join(case_dir, "v_swa.bin"), v_swa),
            "k_full": write_tensor(os.path.join(case_dir, "k_full.bin"), k_full),
            "v_full": write_tensor(os.path.join(case_dir, "v_full.bin"), v_full),
            "h_pre": write_tensor(os.path.join(case_dir, "h_pre.bin"),
                                  captured["h_pre"][0, 0].numpy()),
        }
        for i in range(4):
            entries[f"layer{i}"] = write_tensor(
                os.path.join(case_dir, f"layer{i}.bin"),
                captured[f"layer{i}"][0, 0].numpy())
        entries["h_norm"] = write_tensor(os.path.join(case_dir, "h_norm.bin"),
                                         captured["h_norm"][0, 0].numpy())
        entries["last_hidden_out"] = write_tensor(
            os.path.join(case_dir, "last_hidden_out.bin"),
            captured["last_hidden"][0, 0].numpy())
        entries["logits"] = write_tensor(os.path.join(case_dir, "logits.bin"), logits)
        entries["argmax"] = write_tensor(os.path.join(case_dir, "argmax.bin"),
                                         np.array([argmax], dtype=np.int32))

        assert np.allclose(out.last_hidden_state[0, 0].numpy(),
                           captured["last_hidden"][0, 0].numpy(), atol=1e-5)
        assert int(out.logits[0, 0].numpy().argmax()) == argmax

        manifest["cases"].append({
            "name": name,
            "kv_len": kv_len,
            "position": position,
            "argmax": argmax,
            "logits_max": float(logits.max()),
            "logits_min": float(logits.min()),
            "dir": name,
            "tensors": entries,
        })
        print(f"{name}: kv_len={kv_len} argmax={argmax} "
              f"logits=[{logits.min():.2f}, {logits.max():.2f}]")

    for h in handles:
        h.remove()

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'manifest.json')} ({len(CASES)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
