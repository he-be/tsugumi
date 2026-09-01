#!/usr/bin/env python3
"""M3.5 (docs/mtp/14-M3.5-RESULTS.md §2-3): replay the runtime's *real* drafter
rounds through upstream's `Gemma4AssistantForCausalLM` in float32.

    replay_draft_rounds.py <dump-dir> <probe.tsv> <assistant-snapshot-dir>

`<dump-dir>` is what `TF_DRAFT_DUMP=<dir>` writes next to `TF_DRAFT_PROBE`:
one record per round (bonus-token embedding, the target's post-norm hidden, the
RoPE position, the cache length) plus the final layer-28/29 K/V slabs, which
every round slices as a prefix. `<probe.tsv>` supplies, per round, what the
runtime proposed and what the target actually drew.

12-M2's fixtures pin the drafter against *random* K/V at a synthetic position.
That cannot catch a fault which only shows on the structured cache the runtime
really has — nor one whose damage lands outside the drafter's own arithmetic
(the `attnOut` overrun of §1 did both). This feeds the reference the exact
numbers the Metal kernels ran on, so a disagreement is a runtime fault and an
agreement moves the question onto the weights or the target.

The snapshot may be bf16 or an MLX affine-int4 conversion of any group size —
swapping checkpoints is how §2-4 ruled out the drafter's quantization.

`ABLATE=hidden-zero|hidden-shift|embed-zero|kv-1` knocks out one input at a
time (§2-5). A knock-out that costs nothing means the runtime is handing over
something the drafter cannot read.
"""

import glob
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM

# Target-side geometry, same pinning as dump_draft_fixtures.py (01 §4).
NUM_KV_HEADS_SWA, HEAD_DIM_SWA = 8, 256
NUM_KV_HEADS_FULL, HEAD_DIM_FULL = 2, 512


def dequant_int4(packed, scales, biases, group):
    """MLX affine int4: little-endian U32 words, even element in the low nibble,
    one BF16 scale/bias per row group (12-M2 §2)."""
    rows, words = packed.shape
    n = words * 8
    b = packed.view(np.uint8).reshape(rows, words * 4)
    vals = np.empty((rows, n), dtype=np.float32)
    vals[:, 0::2] = (b & 0x0F).astype(np.float32)
    vals[:, 1::2] = (b >> 4).astype(np.float32)
    return (vals.reshape(rows, n // group, group) * scales[..., None]
            + biases[..., None]).reshape(rows, n)


def build(snapshot: str) -> Gemma4AssistantForCausalLM:
    raw = json.load(open(os.path.join(snapshot, "config.json")))
    cfg = Gemma4AssistantConfig(
        text_config={k: v for k, v in raw["text_config"].items() if v is not None},
        **{k: raw[k] for k in ("backbone_hidden_size", "use_ordered_embeddings",
                               "tie_word_embeddings", "num_centroids",
                               "centroid_intermediate_top_k")})
    model = Gemma4AssistantForCausalLM(cfg)

    packed = {}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        packed.update(load_file(path))

    group = (raw.get("quantization") or {}).get("group_size")
    state = {}
    for name, value in packed.items():
        base, _, leaf = name.rpartition(".")
        if leaf in ("scales", "biases"):
            continue  # folded into the weight below
        if group and name.endswith(".weight") and (base + ".scales") in packed:
            state[name] = torch.from_numpy(dequant_int4(
                value.view(torch.int32).numpy().astype(np.uint32),
                packed[base + ".scales"].to(torch.float32).numpy(),
                packed[base + ".biases"].to(torch.float32).numpy(),
                group))
        else:
            state[name] = value.to(torch.float32)

    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("layer_scalar")]
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected[:8]}")
    missing = [k for k in missing if k != "lm_head.weight"]  # tied
    if missing:
        raise SystemExit(f"missing keys: {missing[:8]}")
    model.tie_weights()
    for i, layer in enumerate(model.model.layers):
        layer.layer_scalar.copy_(state[f"model.layers.{i}.layer_scalar"])

    print(f"loaded {os.path.basename(snapshot)} (group_size={group})", flush=True)
    return model.to(torch.float32).eval()


def read_rounds(dump: str):
    meta = json.load(open(os.path.join(dump, "meta.json")))
    rows = meta["rows"]

    def slab(name, heads, dim):
        return np.fromfile(os.path.join(dump, f"{name}.bin"),
                           dtype=np.float32).reshape(rows, heads, dim)

    cache = {
        "sliding_attention": (slab("k_swa", NUM_KV_HEADS_SWA, HEAD_DIM_SWA),
                              slab("v_swa", NUM_KV_HEADS_SWA, HEAD_DIM_SWA)),
        "full_attention": (slab("k_full", NUM_KV_HEADS_FULL, HEAD_DIM_FULL),
                           slab("v_full", NUM_KV_HEADS_FULL, HEAD_DIM_FULL)),
    }

    records = []
    with open(os.path.join(dump, "rounds.bin"), "rb") as f:
        while True:
            head = f.read(16)
            if len(head) < 16:
                break
            _bonus, position, kv_len, width = struct.unpack("<iiii", head)
            embed = np.frombuffer(f.read(4 * width), dtype=np.float32)
            hidden = np.frombuffer(f.read(4 * width), dtype=np.float32)
            f.read(4 * width)  # pre-norm residual, diagnostics only
            records.append((position, kv_len, embed, hidden))
    return records, cache, rows


def read_probe(path: str) -> dict:
    """position -> (what the runtime proposed, what the target drew)."""
    out = {}
    for line in open(path):
        if line.startswith("#") or line.startswith("pos"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 5:
            continue
        proposed = [int(x) for x in f[2].split(",") if x]
        actual = [int(x) for x in f[3].split(",") if x]
        out[int(f[0])] = (proposed[0] if proposed else None,
                          actual[0] if actual else None)
    return out


def main() -> int:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    dump, tsv = sys.argv[1], sys.argv[2]
    snapshot = sorted(glob.glob(sys.argv[3]))[-1]
    ablate = os.environ.get("ABLATE", "none")

    records, cache, rows = read_rounds(dump)
    probe = read_probe(tsv)
    model = build(snapshot)
    print(f"records={len(records)} rows={rows} ablate={ablate}", flush=True)

    n = agree = reference_hit = runtime_hit = 0
    for i, (position, kv_len, embed, hidden) in enumerate(records):
        if ablate == "hidden-zero":
            hidden = np.zeros_like(hidden)
        elif ablate == "hidden-shift":
            hidden = records[max(0, i - 1)][3]
        elif ablate == "embed-zero":
            embed = np.zeros_like(embed)
        elif ablate == "kv-1":
            kv_len = 1

        shared = {
            name: (torch.from_numpy(k[:kv_len].transpose(1, 0, 2)[None].copy()),
                   torch.from_numpy(v[:kv_len].transpose(1, 0, 2)[None].copy()))
            for name, (k, v) in cache.items()
        }
        with torch.no_grad():
            out = model(inputs_embeds=torch.from_numpy(
                            np.concatenate([embed, hidden])[None, None, :].copy()),
                        position_ids=torch.tensor([[position]], dtype=torch.long),
                        attention_mask=None,
                        shared_kv_states=shared)
        reference = int(out.logits[0, 0].numpy().argmax())

        proposed, actual = probe.get(position, (None, None))
        n += 1
        reference_hit += (reference == actual)
        if proposed is not None:
            runtime_hit += (proposed == actual)
            agree += (reference == proposed)

    print(f"rounds                : {n}")
    print(f"reference == runtime  : {agree}/{n} = {agree / n:.4f}")
    print(f"reference accepted    : {reference_hit}/{n} = {reference_hit / n:.4f}")
    print(f"runtime   accepted    : {runtime_hit}/{n} = {runtime_hit / n:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
