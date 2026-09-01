#!/usr/bin/env python3
"""Measure the FP16 error floor of the drafter reference (vision's
`fp16_error_floor.py` pattern): run the same fixtures through upstream's
model in float16 and score it against the float32 fixtures. If our GPU
pipeline sits at this floor, its error is FP16 itself, not our arithmetic."""

import json, struct, sys
import numpy as np
import torch

from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM

sys.path.insert(0, "Scripts/mtp")
from dump_draft_fixtures import (build_model, read_safetensors, bf16_bits_to_f32,
                                 dequant_int4, WEIGHTS_PATH, CONFIG_PATH,
                                 BACKBONE_HIDDEN, EMBED_SCALE,
                                 NUM_KV_HEADS_SWA, HEAD_DIM_SWA,
                                 NUM_KV_HEADS_FULL, HEAD_DIM_FULL, CASES, OUT_DIR)


def rel(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return (np.abs(actual - reference).max() / max(np.abs(reference).max(), 1e-6),
            np.sqrt(((actual - reference) ** 2).mean()) / max(np.abs(reference).max(), 1e-6))


def read_fixture(path):
    raw = open(path, "rb").read()
    assert raw[:8] == b"TFVFIX01"
    dtype, ndim = struct.unpack("<II", raw[8:16])
    dims = struct.unpack("<" + "I" * ndim, raw[16:16 + 4 * ndim])
    count = int(np.prod(dims)) if dims else 1
    payload = np.frombuffer(raw[16 + 4 * ndim:], dtype="<f4" if dtype == 0 else "<i4")
    return payload.reshape(dims)


def main():
    torch.manual_seed(0)
    model = build_model().to(torch.float16).eval()
    for name, kv_len, seed in CASES:
        case_dir = f"{OUT_DIR}/{name}"
        target_embed = read_fixture(f"{case_dir}/target_embed_in.bin")
        hidden = read_fixture(f"{case_dir}/last_hidden_in.bin")
        k_swa = torch.from_numpy(read_fixture(f"{case_dir}/k_swa.bin")).to(torch.float16)
        v_swa = torch.from_numpy(read_fixture(f"{case_dir}/v_swa.bin")).to(torch.float16)
        k_full = torch.from_numpy(read_fixture(f"{case_dir}/k_full.bin")).to(torch.float16)
        v_full = torch.from_numpy(read_fixture(f"{case_dir}/v_full.bin")).to(torch.float16)

        inputs = torch.from_numpy(
            np.concatenate([target_embed * EMBED_SCALE, hidden])[None, None, :]).to(torch.float16)
        shared = {"sliding_attention": (k_swa, v_swa), "full_attention": (k_full, v_full)}
        position = kv_len - 1
        with torch.no_grad():
            model(inputs_embeds=inputs,
                  position_ids=torch.tensor([[position]]),
                  attention_mask=None, shared_kv_states=shared)

        # capture via hooks is gone; recompute stage errors from the layer
        # outputs the fixtures carry, using module-level forward hooks.
        # Simpler: compare only last_hidden + logits + argmax.
        out_hidden = None
        # rerun with hook on post_projection
        cap = {}
        h1 = model.post_projection.register_forward_hook(
            lambda m, i, o: cap.__setitem__("last", o.detach()))
        h2 = model.lm_head.register_forward_hook(
            lambda m, i, o: cap.__setitem__("logits", o.detach()))
        with torch.no_grad():
            model(inputs_embeds=inputs,
                  position_ids=torch.tensor([[position]]),
                  attention_mask=None, shared_kv_states=shared)
        h1.remove(); h2.remove()

        ref_last = read_fixture(f"{case_dir}/last_hidden_out.bin")
        ref_logits = read_fixture(f"{case_dir}/logits.bin")
        m, r = rel(cap["last"][0, 0].float().numpy(), ref_last)
        ml, rl = rel(cap["logits"][0, 0].float().numpy(), ref_logits)
        arg = int(cap["logits"][0, 0].argmax())
        ref_arg = int(read_fixture(f"{case_dir}/argmax.bin")[0])
        print(f"{name}: last_hidden max={m:.2e} rms={r:.2e} | logits max={ml:.2e} rms={rl:.2e} "
              f"| argmax fp16={arg} fp32={ref_arg} {'OK' if arg == ref_arg else 'DIFF'}")


if __name__ == "__main__":
    main()
