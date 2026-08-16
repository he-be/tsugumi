#!/usr/bin/env python3
"""PLAN_VISION V0: dump reference intermediates for the Gemma 4 vision tower.

Builds `Gemma4VisionModel` + `Gemma4MultimodalEmbedder` from the 356 range-fetched
vision tensors alone - the 26B text side is never loaded - runs each test image
through it, and writes every stage to a flat binary the Swift side can read.

The reference runs in **float32**, not bf16. The point of these fixtures is to
separate our kernel bugs from our precision, so the reference has to be the
more precise of the two; a bf16 reference would fold its own rounding into the
tolerance and hide exactly the errors §6-1 layer B exists to catch.

Output: scratch/vision-fixtures/<image>-s<soft_tokens>/*.bin + manifest.json
"""

import json
import os
import struct
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
)

CONFIG_PATH = "scratch/vision-ref-config.json"
PROCESSOR_PATH = "scratch/vision-ref-processor.json"
WEIGHTS_PATH = "scratch/vision-weights/vision.safetensors"
IMAGE_DIR = "scratch/vision-fixtures/images"
OUT_DIR = "scratch/vision-fixtures"

SOFT_TOKENS = [70, 280]

FIXTURE_MAGIC = b"TFVFIX01"
DTYPE_F32 = 0
DTYPE_I32 = 1


def write_tensor(path: str, array: np.ndarray) -> dict:
    """Flat little-endian dump: magic, dtype, ndim, dims, payload."""
    if array.dtype == np.int32 or array.dtype == np.int64:
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


def build_models() -> tuple[Gemma4VisionModel, Gemma4MultimodalEmbedder, dict]:
    raw = json.load(open(CONFIG_PATH))
    vision_config = Gemma4VisionConfig(**raw["vision_config"])
    vision_config._attn_implementation = "eager"
    text_config = Gemma4TextConfig(**raw["text_config"])

    state = load_file(WEIGHTS_PATH)
    print(f"loaded {len(state)} vision tensors")

    tower_state, embed_state = {}, {}
    for name, tensor in state.items():
        value = tensor.to(torch.float32)
        if name.startswith("model.vision_tower."):
            tower_state[name[len("model.vision_tower.") :]] = value
        elif name.startswith("model.embed_vision."):
            embed_state[name[len("model.embed_vision.") :]] = value
        else:
            raise SystemExit(f"unexpected tensor {name}")

    tower = Gemma4VisionModel(vision_config)
    missing, unexpected = tower.load_state_dict(tower_state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected tower keys: {unexpected}")
    if missing:
        raise SystemExit(f"missing tower keys: {missing}")

    embedder = Gemma4MultimodalEmbedder(vision_config, text_config)
    missing, unexpected = embedder.load_state_dict(embed_state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected embedder keys: {unexpected}")
    if missing:
        raise SystemExit(f"missing embedder keys: {missing}")

    tower = tower.to(torch.float32).eval()
    embedder = embedder.to(torch.float32).eval()
    return tower, embedder, raw


def main() -> int:
    torch.manual_seed(0)
    tower, embedder, raw = build_models()

    proc_kwargs = dict(json.load(open(PROCESSOR_PATH))["image_processor"])
    proc_kwargs.pop("image_processor_type", None)
    proc_kwargs.pop("image_seq_length", None)

    layer_probe_indices = [0, 13, 26]
    captured: dict[str, torch.Tensor] = {}

    def capture(key):
        def hook(_module, _inputs, output):
            captured[key] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    handles = [tower.patch_embedder.register_forward_hook(capture("patch_embed"))]
    for i in layer_probe_indices:
        handles.append(tower.encoder.layers[i].register_forward_hook(capture(f"layer{i}")))

    manifest = {
        "source_repo": "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
        "source_revision": "f1e06dc520982d9b9edd76859fdb7ab209449949",
        "reference": "transformers 5.6.2, eager attention, float32",
        "fixture_magic": FIXTURE_MAGIC.decode(),
        "layer_probe_indices": layer_probe_indices,
        "vision_config": raw["vision_config"],
        "cases": [],
    }

    for image_name in sorted(os.listdir(IMAGE_DIR)):
        if not image_name.endswith(".png"):
            continue
        image = Image.open(os.path.join(IMAGE_DIR, image_name)).convert("RGB")
        for soft in SOFT_TOKENS:
            stem = f"{os.path.splitext(image_name)[0]}-s{soft}"
            case_dir = os.path.join(OUT_DIR, stem)
            os.makedirs(case_dir, exist_ok=True)

            processor = Gemma4ImageProcessor(**{**proc_kwargs, "max_soft_tokens": soft})
            batch = processor(images=[image], return_tensors="pt")
            pixel_values = batch["pixel_values"].to(torch.float32)
            position_ids = batch["image_position_ids"]
            n_soft = int(batch["num_soft_tokens_per_image"][0])

            # PLAN_VISION §4-4 processes one image at a time, so the padded rows
            # the batch API adds are dropped here: every fixture is exactly the
            # patches the runtime will see, and nothing downstream depends on a
            # padding mask.
            real = int((position_ids[0] >= 0).all(dim=-1).sum())
            pixel_values = pixel_values[:, :real]
            position_ids = position_ids[:, :real]
            pw = int(position_ids[0, :, 0].max()) + 1
            ph = int(position_ids[0, :, 1].max()) + 1
            assert pw * ph == real, f"{pw}x{ph} != {real}"
            assert real // 9 == n_soft, f"{real}//9 != {n_soft}"

            captured.clear()
            with torch.no_grad():
                out = tower(pixel_values=pixel_values, pixel_position_ids=position_ids)
                pooled = out.last_hidden_state
                soft_tokens = embedder(pooled)

            assert pooled.shape[0] == n_soft, f"{pooled.shape} vs {n_soft}"

            entries = {
                "pixel_values": write_tensor(
                    os.path.join(case_dir, "pixel_values.bin"), pixel_values[0].numpy()
                ),
                "position_ids": write_tensor(
                    os.path.join(case_dir, "position_ids.bin"),
                    position_ids[0].numpy().astype(np.int32),
                ),
                "patch_embed": write_tensor(
                    os.path.join(case_dir, "patch_embed.bin"),
                    captured["patch_embed"][0].numpy(),
                ),
                "pooled": write_tensor(
                    os.path.join(case_dir, "pooled.bin"), pooled.numpy()
                ),
                "soft_tokens": write_tensor(
                    os.path.join(case_dir, "soft_tokens.bin"), soft_tokens.numpy()
                ),
            }
            for i in layer_probe_indices:
                entries[f"layer{i}"] = write_tensor(
                    os.path.join(case_dir, f"layer{i}.bin"), captured[f"layer{i}"][0].numpy()
                )

            case = {
                "name": stem,
                "image": image_name,
                "image_size": [image.width, image.height],
                "max_soft_tokens": soft,
                "patch_grid": [pw, ph],
                "num_patches": real,
                "num_soft_tokens": n_soft,
                "dir": stem,
                "tensors": entries,
            }
            manifest["cases"].append(case)
            print(
                f"{stem}: {image.width}x{image.height} -> {pw}x{ph} patches "
                f"({real}) -> {n_soft} soft tokens"
            )

    for h in handles:
        h.remove()

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'manifest.json')} ({len(manifest['cases'])} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
