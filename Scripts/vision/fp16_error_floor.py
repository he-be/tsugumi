#!/usr/bin/env python3
"""PLAN_VISION V4: measure the FP16 error floor of the vision tower.

The assembled tower matches the float32 reference to rms ~5e-4 but with a
worst-element relative error of up to 6e-2 at 280 soft tokens, which is over the
2e-2 the plan pencilled in for §6-1 layer B. That threshold was explicitly
provisional ("V4 で実測してから確定"), and there are two possible readings of
the gap:

  a. our kernels get something wrong in a way that only shows on the largest
     activations, or
  b. an FP16 residual stream cannot track a float32 one any closer through 27
     layers of a network whose activations reach 2600.

This script decides between them without involving our Metal code at all. It
runs the *upstream* implementation twice on the same input: once in float32 (the
fixture), and once with our precision imposed on it — FP16 inputs, and the
hidden state rounded to FP16 after every stage the runtime materializes in FP16.
Nothing else changes. Whatever error that produces is the floor: no
implementation carrying the tower in FP16 can beat it, and if our measured error
sits at or below it, (a) is ruled out.

Usage: scratch/vision-venv/bin/python Scripts/vision/fp16_error_floor.py [case ...]
"""

import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_vision_fixtures import build_models  # noqa: E402

FIXTURE_ROOT = "scratch/vision-fixtures"
DEFAULT_CASES = ["tall-480x1200-s280", "square-512-s280", "wide-1024x768-s70"]


def read_tensor(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        payload = f.read()
    assert payload[:8] == b"TFVFIX01", path
    dtype_tag, ndim = struct.unpack_from("<II", payload, 8)
    dims = struct.unpack_from("<" + "I" * ndim, payload, 16)
    start = 16 + 4 * ndim
    dtype = np.float32 if dtype_tag == 0 else np.int32
    return np.frombuffer(payload, dtype=dtype, offset=start).reshape(dims)


def to_fp16_and_back(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def profile(actual: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    diff = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    max_ref = float(np.max(np.abs(reference.astype(np.float64))))
    rms = float(np.sqrt(np.mean(diff**2)))
    return float(np.max(diff)) / max_ref, rms / max_ref, max_ref


def main() -> int:
    torch.manual_seed(0)
    tower, embedder, _ = build_models()

    # Impose the runtime's precision on the reference: every tensor the Metal
    # tower materializes in FP16 is rounded here too. The kernels accumulate
    # their reductions in float, exactly as torch does at float32, so only the
    # materialized tensors are rounded — not the arithmetic inside a stage.
    handles = [
        tower.patch_embedder.register_forward_hook(
            lambda _m, _i, out: to_fp16_and_back(out)
        )
    ]
    for layer in tower.encoder.layers:
        handles.append(
            layer.register_forward_hook(lambda _m, _i, out: to_fp16_and_back(out))
        )
    # No hook on the pooler. Upstream's pooler returns `mean * sqrt(1152)`
    # *before* standardization, and on a 48x48 grid that intermediate reaches
    # ~9e4 — past FP16's 65504. Rounding it here produced Inf, which is not a
    # statement about the runtime: `vision_pool_std_block` computes the mean,
    # the sqrt scaling and the standardization in one kernel with a float
    # accumulator, and only the standardized result (max ~8) is ever
    # materialized as FP16. The hook therefore goes where the runtime actually
    # rounds — on the standardized output, below.
    #
    # This is a real constraint on the tower, not an accident of this script:
    # splitting that kernel so the scaled mean crosses an FP16 buffer would
    # overflow on square images at 280 soft tokens.

    manifest = json.load(open(os.path.join(FIXTURE_ROOT, "manifest.json")))
    by_name = {case["name"]: case for case in manifest["cases"]}
    wanted = sys.argv[1:] or DEFAULT_CASES

    print(f"{'case':<22} {'stage':<12} {'max rel':>10} {'rms rel':>10} {'max|ref|':>10}")
    for name in wanted:
        case = by_name[name]
        case_dir = os.path.join(FIXTURE_ROOT, case["dir"])
        pixel_values = torch.from_numpy(read_tensor(os.path.join(case_dir, "pixel_values.bin")).copy())
        position_ids = torch.from_numpy(read_tensor(os.path.join(case_dir, "position_ids.bin")).copy())

        # The runtime's preprocessor emits FP16 patch rows, so the reference
        # gets the same rounded input rather than the float32 original.
        pixel_values = to_fp16_and_back(pixel_values).unsqueeze(0)
        position_ids = position_ids.unsqueeze(0).long()

        with torch.no_grad():
            out = tower(pixel_values=pixel_values, pixel_position_ids=position_ids)
            pooled = to_fp16_and_back(out.last_hidden_state)
            soft_tokens = to_fp16_and_back(embedder(pooled))

        for stage, actual in (("pooled", pooled), ("soft-tokens", soft_tokens)):
            reference = read_tensor(
                os.path.join(case_dir, "pooled.bin" if stage == "pooled" else "soft_tokens.bin")
            )
            max_rel, rms_rel, max_ref = profile(actual.numpy(), reference)
            print(f"{name:<22} {stage:<12} {max_rel:>10.3e} {rms_rel:>10.3e} {max_ref:>10.4g}")

    for handle in handles:
        handle.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
