# Vision reference fixtures

Regenerates the reference intermediates the vision work is measured against
(`PLAN_VISION.md` §5-V0, §6-1). Nothing here runs during a build, a test, or an
install — it is the tooling that produces the ground truth, and its output lands
in `scratch/`, which is git-ignored.

The reason it exists: three different things all look like "the model gave a
slightly odd answer about the picture" — a kernel bug, a resize difference, and
the checkpoint's own quality. Without an independent reference there is no way
to tell them apart, so the fixtures come first and the kernels come after.

## What gets produced

| Path | Size | What it is |
| --- | ---: | --- |
| `scratch/vision-venv/` | ~2 GB | `transformers==5.6.2`, torch (CPU), torchvision, pillow |
| `scratch/vision-ref-config.json` | 4 KB | The checkpoint's `config.json` |
| `scratch/vision-ref-processor.json` | 2 KB | The checkpoint's `processor_config.json` |
| `scratch/vision-weights/vision.safetensors` | 1.15 GB | The 356 BF16 vision tensors |
| `scratch/vision-fixtures/` | 195 MB | Test images and reference dumps |

## Running it

From the repository root, in order. Steps 1-3 are one-time; step 5 is the one
worth re-running when the reference implementation is bumped.

```bash
# 1. Reference environment. The transformers pin is not optional: the fixtures
#    record what a specific version computes, and the runtime is compared
#    against them long after this runs.
uv venv --python 3.12 scratch/vision-venv
VIRTUAL_ENV=scratch/vision-venv uv pip install \
  "transformers==5.6.2" torch torchvision pillow safetensors numpy requests

# 2. Checkpoint metadata.
REV=f1e06dc520982d9b9edd76859fdb7ab209449949
BASE=https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized/resolve/$REV
curl -sSL "$BASE/config.json"           -o scratch/vision-ref-config.json
curl -sSL "$BASE/processor_config.json" -o scratch/vision-ref-processor.json

# 3. Vision weights: 1.15 GB pulled out of a 46 GB shard with HTTP Range.
scratch/vision-venv/bin/python Scripts/vision/fetch_vision_weights.py

# 4. Test images (deterministic; safe to re-run).
scratch/vision-venv/bin/python Scripts/vision/make_test_images.py

# 5. Reference dumps.
scratch/vision-venv/bin/python Scripts/vision/dump_vision_fixtures.py
```

Then, from Swift, `VisionFixtures.load()` reads them; the suites under
`Tests/Tsugumi/Core/Vision/` skip silently when they are absent, so a
clean checkout still passes `Scripts/test.sh`.

## Notes that are easy to get wrong

- **`fetch_vision_weights.py` verifies before it writes.** The index SHA-256,
  the tensor count (356), the byte total (1,145,588,832), and that every tensor
  is BF16 are all checked against the values `PLAN_VISION.md` §1-1 derived
  independently. A silent mismatch here would poison every later comparison, so
  it fails rather than downloads.
- **The reference runs in float32, not bf16.** A reference has to be more
  precise than the thing it measures; a bf16 reference would fold its own
  rounding into the tolerance and hide the writing-it-down errors the fixtures
  exist to catch.
- **The 26B text weights are never loaded.** The tower and the projector are
  built from the 356 vision tensors alone, so this runs in a couple of GB.
- **The test images are synthetic on purpose.** They are reproducible from this
  repository, and their aspect ratios are chosen so that all six cases land on
  *different* soft-token counts (64, 256, 65, 260, 63, 266) — the property that
  makes a "280 tokens per image" assumption fail loudly instead of quietly.

## Fixture format

`TFVFIX01` + `dtype u32` + `ndim u32` + `dims u32[ndim]` + little-endian
payload, one tensor per file, with `manifest.json` listing the cases. Flat on
purpose: it is read by `VisionFixtures.readTensor` in about thirty lines, with
no dependency on a serialization library on either side.
