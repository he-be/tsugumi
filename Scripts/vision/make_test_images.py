#!/usr/bin/env python3
"""Deterministic test images for the vision reference fixtures.

Synthetic rather than photographic so the fixtures are reproducible from this
repository alone, and so the aspect ratios are chosen rather than inherited:
each image lands on a different soft-token count, which is the property
PLAN_VISION §2-1 says the "280 is a cap, not a constant" rule turns on.

Output: scratch/vision-fixtures/images/*.png
"""

import math
import os

from PIL import Image

OUT = "scratch/vision-fixtures/images"

# (name, width, height) - all three resize to different patch grids.
SPECS = [
    ("square-512", 512, 512),
    ("wide-1024x768", 1024, 768),
    ("tall-480x1200", 480, 1200),
]


def render(width: int, height: int) -> Image.Image:
    """A fixed pattern with structure at several scales.

    Bicubic downscaling has to actually do something (so preprocessing
    differences show up), and no two patches may be identical (so a transposed
    patch order fails loudly rather than passing by symmetry).
    """
    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        for x in range(width):
            # Low-frequency gradient: distinguishes every patch position.
            r = (x * 255) // max(1, width - 1)
            g = (y * 255) // max(1, height - 1)
            # High-frequency ring pattern: survives only if the resize is real.
            d = math.hypot(x - width / 2.0, y - height / 2.0)
            b = int(127.5 * (1.0 + math.sin(d / 7.0)))
            # Asymmetric checker so x/y and row/column swaps are detectable.
            if ((x // 23) + 2 * (y // 31)) % 3 == 0:
                r = 255 - r
            px[x, y] = (r, g, b)
    return img


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, w, h in SPECS:
        path = os.path.join(OUT, f"{name}.png")
        render(w, h).save(path, "PNG", optimize=True)
        print(f"{path}  {w}x{h}  {os.path.getsize(path)} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
