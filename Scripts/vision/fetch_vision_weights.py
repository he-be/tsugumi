#!/usr/bin/env python3
"""Range-fetch the Gemma 4 vision tower weights into one local safetensors file.

The QAT repository's shard 1 is 46 GB but the 356 vision tensors inside it are
1.15 GB. We read the safetensors header, keep only the `vision_tower` /
`embed_vision` entries, and pull their byte ranges with HTTP Range requests, so
nothing but those 1.15 GB ever crosses the wire.

Output: scratch/vision-weights/vision.safetensors (+ vision_manifest.json)
"""

import hashlib
import json
import os
import struct
import sys
import time

import requests

REPO = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
REVISION = "f1e06dc520982d9b9edd76859fdb7ab209449949"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
OUT_DIR = "scratch/vision-weights"

# Expected totals from PLAN_VISION §1-1, checked rather than trusted.
EXPECTED_TENSORS = 356
EXPECTED_BYTES = 1_145_588_832
EXPECTED_INDEX_SHA256 = "907826a6e46ff454272bd6db1fee629d5531a2303be22986d825a0871d7dc7a7"


def is_vision(name: str) -> bool:
    return name.startswith("model.vision_tower.") or name.startswith("model.embed_vision.")


def get(url: str, headers=None, tries: int = 5) -> requests.Response:
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=headers or {}, timeout=120)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001 - retry every transport failure
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1}/{tries} after {exc}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("index.json ...")
    index_raw = get(f"{BASE}/model.safetensors.index.json").content
    index_sha = hashlib.sha256(index_raw).hexdigest()
    print(f"  sha256 {index_sha}")
    if index_sha != EXPECTED_INDEX_SHA256:
        print(f"  FAIL: expected {EXPECTED_INDEX_SHA256}", file=sys.stderr)
        return 1
    weight_map = json.loads(index_raw)["weight_map"]

    shards = {weight_map[n] for n in weight_map if is_vision(n)}
    if shards != {"model-00001-of-00002.safetensors"}:
        print(f"  FAIL: vision tensors span {shards}", file=sys.stderr)
        return 1
    shard = shards.pop()
    shard_url = f"{BASE}/{shard}"

    print(f"{shard} header ...")
    head8 = get(shard_url, headers={"Range": "bytes=0-7"}).content
    header_len = struct.unpack("<Q", head8)[0]
    header_raw = get(shard_url, headers={"Range": f"bytes=8-{8 + header_len - 1}"}).content
    header = json.loads(header_raw)
    header.pop("__metadata__", None)
    data_start = 8 + header_len
    print(f"  header {header_len} B, {len(header)} tensors, data starts at {data_start}")

    entries = sorted(
        ((n, m) for n, m in header.items() if is_vision(n)),
        key=lambda kv: kv[1]["data_offsets"][0],
    )
    total = sum(m["data_offsets"][1] - m["data_offsets"][0] for _, m in entries)
    print(f"  vision: {len(entries)} tensors, {total} B")
    if len(entries) != EXPECTED_TENSORS or total != EXPECTED_BYTES:
        print(
            f"  FAIL: expected {EXPECTED_TENSORS} tensors / {EXPECTED_BYTES} B",
            file=sys.stderr,
        )
        return 1
    for _, m in entries:
        if m["dtype"] != "BF16":
            print(f"  FAIL: non-BF16 vision tensor {m}", file=sys.stderr)
            return 1

    # Rebuild a standalone safetensors: same tensors, offsets rebased to 0.
    out_header = {}
    cursor = 0
    for name, meta in entries:
        size = meta["data_offsets"][1] - meta["data_offsets"][0]
        out_header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    out_header_raw = json.dumps(out_header, separators=(",", ":")).encode()
    pad = (-len(out_header_raw)) % 8
    out_header_raw += b" " * pad

    # Coalesce adjacent source ranges so 356 tensors become a handful of requests.
    runs = []
    for name, meta in entries:
        lo, hi = meta["data_offsets"]
        if runs and runs[-1][1] == lo and (hi - runs[-1][0]) <= 256 << 20:
            runs[-1][1] = hi
        else:
            runs.append([lo, hi])
    print(f"  {len(runs)} coalesced ranges")

    out_path = os.path.join(OUT_DIR, "vision.safetensors")
    fetched = 0
    digest = hashlib.sha256()
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(out_header_raw)))
        f.write(out_header_raw)
        for i, (lo, hi) in enumerate(runs):
            a, b = data_start + lo, data_start + hi - 1
            body = get(shard_url, headers={"Range": f"bytes={a}-{b}"}).content
            if len(body) != hi - lo:
                print(f"  FAIL: short range {len(body)} != {hi - lo}", file=sys.stderr)
                return 1
            f.write(body)
            digest.update(body)
            fetched += len(body)
            print(f"  [{i + 1}/{len(runs)}] {fetched / 1e9:.3f} / {total / 1e9:.3f} GB")

    if fetched != total:
        print(f"  FAIL: fetched {fetched} != {total}", file=sys.stderr)
        return 1

    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "shard": shard,
        "index_sha256": index_sha,
        "tensor_count": len(entries),
        "payload_bytes": total,
        "payload_sha256": digest.hexdigest(),
    }
    with open(os.path.join(OUT_DIR, "vision_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {out_path}")
    print(f"payload sha256 {digest.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
