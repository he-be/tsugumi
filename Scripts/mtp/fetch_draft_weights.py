#!/usr/bin/env python3
"""Fetch the Gemma 4 MTP drafter weights for the Python-side reference.

The Swift installer (`TsugumiRepack --add-draft`) has its own pinned
fetch path; this script exists for `dump_draft_fixtures.py`, which needs the
same 236 MB checkpoint locally to run the float32 reference
(`docs/mtp/04-PHASES.md` §5).

The whole checkpoint is one 236 MB `model.safetensors`, so unlike the vision
fetch there is no header surgery to do: stream the file, then verify it is the
pinned conversion by (a) size, (b) tensor count and (c) the SHA-256 of the
three tensors Google's BF16 assistant release also contains byte-for-byte
(`docs/mtp/01-CHECKPOINT.md` §1 — the same provenance evidence
`DraftModelSource.pin.provenanceTensors` checks at install time).

Output: scratch/mtp-weights/draft.safetensors (+ draft_manifest.json)
"""

import hashlib
import json
import os
import struct
import sys
import time

import requests

REPO = "mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit"
REVISION = "bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
OUT_DIR = "scratch/mtp-weights"

# From DraftModelSource.pin / docs/mtp/11-M1-RESULTS.md §1, checked rather
# than trusted.
EXPECTED_TENSORS = 94
EXPECTED_BYTES = 236_124_704
EXPECTED_PAYLOAD_BYTES = 236_114_440
EXPECTED_INDEX_SHA256 = "ab54b0e481714d358d800ad10366f585841e678f982be3274ea6660e9bedd3eb"
PROVENANCE = {
    "model.norm.weight":
        "3bf68317e6d4e33e29a3d019eb744d52d5fb3ebf5dca52513e878e1f845f9047",
    "model.layers.0.input_layernorm.weight":
        "fbd6be5ad58d336c6bd398bd161db25bcc13bf2e07e50db8c6b55d8f400959eb",
    "model.layers.3.self_attn.q_norm.weight":
        "07101aaa0ac6e6df4b28cc3b209f04edaa9c277fc5037b50525478e3e60df15b",
}


def get(url: str, headers=None, tries: int = 5) -> requests.Response:
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=headers or {}, timeout=120, stream=True)
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
    out_path = os.path.join(OUT_DIR, "draft.safetensors")

    print("model.safetensors.index.json ...")
    index_raw = get(f"{BASE}/model.safetensors.index.json").content
    index_sha = hashlib.sha256(index_raw).hexdigest()
    print(f"  sha256 {index_sha}")
    if index_sha != EXPECTED_INDEX_SHA256:
        print(f"  FAIL: expected {EXPECTED_INDEX_SHA256}", file=sys.stderr)
        return 1
    weight_map = json.loads(index_raw)["weight_map"]
    shards = set(weight_map.values())
    if shards != {"model.safetensors"}:
        print(f"  FAIL: tensors span {shards}", file=sys.stderr)
        return 1

    if os.path.exists(out_path) and os.path.getsize(out_path) == EXPECTED_BYTES:
        print(f"{out_path} already present with the pinned size; skipping download")
        with open(out_path, "rb") as f:
            body = f.read()
    else:
        print(f"model.safetensors ({EXPECTED_BYTES} B) ...")
        fetched = 0
        digest = hashlib.sha256()
        with open(out_path, "wb") as f:
            for chunk in get(f"{BASE}/model.safetensors").iter_content(chunk_size=1 << 20):
                f.write(chunk)
                digest.update(chunk)
                fetched += len(chunk)
                if fetched % (64 << 20) < (1 << 20):
                    print(f"  {fetched / 1e9:.3f} GB")
        if fetched != EXPECTED_BYTES:
            print(f"  FAIL: fetched {fetched} != {EXPECTED_BYTES}", file=sys.stderr)
            return 1
        print(f"  file sha256 {digest.hexdigest()}")
        with open(out_path, "rb") as f:
            body = f.read()

    header_len = struct.unpack("<Q", body[:8])[0]
    header = json.loads(body[8 : 8 + header_len])
    header.pop("__metadata__", None)
    data_start = 8 + header_len
    payload = len(body) - data_start
    print(f"  header {header_len} B, {len(header)} tensors, payload {payload} B")
    if len(header) != EXPECTED_TENSORS or payload != EXPECTED_PAYLOAD_BYTES:
        print(f"  FAIL: expected {EXPECTED_TENSORS} tensors / "
              f"{EXPECTED_PAYLOAD_BYTES} B payload", file=sys.stderr)
        return 1

    for name, expected_sha in PROVENANCE.items():
        meta = header[name]
        lo, hi = meta["data_offsets"]
        actual = hashlib.sha256(body[data_start + lo : data_start + hi]).hexdigest()
        ok = actual == expected_sha
        print(f"  provenance {name}: {'match' if ok else 'MISMATCH'}")
        if not ok:
            print(f"    expected {expected_sha}\n    actual   {actual}", file=sys.stderr)
            return 1

    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "index_sha256": index_sha,
        "tensor_count": len(header),
        "payload_bytes": payload,
    }
    with open(os.path.join(OUT_DIR, "draft_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
