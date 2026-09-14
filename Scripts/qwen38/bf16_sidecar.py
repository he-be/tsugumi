#!/usr/bin/env python3
"""DS4-IQ2 GGUF の GEMV で読む F32 テンソルを BF16 に詰めたサイドカーを書く (docs/qwen38/15 §2 W-4)。

対象は `ffn_gate_inp`・`ffn_gate_inp_shexp`・`ssm_alpha`・`ssm_beta` (MTP の blk.48 も含む、279 MiB)。
`f32_bf16_check.py` で全要素の仮数の下位 16 ビットが 0 と分かっているので、BF16 = float32 のビット列 >> 16 で損失は無く、
カーネルは `bit_cast<float>(uint(b) << 16)` で元の float32 をビット単位に戻す (`ggml_dense.metal`)。
norm 類・conv1d・`ssm_a` (変換器が −exp(A_log) を計算した値で BF16 では表せない) は生の float32 として読むので対象外。
元の GGUF は残す。書く前に下位 16 ビットを確かめ、1 要素でも 0 でなければ止まる。

    ~/LLM/venv/bin/python Scripts/qwen38/bf16_sidecar.py repack
    ~/LLM/venv/bin/python Scripts/qwen38/bf16_sidecar.py verify
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "LLM/llama.cpp/gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402

MODEL_DIR = Path.home() / "LLM/Qwen3.8-Flash-Next-DS4-IQ2"
DEFAULT_SRC = MODEL_DIR / "Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf"
DEFAULT_OUT = MODEL_DIR / "bf16/Qwen3.8-Flash-Next-GatesBF16.gguf"
FAMILIES = re.compile(r"^blk\.\d+\.(ffn_gate_inp|ffn_gate_inp_shexp|ssm_alpha|ssm_beta)\.weight$")
F_NOCACHE = 48
ALIGN = 16384
GGML_BF16 = 30


def pread_all(fd: int, n: int, off: int) -> np.ndarray:
    buf = np.empty(n, np.uint8)
    mv = memoryview(buf)
    got = 0
    while got < n:
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            raise IOError(f"short read at {off + got}")
        got += k
    return buf


def gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def targets(src: Path):
    r = GGUFReader(src, "r")
    ts = [t for t in r.tensors if FAMILIES.match(t.name)]
    assert ts and all(t.tensor_type == GGMLQuantizationType.F32 for t in ts)
    return ts


def repack(src: Path, out: Path) -> int:
    ts = targets(src)
    kv = [
        ("general.alignment", 4, struct.pack("<I", ALIGN)),
        ("tsugumi.bf16.of_f32", 7, struct.pack("<B", 1)),
        ("tsugumi.bf16.source", 8, gguf_string(src.name)),
    ]
    h = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(ts), len(kv)))
    for key, typ, val in kv:
        h += gguf_string(key) + struct.pack("<I", typ) + val
    off = 0
    for t in ts:
        dims = [int(x) for x in t.shape]
        h += gguf_string(t.name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        h += struct.pack("<IQ", GGML_BF16, off)
        off += (t.n_bytes // 2 + ALIGN - 1) // ALIGN * ALIGN
    h += b"\0" * ((-len(h)) % ALIGN)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    fin = os.open(src, os.O_RDONLY)
    fout = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    fcntl.fcntl(fin, F_NOCACHE, 1)
    fcntl.fcntl(fout, F_NOCACHE, 1)
    os.write(fout, bytes(h))
    t0 = time.time()
    n_in = n_out = 0
    for t in ts:
        bits = pread_all(fin, t.n_bytes, t.data_offset).view(np.uint32)
        bad = int(((bits & 0xFFFF) != 0).sum())
        if bad:
            print(f"{t.name}: {bad} elements have low 16 bits set, not BF16-exact; stopping")
            os.close(fout)
            os.unlink(tmp)
            return 1
        half = (bits >> 16).astype(np.uint16)
        data = half.tobytes()
        os.write(fout, data + b"\0" * ((-len(data)) % ALIGN))
        n_in += t.n_bytes
        n_out += len(data)
    os.fsync(fout)
    os.close(fout)
    os.close(fin)
    os.replace(tmp, out)
    print(f"{len(ts)} tensors: {n_in / 2**20:.1f} MiB F32 -> {n_out / 2**20:.1f} MiB BF16, "
          f"file {os.path.getsize(out) / 2**20:.1f} MiB, {time.time() - t0:.1f} s -> {out}")
    return 0


def verify(src: Path, out: Path) -> int:
    ts = targets(src)
    r2 = GGUFReader(out, "r")
    side = {t.name: t for t in r2.tensors}
    assert len(side) == len(ts)
    fin = os.open(src, os.O_RDONLY)
    fs = os.open(out, os.O_RDONLY)
    fcntl.fcntl(fin, F_NOCACHE, 1)
    fcntl.fcntl(fs, F_NOCACHE, 1)
    digest = hashlib.sha256()
    elements = 0
    for t in ts:
        s = side[t.name]
        assert s.tensor_type == GGMLQuantizationType.BF16 and list(s.shape) == list(t.shape), t.name
        bits = pread_all(fin, t.n_bytes, t.data_offset).view(np.uint32)
        half = pread_all(fs, s.n_bytes, s.data_offset).view(np.uint16)
        back = half.astype(np.uint32) << 16
        if not np.array_equal(back, bits):
            print(f"{t.name}: MISMATCH")
            return 1
        digest.update(half.tobytes())
        elements += bits.size
    print(f"all {len(ts)} tensors: BF16 << 16 == F32 bits for {elements:,} elements; sha256 {digest.hexdigest()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["repack", "verify"])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    return repack(args.src, args.out) if args.cmd == "repack" else verify(args.src, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
