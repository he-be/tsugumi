#!/usr/bin/env python3
"""DS4-IQ2 GGUF の `ffn_down_exps` (Q2_K、640 列を 768 列に pad) から pad を落としたサイドカーを書く (docs/qwen38/15 §2 W-1)。

Q2_K のブロック (84 B) は scales[16] @0 | qs[64] @16 | d @80 | dmin @82。重み w (0..255) の逆量子化に効くのは
d・dmin・scales[w // 16]・qs[w // 4] だけ。行の 3 ブロック目 (列 512..767) のうち列 640..767 = 重み 128..255 は
acts が常に 0 なので、scales[8..15] (8 B) と qs[32..63] (32 B) は値が何であっても出力に効かない。
3 ブロック目を scales[0..7] | qs[0..31] | d | dmin の 44 B に詰め、1 行 252 → 212 B にする。

サイドカーは GGUF v3 の容器で、テンソル `blk.{L}.ffn_down_exps.weight` を型 I8・形 [212, 2560, 512] で持つ
(GGML の Q2_K では 640 列を表せないため、バイト列として持ち、並びはメタデータ `tsugumi.down.*` で宣言する)。
元の GGUF は残す (`reference_forward.py`・`expert_kernel_fixture.py` は元を読む)。

    ~/LLM/venv/bin/python Scripts/qwen38/down_sidecar.py repack   # 14.8 GiB 読んで 12.4 GiB 書く (層ごと、F_NOCACHE)
    ~/LLM/venv/bin/python Scripts/qwen38/down_sidecar.py verify   # 全層のバイト比較 + 第 24 層の逆量子化の一致
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "LLM/llama.cpp/gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

MODEL_DIR = Path.home() / "LLM/Qwen3.8-Flash-Next-DS4-IQ2"
DEFAULT_SRC = MODEL_DIR / "Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf"
DEFAULT_OUT = MODEL_DIR / "down/Qwen3.8-Flash-Next-Q2KDown640.gguf"

F_NOCACHE = 48
QK_K = 256
BLOCK = 84
TAIL = 44
LOGICAL_IN = 640
PHYSICAL_IN = 768
ALIGN = 16384
GGML_I8 = 24

# 1 行 252 B のうち残すバイト: ブロック 0・1 の全部、ブロック 2 の scales[0..7]・qs[0..31]・d・dmin。
KEEP = np.concatenate([
    np.arange(0, 2 * BLOCK),
    2 * BLOCK + np.arange(0, 8),
    2 * BLOCK + 16 + np.arange(0, 32),
    2 * BLOCK + 80 + np.arange(0, 4),
])
ROW_OUT = len(KEEP)
assert ROW_OUT == 2 * BLOCK + TAIL == 212


def nocache(fd: int) -> None:
    fcntl.fcntl(fd, F_NOCACHE, 1)


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


def down_tensors(src: Path):
    r = GGUFReader(src, "r")
    out = []
    for t in r.tensors:
        # blk.48 (MTP) の down は MXFP4 で pad 無し (640 は 32 で割り切れる)。対象は Q2_K の 48 層だけ。
        if t.name.endswith(".ffn_down_exps.weight") and t.tensor_type == GGMLQuantizationType.Q2_K:
            layer = int(t.name.split(".")[1])
            out.append((layer, t))
    out.sort()
    layers = [l for l, _ in out]
    field = r.fields.get("ds4.qwen4.down.logical_input")
    assert field is not None and int(field.contents()) == LOGICAL_IN
    field = r.fields.get("ds4.qwen4.down.physical_input")
    assert field is not None and int(field.contents()) == PHYSICAL_IN
    return r, out, layers


def gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def header(tensors: list[tuple[str, tuple[int, ...], int]], src: Path) -> bytes:
    kv = [
        ("general.alignment", 4, struct.pack("<I", ALIGN)),
        ("tsugumi.down.row_bytes", 4, struct.pack("<I", ROW_OUT)),
        ("tsugumi.down.logical_input", 4, struct.pack("<I", LOGICAL_IN)),
        ("tsugumi.down.physical_input", 4, struct.pack("<I", PHYSICAL_IN)),
        ("tsugumi.down.layout", 8, gguf_string(
            "q2_K rows over 640 inputs: blocks 0..1 whole (84 B), block 2 = scales[0..7] qs[0..31] d dmin (44 B)")),
        ("tsugumi.down.source", 8, gguf_string(src.name)),
    ]
    h = bytearray()
    h += b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, typ, val in kv:
        h += gguf_string(key) + struct.pack("<I", typ) + val
    off = 0
    for name, dims, nbytes in tensors:
        h += gguf_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        h += struct.pack("<IQ", GGML_I8, off)
        off += (nbytes + ALIGN - 1) // ALIGN * ALIGN
    h += b"\0" * ((-len(h)) % ALIGN)
    return bytes(h)


def repack(src: Path, out: Path) -> int:
    _, downs, layers = down_tensors(src)
    assert layers == list(range(48)), layers
    shapes = {tuple(int(x) for x in t.shape) for _, t in downs}
    assert shapes == {(PHYSICAL_IN, 2560, 512)}, shapes
    assert all(t.tensor_type == GGMLQuantizationType.Q2_K for _, t in downs)
    rows = 2560 * 512
    specs = [(t.name, (ROW_OUT, 2560, 512), rows * ROW_OUT) for _, t in downs]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    fin = os.open(src, os.O_RDONLY)
    fout = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    nocache(fin)
    nocache(fout)
    h = header(specs, src)
    os.write(fout, h)
    t0 = time.time()
    done_in = done_out = 0
    for (layer, t), (_, _, nbytes) in zip(downs, specs):
        src_bytes = rows * 3 * BLOCK
        assert t.n_bytes == src_bytes, (t.name, t.n_bytes)
        raw = pread_all(fin, src_bytes, t.data_offset).reshape(rows, 3 * BLOCK)
        packed = np.ascontiguousarray(raw[:, KEEP])
        del raw
        buf = packed.tobytes() + b"\0" * ((-nbytes) % ALIGN)
        del packed
        n = os.write(fout, buf)
        assert n == len(buf)
        done_in += src_bytes
        done_out += nbytes
        dt = time.time() - t0
        print(f"layer {layer:2d}: in {done_in / 2**30:6.2f} GiB  out {done_out / 2**30:6.2f} GiB  "
              f"{dt:6.1f} s  {done_in / 2**30 / dt:.2f} GiB/s", flush=True)
        del buf
    os.fsync(fout)
    os.close(fout)
    os.close(fin)
    os.replace(tmp, out)
    print(f"wrote {out} ({os.path.getsize(out) / 2**30:.2f} GiB) in {time.time() - t0:.1f} s")
    return 0


def dequant_tail_layout(rows212: np.ndarray) -> np.ndarray:
    """212 B の行 [n, 212] を [n, 640] の float32 に。`dequantize_row_q2_K` の定義をブロックごとに直接書き、gguf-py とは独立。"""
    n = rows212.shape[0]
    out = np.empty((n, LOGICAL_IN), np.float32)
    starts = [(0, 0, 256), (BLOCK, 0, 256), (2 * BLOCK, 0, 128)]
    for bi, (base, _, nw) in enumerate(starts):
        nsc = nw // 16
        if bi < 2:
            sc = rows212[:, base:base + 16]
            qs = rows212[:, base + 16:base + 80]
            dd = rows212[:, base + 80:base + 84]
        else:
            sc = rows212[:, base:base + 8]
            qs = rows212[:, base + 8:base + 40]
            dd = rows212[:, base + 40:base + 44]
        d = np.frombuffer(np.ascontiguousarray(dd[:, 0:2]).tobytes(), np.float16).astype(np.float32)
        m = np.frombuffer(np.ascontiguousarray(dd[:, 2:4]).tobytes(), np.float16).astype(np.float32)
        for s in range(nsc):
            dl = d * (sc[:, s] & 0xF)
            ml = m * (sc[:, s] >> 4)
            byte0 = 32 * (s // 8) + 16 * (s % 2)
            shift = 2 * ((s % 8) // 2)
            q = (qs[:, byte0:byte0 + 16] >> shift) & 3
            col = 256 * bi + 16 * s
            out[:, col:col + 16] = dl[:, None] * q.astype(np.float32) - ml[:, None]
    return out


def verify(src: Path, out: Path, layer_check: int, experts: list[int]) -> int:
    _, downs, _ = down_tensors(src)
    r2 = GGUFReader(out, "r")
    side = {t.name: t for t in r2.tensors}
    assert int(r2.fields["tsugumi.down.row_bytes"].contents()) == ROW_OUT
    rows = 2560 * 512
    fin = os.open(src, os.O_RDONLY)
    fside = os.open(out, os.O_RDONLY)
    nocache(fin)
    nocache(fside)
    t0 = time.time()
    digest = hashlib.sha256()
    for layer, t in downs:
        s = side[t.name]
        assert tuple(int(x) for x in s.shape) == (ROW_OUT, 2560, 512) and s.tensor_type == GGMLQuantizationType.I8
        assert s.n_bytes == rows * ROW_OUT
        raw = pread_all(fin, rows * 3 * BLOCK, t.data_offset).reshape(rows, 3 * BLOCK)
        got = pread_all(fside, rows * ROW_OUT, s.data_offset).reshape(rows, ROW_OUT)
        same = np.array_equal(raw[:, KEEP], got)
        digest.update(got.tobytes())
        if not same:
            bad = np.nonzero((raw[:, KEEP] != got).any(axis=1))[0]
            print(f"layer {layer}: MISMATCH in {len(bad)} rows, first row {bad[0]}")
            return 1
        if layer == layer_check:
            raw3 = raw.reshape(512, 2560, 3 * BLOCK)
            got3 = got.reshape(512, 2560, ROW_OUT)
            for e in experts:
                ref = dequantize(raw3[e], GGMLQuantizationType.Q2_K).reshape(2560, PHYSICAL_IN)
                mine = dequant_tail_layout(got3[e])
                eq = np.array_equal(ref[:, :LOGICAL_IN], mine)
                # 自前の逆量子化が gguf-py と同じ定義であることを、元の 252 B 行 (ブロック 2 も全部) でも確かめる
                full = dequant_tail_layout(raw3[e][:, KEEP])
                print(f"layer {layer} expert {e}: cols 0..639 of gguf-py(original) == dequant(212 B) "
                      f"{eq}, max |diff| {np.abs(ref[:, :LOGICAL_IN] - mine).max():.3g}, "
                      f"pad cols 640..767 of original nonzero: {int((ref[:, LOGICAL_IN:] != 0).sum())}, "
                      f"self-consistent {np.array_equal(full, mine)}")
                if not eq:
                    return 1
        print(f"layer {layer:2d}: bytes equal  {time.time() - t0:6.1f} s", flush=True)
    print(f"all {len(downs)} layers equal; sha256 of the packed tensors {digest.hexdigest()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["repack", "verify"])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--experts", default="0,17,255,511")
    args = ap.parse_args()
    if args.cmd == "repack":
        return repack(args.src, args.out)
    return verify(args.src, args.out, args.layer, [int(x) for x in args.experts.split(",")])


if __name__ == "__main__":
    raise SystemExit(main())
