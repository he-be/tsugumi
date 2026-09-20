#!/usr/bin/env python3
"""Prism Bonsai の GGUF (PQ2_0 型 142 と `prism.hadamard.*`) を読む。

gguf-py は型 142 を知らないので `GGUFReader` がヘッダの段で落ちる。ここでは GGUF を自前で読み、
`Scripts/qwen38_27b/reference_forward.py` の `_worker_rows` が期待する形 (行ごとの生バイト、
逆量子化して float32 [rows, cols]) でテンソルを返す。算式の出どころは PrismML-Eng/llama.cpp の
`prism-v7` ブランチ (読むだけ):

- PQ2_0: `ggml-common.h` の `block_pq2_0` = fp16 の d 1 個 + 128 重み分の 2 bit (34 B / 128 重み)。
  値は `ggml-quants.c` の `dequantize_row_pq2_0` で `(q - 1) * d`、q は 00=-1 / 01=0 / 10=+1。
- Hadamard: `llama-model.cpp` が block_size × block_size の行列を
  `H[r][c] = (-1)^popcount(r & c) / sqrt(block_size)` で作り (正規直交・対称)、
  `llama-graph.cpp` の `build_lora_mm` が折り込み済みの重み w ごとに活性 x を
  `x' = H · (s ⊙ perm(x))` に変換してから `w · x'` を計算する。
  `token_embd` だけは逆で、行を引いた後に `h = s ⊙ (H z)` で元の基底に戻す。
  `s` は入力幅ごとの ±1 ベクトル (`prism.hadamard.sign_values` を `sign_widths` の順に連結したもの)。
  `gdn_v_grouped` が真なら `ssm_out` の入力だけ、変換の前に [hd, nk, rep] → [hd, rep, nk] に並べ替える。
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

GGML_F32, GGML_F16, GGML_BF16, GGML_PQ2_0 = 0, 1, 30, 142

# 型 -> (ブロックの重み数, ブロックのバイト数)。ここに無い型は gguf-py の表と逆量子化に回す
# (ISTA の IQ3_S-mtp を同じ Reader で読むため)。
BLOCK = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_PQ2_0: (128, 34),
}


def _gguf_py():
    import sys
    path = str(Path.home() / "LLM/llama.cpp/gguf-py")
    if path not in sys.path:
        sys.path.insert(0, path)
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    from gguf.quants import dequantize
    return GGML_QUANT_SIZES, GGMLQuantizationType, dequantize


def block_shape(ggml_type: int) -> tuple:
    if ggml_type in BLOCK:
        return BLOCK[ggml_type]
    sizes, enum, _ = _gguf_py()
    return sizes[enum(ggml_type)]

_SCALARS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


class Tensor:
    """GGUF の 1 テンソル。`shape` は GGUF の並び (ne[0] が入力側)。"""

    def __init__(self, name, ne, ggml_type, offset, mm, data_start):
        self.name = name
        self.ne = ne
        self.ggml_type = ggml_type
        self.n_rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        self.n_cols = ne[0]
        wpb, bpb = block_shape(ggml_type)
        assert self.n_cols % wpb == 0, (name, self.n_cols, wpb)
        self.row_bytes = self.n_cols // wpb * bpb
        self.offset = data_start + offset
        self._mm = mm

    @property
    def raw(self):
        """[n_rows, row_bytes] の uint8 ビュー (コピーしない)。"""
        n = self.n_rows * self.row_bytes
        return np.frombuffer(self._mm, dtype=np.uint8, count=n, offset=self.offset).reshape(self.n_rows, self.row_bytes)

    def rows(self, r0: int, r1: int) -> np.ndarray:
        """行 [r0, r1) を float32 [r1 - r0, n_cols] にして返す。"""
        return dequant_rows(self.raw[r0:r1], self.ggml_type)

    def __repr__(self):
        return f"Tensor({self.name}, ne={self.ne}, type={self.ggml_type})"


def dequant_rows(blk: np.ndarray, ggml_type: int) -> np.ndarray:
    """行ごとの生バイト [rows, row_bytes] を float32 [rows, cols] にする。"""
    rows = blk.shape[0]
    if ggml_type == GGML_F32:
        return blk.view(np.float32).reshape(rows, -1)
    if ggml_type == GGML_F16:
        return blk.view(np.float16).astype(np.float32).reshape(rows, -1)
    if ggml_type == GGML_BF16:
        return (blk.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(rows, -1)
    if ggml_type == GGML_PQ2_0:
        return dequant_pq2_0(blk)
    _, enum, dequantize = _gguf_py()
    return dequantize(np.ascontiguousarray(blk), enum(ggml_type)).reshape(rows, -1).astype(np.float32)


def dequant_pq2_0(blk: np.ndarray) -> np.ndarray:
    """block_pq2_0 (fp16 の d + 2 bit × 128) を float32 にする。`(q - 1) * d`。"""
    rows, row_bytes = blk.shape
    assert row_bytes % 34 == 0, row_bytes
    nb = row_bytes // 34
    b = blk.reshape(rows, nb, 34)
    d = np.ascontiguousarray(b[:, :, :2]).view(np.float16).astype(np.float32)      # [rows, nb, 1]
    qs = b[:, :, 2:]                                                                # [rows, nb, 32]
    q = np.stack([(qs >> s) & np.uint8(3) for s in (0, 2, 4, 6)], axis=-1)          # [rows, nb, 32, 4]
    q = q.reshape(rows, nb, 128).astype(np.float32) - np.float32(1)
    return (q * d).reshape(rows, nb * 128)


class Reader:
    """GGUF v3 をメタデータとテンソル表だけ読んで mmap を持つ。"""

    def __init__(self, path: Path | str):
        self.path = str(path)
        self._f = open(self.path, "rb")
        import mmap
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        d = self._mm
        assert d[:4] == b"GGUF", self.path
        self._o = 4
        version = self._scalar("<I")
        assert version == 3, version
        n_tensors = self._scalar("<Q")
        n_kv = self._scalar("<Q")
        self.fields = {}
        for _ in range(n_kv):
            key = self._string()
            self.fields[key] = self._value(self._scalar("<I"))
        infos = []
        for _ in range(n_tensors):
            name = self._string()
            nd = self._scalar("<I")
            ne = [self._scalar("<Q") for _ in range(nd)]
            ggml_type = self._scalar("<I")
            offset = self._scalar("<Q")
            infos.append((name, ne, ggml_type, offset))
        align = int(self.fields.get("general.alignment", 32))
        data_start = (self._o + align - 1) // align * align
        self.tensors = {n: Tensor(n, ne, t, off, self._mm, data_start) for n, ne, t, off in infos}

    # --- 読み取り ---
    def _scalar(self, fmt):
        v = struct.unpack_from(fmt, self._mm, self._o)[0]
        self._o += struct.calcsize(fmt)
        return v

    def _string(self):
        n = self._scalar("<Q")
        v = self._mm[self._o:self._o + n].decode(errors="replace")
        self._o += n
        return v

    def _value(self, t):
        if t == 8:
            return self._string()
        if t == 9:
            et = self._scalar("<I")
            n = self._scalar("<Q")
            if et in _SCALARS:  # 数値の配列は一度に読む
                fmt = _SCALARS[et]
                a = np.frombuffer(self._mm, dtype=np.dtype(fmt[1:]).newbyteorder("<"), count=n, offset=self._o).copy()
                self._o += n * struct.calcsize(fmt)
                return a
            return [self._value(et) for _ in range(n)]
        return self._scalar(_SCALARS[t])

    def close(self):
        self._mm.close()
        self._f.close()


# --- Hadamard ------------------------------------------------------------------------

class Hadamard:
    """`prism.hadamard.*` の読み出しと、活性への適用。"""

    def __init__(self, fields: dict):
        self.block = 0
        self.forward = {}   # 重み名 -> 入力幅 (折り込み済み)
        self.inverse = {}   # 重み名 -> 入力幅 (引いた後に戻す表)
        self.signs = {}     # 入力幅 -> float32 [幅] の ±1
        self.gdn_v_grouped = False
        version = fields.get("prism.hadamard.version")
        if version is None:
            return
        assert int(version) == 1, version
        assert fields["prism.hadamard.transform"] == "normalized-sylvester-walsh-hadamard"
        assert fields["prism.hadamard.axis"] == "input-last-dimension"
        assert fields["prism.hadamard.sign_mode"] == "explicit"
        self.block = int(fields["prism.hadamard.block_size"])
        assert self.block > 0 and self.block & (self.block - 1) == 0, self.block
        widths = [int(w) for w in fields["prism.hadamard.sign_widths"]]
        values = np.asarray(fields["prism.hadamard.sign_values"], dtype=np.int32)
        assert values.size == sum(widths), (values.size, widths)
        at = 0
        for w in widths:
            s = values[at:at + w]
            assert np.all(np.abs(s) == 1), w
            self.signs[w] = s.astype(np.float32)
            at += w
        self.forward = {n: self.block for n in fields["prism.hadamard.weight_names"]}
        for n in fields.get("prism.hadamard.inverse_weight_names", []):
            self.inverse[n] = self.block
        self.gdn_v_grouped = bool(fields.get("prism.hadamard.gdn_v_grouped", False))

    def __bool__(self):
        return self.block > 0

    def matrix(self) -> np.ndarray:
        """llama-model.cpp が作る block × block の行列 (検査用。実際は fwht を使う)。"""
        n = self.block
        r = np.arange(n, dtype=np.uint32)[:, None]
        c = np.arange(n, dtype=np.uint32)[None, :]
        parity = np.bitwise_count(r & c) & 1 if hasattr(np, "bitwise_count") else _popcount(r & c) & 1
        return np.where(parity == 1, -1.0, 1.0).astype(np.float32) / np.float32(np.sqrt(n))

    def rotate(self, x: np.ndarray, width: int, perm: tuple | None = None) -> np.ndarray:
        """`build_lora_mm` と同じ順序で x [T, width] -> H · (s ⊙ perm(x))。"""
        assert x.shape[-1] == width, (x.shape, width)
        if perm is not None:
            # ggml の [hd, nk, rep] (ne[0] が最速) は numpy では (rep, nk, hd)。
            # ggml_permute(x, 0, 2, 1, 3) は [hd, rep, nk] = numpy (nk, rep, hd) にする。
            hd, nk, rep = perm
            x = np.ascontiguousarray(x.reshape(-1, rep, nk, hd).transpose(0, 2, 1, 3)).reshape(-1, width)
        s = self.signs.get(width)
        if s is not None:
            x = x * s
        return fwht(x, self.block)

    def unrotate(self, z: np.ndarray, width: int) -> np.ndarray:
        """`token_embd` の行に対する `h = s ⊙ (H z)`。"""
        h = fwht(z, self.block)
        s = self.signs.get(width)
        return h * s if s is not None else h


def _popcount(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.uint32)
    out = np.zeros_like(a)
    for i in range(32):
        out += (a >> np.uint32(i)) & np.uint32(1)
    return out


def fwht(x: np.ndarray, n: int) -> np.ndarray:
    """最後の軸を n ごとに区切って正規直交の Walsh-Hadamard 変換 (Sylvester 順)。"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    shape = x.shape
    assert shape[-1] % n == 0, (shape, n)
    y = x.reshape(-1, n).copy()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :].copy()
        b = y[:, :, 1, :].copy()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        y = y.reshape(-1, n)
        h *= 2
    y *= np.float32(1.0 / np.sqrt(n))
    return y.reshape(shape)


# --- ssm_out の並べ替え ---------------------------------------------------------------

def gdn_v_perm(ne0: int, n_v: int, n_k: int) -> tuple:
    """`llama-model.cpp` の perm_hd / perm_nk / perm_rep。ssm_out の入力だけに効く。"""
    assert n_k > 0 and n_v > 0 and n_v % n_k == 0 and ne0 % n_v == 0, (ne0, n_v, n_k)
    return ne0 // n_v, n_k, n_v // n_k
