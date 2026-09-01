"""MLX affine 量子化テンソルを float32 に戻す。

`checkpoint_io.Checkpoint` が返す生バイトを、`config.json` の `quantization`
(既定 + テンソルごとの上書き) に従って展開する。GPU も mlx も要らない。

MLX の格納規約 (実測(手元)。形から読める不変量 `bits × group = 32 × packed_last
/ scales_last` が 314 本すべてで成り立つ — `audit_checkpoint.py`):

- `X.weight`  : U32 `[..., cols * bits / 32]`。**低いビットが先**に詰まる
- `X.scales`  : BF16 `[..., cols / group_size]`
- `X.biases`  : BF16 同上 (`mode: affine`)
- 値は `w = q * scale + bias`
"""

from __future__ import annotations

import numpy as np

from checkpoint_io import Checkpoint


def unpack(packed: np.ndarray, bits: int) -> np.ndarray:
    """U32 の詰め物を最終軸に展開する。返るのは uint32 の量子化値。"""
    if 32 % bits:
        raise ValueError(f"32 を割り切らない bits: {bits}")
    per_word = 32 // bits
    shifts = (np.arange(per_word, dtype=np.uint32) * bits)
    q = (packed[..., None] >> shifts) & np.uint32((1 << bits) - 1)
    return q.reshape(*packed.shape[:-1], packed.shape[-1] * per_word)


def dequantize(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
               bits: int, group_size: int) -> np.ndarray:
    """`q * scale + bias` を float32 で返す。"""
    q = unpack(packed, bits).astype(np.float32)
    groups = scales.shape[-1]
    q = q.reshape(*q.shape[:-1], groups, group_size)
    out = q * scales[..., None] + biases[..., None]
    return out.reshape(*packed.shape[:-1], groups * group_size)


class Dequantizer:
    """1 つのチェックポイントから、テンソルを 1 本ずつ float32 で取り出す。

    量子化されていない (BF16 の) テンソルはそのまま広げて返すので、
    呼び手はビット幅を気にしなくてよい。**oQ4e-g64 は attention のビット幅が
    層ごとに違う** ([docs/qwen35moe/13 §4-2]) が、ここは 1 本ずつ引くので効かない。
    """

    def __init__(self, ckpt: Checkpoint):
        self.ckpt = ckpt
        quant = ckpt.config.get("quantization") or {}
        self.default = {k: v for k, v in quant.items() if not isinstance(v, dict)}
        self.per_tensor = {k: v for k, v in quant.items() if isinstance(v, dict)}

    def spec(self, prefix: str) -> dict:
        return self.per_tensor.get(prefix, self.default)

    def is_quantized(self, prefix: str) -> bool:
        ref = self.ckpt.tensors.get(prefix + ".weight")
        return ref is not None and ref.dtype == "U32"

    def _widen(self, name: str, index=None) -> np.ndarray:
        """BF16 を float32 に広げる。**添字は広げる前にかける** — `scales` は
        `[256, 512, 32]` のように大きく、1 枚だけ要るのに全体を広げると重い。"""
        ref = self.ckpt.tensors[name]
        arr = self.ckpt.raw(name)
        if index is not None:
            arr = arr[index]
        if ref.dtype == "BF16":
            return (arr.astype("<u4") << 16).view("<f4")
        return arr.astype(np.float32)

    def matrix(self, prefix: str, rows: np.ndarray | None = None) -> np.ndarray:
        """`prefix.weight` を float32 `[out, in]` で返す。

        `rows` を渡すと**その行だけ**を展開する (embed_tokens 用)。
        """
        name = prefix + ".weight"
        if not self.is_quantized(prefix):
            return self._widen(name, rows)
        spec = self.spec(prefix)
        if spec.get("mode", "affine") != "affine":
            raise NotImplementedError(f"未対応の mode: {spec}")
        packed = self.ckpt.raw(name)
        if rows is not None:
            packed = packed[rows]
        scales = self._widen(prefix + ".scales", rows)
        biases = self._widen(prefix + ".biases", rows)
        return dequantize(packed, scales, biases,
                          int(spec["bits"]), int(spec["group_size"]))

    def slice3(self, prefix: str, index: int) -> np.ndarray:
        """`[E, out, in]` の 3 階テンソルから 1 枚 (エキスパート 1 個) を返す。"""
        if not self.is_quantized(prefix):
            return self._widen(prefix + ".weight", index)
        spec = self.spec(prefix)
        packed = self.ckpt.raw(prefix + ".weight")[index]
        scales = self._widen(prefix + ".scales", index)
        biases = self._widen(prefix + ".biases", index)
        return dequantize(packed, scales, biases,
                          int(spec["bits"]), int(spec["group_size"]))

    def vector(self, name: str) -> np.ndarray:
        """BF16 のまま置かれているもの (norm / A_log / dt_bias / conv1d)。"""
        return self.ckpt.f32(name)
