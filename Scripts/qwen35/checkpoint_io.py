"""safetensors のヘッダとテンソルを、丸ごと読まずに触るための最小の道具。

docs/qwen35moe の照合作業 (04-PHASES.md「次の一手」1〜3) 用。
GPU も mlx も要らない。numpy だけで動く。
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPE = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"),
    "U64": np.dtype("<u8"),
    "U32": np.dtype("<u4"),
    "U16": np.dtype("<u2"),
    "U8": np.dtype("<u1"),
    "BOOL": np.dtype("?"),
}


@dataclass(frozen=True)
class TensorRef:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int
    header_size: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
    return header, size + 8


class Checkpoint:
    """1 つのチェックポイント (ディレクトリ) の全シャードのヘッダ。"""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.tensors: dict[str, TensorRef] = {}
        self.shards: list[Path] = sorted(self.root.glob("*.safetensors"))
        if not self.shards:
            raise FileNotFoundError(f"safetensors が無い: {self.root}")
        self._maps: dict[Path, mmap.mmap] = {}
        # 打ち直し版は古いシャードに旧テンソルを残したまま、追加シャードで
        # 上書きする。どちらが正かは index の weight_map だけが知っている。
        index = self.index()
        weight_map = (index or {}).get("weight_map") or {}
        self.shadowed: list[tuple[str, str]] = []
        for shard in self.shards:
            header, base = read_header(shard)
            header.pop("__metadata__", None)
            for name, spec in header.items():
                owner = weight_map.get(name)
                if owner is not None and owner != shard.name:
                    self.shadowed.append((name, shard.name))
                    continue
                begin, end = spec["data_offsets"]
                self.tensors[name] = TensorRef(
                    name=name,
                    path=shard,
                    dtype=spec["dtype"],
                    shape=tuple(spec["shape"]),
                    begin=base + begin,
                    end=base + end,
                    header_size=base,
                )

    @property
    def config(self) -> dict:
        return json.loads((self.root / "config.json").read_text())

    def index(self) -> dict | None:
        path = self.root / "model.safetensors.index.json"
        return json.loads(path.read_text()) if path.exists() else None

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def __len__(self) -> int:
        return len(self.tensors)

    def _map(self, path: Path) -> mmap.mmap:
        got = self._maps.get(path)
        if got is None:
            fh = path.open("rb")
            got = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._maps[path] = got
        return got

    def raw(self, name: str) -> np.ndarray:
        """格納されているままの値 (BF16 は uint16 のまま返る)。"""
        ref = self.tensors[name]
        buf = self._map(ref.path)[ref.begin : ref.end]
        if ref.dtype == "BF16":
            return np.frombuffer(buf, dtype="<u2").reshape(ref.shape)
        return np.frombuffer(buf, dtype=_DTYPE[ref.dtype]).reshape(ref.shape)

    def f32(self, name: str) -> np.ndarray:
        """float32 に広げて返す。BF16 は上位 16 bit として展開 (無損失)。"""
        ref = self.tensors[name]
        arr = self.raw(name)
        if ref.dtype == "BF16":
            wide = np.zeros(arr.shape, dtype="<u4")
            wide |= arr.astype("<u4") << 16
            return wide.view("<f4")
        return arr.astype(np.float32)

    def close(self) -> None:
        for handle in self._maps.values():
            handle.close()
        self._maps.clear()
