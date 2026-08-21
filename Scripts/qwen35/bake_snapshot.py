#!/usr/bin/env python3
"""`q_norm` に `head_dim ** -0.5` を焼き込んだスナップショットを作る。

docs/qwen35moe/03-DESIGN.md §1-1 の焼き込み。**どちらの候補でも要る。**
q_norm は RoPE の直前にあり、RoPE は回転 (ノルム保存) なので順序を入れ替えてよい。
焼いておくと attention の `scale` を 1.0 に固定でき、`scale == 1.0` ゲートの
tensorops 経路の余地が残る。

**本リポジトリの repack は値を作らない** (バイトをそのまま運ぶ) ので、
焼き込みは Python 側でやる ([02 §3](../../docs/qwen35moe/02-CHECKPOINTS.md))。
元のシャードは 20 GB あるので触らない。`oQ4e-g64` が打ち直しでやったのと同じ手
— **上書き用のシャードを 1 枚足して `index.json` の `weight_map` を書き換える** —
を使う。出力は元ファイルへの symlink + 差分シャード + 書き換えた index なので、
増えるディスクは数十 KB で済む。

`head_dim = 256` なので係数は `1/16 = 2^-4`。**BF16 に対して 2 のべき乗の乗算は
指数を 4 だけ下げるだけで、仮数は 1 bit も動かない (丸めが起きない)。**
本スクリプトは float32 で掛けた結果と、指数を直接引いた結果が
**ビット一致する**ことを毎回検査する。

    Scripts/qwen35/bake_snapshot.py ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 \
        ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-baked
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_io import Checkpoint  # noqa: E402

OVERLAY_SHARD = "model-baked-q_norm.safetensors"
BAKE_MANIFEST = "bake_manifest.json"
Q_NORM = re.compile(r"\.self_attn\.q_norm\.weight$")


def bf16_from_f32(arr: np.ndarray) -> np.ndarray:
    """float32 → BF16 (round-to-nearest-even)。上位 16 bit を取るだけではない。"""
    bits = arr.astype("<f4").view("<u4")
    rounding = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding) >> 16).astype("<u2")


def scale_bf16_by_power_of_two(raw: np.ndarray, exponent: int) -> np.ndarray:
    """BF16 の指数フィールドを直接動かす (照合用の独立実装)。"""
    sign = raw & 0x8000
    exp = (raw >> 7) & 0xFF
    mant = raw & 0x7F
    zero = (exp == 0) & (mant == 0)          # ±0 はそのまま通す
    if np.any(((exp == 0) & ~zero) | (exp == 0xFF)):
        raise ValueError("非正規化数か Inf/NaN があるので指数を直接動かせない")
    shifted = exp.astype(np.int32) + exponent
    if np.any((shifted <= 0) & ~zero) or np.any(shifted >= 0xFF):
        raise ValueError("指数がはみ出す")
    moved = (sign | (shifted.astype("<u2") << 7) | mant).astype("<u2")
    return np.where(zero, sign, moved).astype("<u2")


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict) -> None:
    header: dict = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        payload = arr.tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        blobs.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(encoded)) % 8
    encoded += b" " * pad
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for blob in blobs:
            fh.write(blob)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="MLX 形式のチェックポイント")
    ap.add_argument("output", help="焼き込み済みスナップショットの出力先")
    ap.add_argument("--head-dim", type=int, default=256,
                    help="q_norm に掛ける係数を head_dim**-0.5 で決める (既定 256 → 1/16)")
    ap.add_argument("--force", action="store_true", help="出力先が既にあっても作り直す")
    args = ap.parse_args()

    src = Checkpoint(args.source)
    out = Path(args.output).expanduser()
    index = src.index()
    if index is None:
        raise SystemExit("model.safetensors.index.json が無い")

    exponent = -int(round(np.log2(args.head_dim) / 2))
    factor = float(2.0 ** exponent)
    if abs(factor - args.head_dim ** -0.5) > 1e-12:
        raise SystemExit(f"head_dim {args.head_dim} は 2 のべき乗の係数にならない")

    names = sorted(n for n in src.tensors if Q_NORM.search(n))
    if not names:
        raise SystemExit("q_norm が 1 本も無い")
    print(f"## 焼き込み: q_norm × {factor} (= 2^{exponent}, head_dim {args.head_dim})")

    baked: dict[str, np.ndarray] = {}
    for name in names:
        ref = src.tensors[name]
        if ref.dtype != "BF16":
            raise SystemExit(f"{name} が BF16 でない: {ref.dtype}")
        raw = src.raw(name)
        # 経路 1: float32 で掛けて BF16 に丸め直す
        rounded = bf16_from_f32(src.f32(name) * factor)
        # 経路 2: 指数を直接動かす (2 のべき乗なので丸めは起きないはず)
        shifted = scale_bf16_by_power_of_two(raw, exponent)
        if not np.array_equal(rounded, shifted):
            raise SystemExit(f"{name}: 2 経路が食い違った — 丸めが起きている")
        baked[name] = rounded.reshape(ref.shape)
        print(f"  {name:60s} {list(ref.shape)}  平均 "
              f"{src.f32(name).mean():.5f} → {(src.f32(name).mean() * factor):.5f}")

    if out.exists() and not args.force:
        raise SystemExit(f"出力先が既にある: {out} (--force で作り直す)")
    out.mkdir(parents=True, exist_ok=True)

    # 元のファイルは symlink で持ってくる。index.json だけ書き換える。
    linked = 0
    for entry in sorted(src.root.iterdir()):
        if entry.name in {"model.safetensors.index.json", OVERLAY_SHARD, BAKE_MANIFEST}:
            continue
        target = out / entry.name
        if target.is_symlink() or target.exists():
            target.unlink()
        os.symlink(entry.resolve(), target)
        linked += 1

    write_safetensors(out / OVERLAY_SHARD, baked, {
        "baked_by": "Scripts/qwen35/bake_snapshot.py",
        "source": str(src.root),
        "operation": f"q_norm.weight *= 2^{exponent} (head_dim**-0.5, head_dim={args.head_dim})",
        "lossless": "yes (power of two on BF16: exponent only)",
    })

    weight_map = dict(index["weight_map"])
    replaced = {n: weight_map[n] for n in baked}
    for name in baked:
        weight_map[name] = OVERLAY_SHARD
    new_index = dict(index)
    new_index["weight_map"] = weight_map
    metadata = dict(index.get("metadata") or {})
    metadata["baked"] = f"q_norm.weight *= 2^{exponent}"
    new_index["metadata"] = metadata
    (out / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2, ensure_ascii=False))

    (out / BAKE_MANIFEST).write_text(json.dumps({
        "source": str(src.root),
        "factor": factor,
        "exponent": exponent,
        "head_dim": args.head_dim,
        "tensors": sorted(baked),
        "replaced_from": replaced,
        "overlay_shard": OVERLAY_SHARD,
        "lossless": True,
    }, indent=2, ensure_ascii=False))

    # 出来上がりを読み直して比を検査する。
    check = Checkpoint(out)
    worst = 0.0
    for name in baked:
        ratio = check.f32(name) / np.where(src.f32(name) == 0, np.nan, src.f32(name))
        finite = ratio[np.isfinite(ratio)]
        worst = max(worst, float(np.max(np.abs(finite - factor))))
    print(f"\n  symlink {linked} 本 + 差分シャード 1 枚 + 書き換えた index")
    print(f"  読み直し検査: 比の {factor} からの最大ずれ {worst:.3e}")
    print(f"  → {out}")
    src.close()
    check.close()
    return 0 if worst == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
