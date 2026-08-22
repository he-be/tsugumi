#!/usr/bin/env python3
"""融合 `gate_up_proj` の分割順を、上流の実物で引き直す。

`graft_mtp_head.py` は供給側の `experts.gate_up_proj` `[E, 2*I, D]` を
**連続 `[gate(0:I); up(I:2I)]`** で割る。この向きが逆だと**落ちずに静かに
間違う**ので、根拠を毎回引けるようにしておく
([docs/qwen35moe/30 §3-2 (a)](../../docs/qwen35moe/30-MTP-HEAD-GRAFT.md))。

上流 Ornith は**本体 40 層も同じ融合形**なので、mlx-lm が既に割った手元の
`oQ4e-g64` と突き合わせられる。上流の layer-0 expert-0 (**4 MB**) だけを
HTTP Range で取り、3 つの仮説に相関を当てる。**ネットワークが要る。**

    Scripts/qwen35/verify_fused_split.py ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_io import Checkpoint  # noqa: E402
from mlx_quant import Dequantizer  # noqa: E402

REPO = "ornith-ai/Ornith-1.5-35B-A3B"
UPSTREAM = "model.language_model.layers.0.mlp.experts.gate_up_proj"
LOCAL = "language_model.model.layers.0.mlp.switch_mlp"


def get(url: str, begin: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    if begin is not None:
        req.add_header("Range", f"bytes={begin}-{end}")
    with urllib.request.urlopen(req, timeout=120) as fh:
        return fh.read()


def widen(raw: np.ndarray) -> np.ndarray:
    return (raw.astype("<u4") << 16).view("<f4")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="mlx-lm が既に割ったチェックポイント (oQ4e-g64)")
    ap.add_argument("--json", help="結果の書き出し先")
    args = ap.parse_args()

    base = f"https://huggingface.co/{REPO}/resolve/main/"
    index = json.loads(get(base + "model.safetensors.index.json"))
    shard = index["weight_map"][UPSTREAM]
    url = base + shard
    (header_size,) = struct.unpack("<Q", get(url, 0, 7))
    header = json.loads(get(url, 8, 8 + header_size - 1))
    spec = header[UPSTREAM]
    if spec["dtype"] != "BF16":
        raise SystemExit(f"上流が BF16 でない: {spec['dtype']}")
    experts, fused_dim, hidden = spec["shape"]
    inter = fused_dim // 2
    slab = fused_dim * hidden * 2
    begin = 8 + header_size + spec["data_offsets"][0]
    print(f"## 上流 {REPO} / {shard}")
    print(f"   {UPSTREAM} {spec['shape']} — expert 0 の {slab / 1e6:.1f} MB だけ取る")
    raw = np.frombuffer(get(url, begin, begin + slab - 1), dtype="<u2")
    if raw.size != fused_dim * hidden:
        raise SystemExit(f"取れたバイトが足りない: {raw.size}")
    up_fused = widen(raw).reshape(fused_dim, hidden)

    ckpt = Checkpoint(args.checkpoint)
    deq = Dequantizer(ckpt)
    gate = deq.slice3(LOCAL + ".gate_proj", 0)
    up = deq.slice3(LOCAL + ".up_proj", 0)
    if gate.shape != (inter, hidden):
        raise SystemExit(f"手元の形が合わない: {gate.shape} != {(inter, hidden)}")

    hypotheses = {
        "contiguous [gate; up]": (up_fused[:inter], up_fused[inter:]),
        "contiguous [up; gate]": (up_fused[inter:], up_fused[:inter]),
        "interleaved (stride 2)": (up_fused[0::2], up_fused[1::2]),
    }
    results = {}
    for name, (cand_gate, cand_up) in hypotheses.items():
        results[name] = (corr(cand_gate, gate), corr(cand_up, up))
        print(f"  {name:24s} gate {results[name][0]:+.5f}  up {results[name][1]:+.5f}")

    winner = max(results, key=lambda k: min(results[k]))
    print(f"\n  → {winner}")
    ckpt.close()
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"repo": REPO, "tensor": UPSTREAM, "shape": spec["shape"],
             "local": args.checkpoint, "correlations": results, "winner": winner},
            indent=2, ensure_ascii=False))
    ok = winner == "contiguous [gate; up]" and min(results[winner]) > 0.99
    if not ok:
        print("  ! graft_mtp_head.py の分割順と食い違う")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
