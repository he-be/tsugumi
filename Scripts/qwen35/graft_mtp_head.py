#!/usr/bin/env python3
"""出荷版の MTP ヘッド 42 本を、学習済みのヘッドで差し替えたスナップショットを作る。

同梱の `mtp.*` は**乱数初期化のまま**である
([docs/qwen35moe/30 §1](../../docs/qwen35moe/30-MTP-HEAD-GRAFT.md))。
供給側は `shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY` の **BF16 19 本**で、
これが `oQ4e-g64` に既にある **42 本にちょうど 1:1 で写る** (同 §3-1)。

    平文 9 本 + 量子化 11 本 × (weight/scales/biases) = 42

変換で値をいじるのは 3 か所だけで、どれも同 §3-2 で実測済み:

- **融合 `gate_up_proj` の分割**は連続 `[gate(0:512); up(512:1024)]`
  (相関 0.995。`verify_fused_split.py` が上流の実物で毎回引き直せる)
- **norm は zero-centered gamma** (`w-1` を保存) なので **+1** して絶対値形にする
- **量子化**は差し替え先のテンソルごとの規約をそのまま使う
  (`config.json` の per-tensor override。attention と shared expert は 8b/g64、
  routed expert は既定の 4b/g64)。GPU は使わない — CPU ストリームで `mx.quantize`

出力は `bake_snapshot.py` と同じ手 — **元のシャードはハードリンクで持ってきて、
差し替え分だけを新しいシャード 1 枚に書き、`index.json` の `weight_map` を
書き換える**。増えるディスクは 503 MB (ヘッドの実体) だけで、元は無傷。

    Scripts/qwen35/graft_mtp_head.py ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 \
        ~/LLM/Ornith-1.5-35B-A3B-MTP-ONLY \
        ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa

`q_norm` の焼き込みは**このあと** `bake_snapshot.py` を通す (差し替えた
`mtp.…q_norm` にも 1/16 が要る)。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_io import Checkpoint, read_header  # noqa: E402
from mlx_quant import Dequantizer  # noqa: E402

OVERLAY_SHARD = "model-mtp-graft.safetensors"
GRAFT_MANIFEST = "mtp_graft_manifest.json"

DONOR_PREFIX = "mtp."
TARGET_PREFIX = "language_model.mtp."

# 供給側の名前 (DONOR_PREFIX を落とした残り) → 差し替え先 (TARGET_PREFIX を落とした残り)。
PLAIN = [
    "fc.weight",
    "layers.0.mlp.gate.weight",
]
# zero-centered gamma。読むときは 1+w なので、保存する側で +1 する。
NORMS = [
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
]
# 供給側も `.weight` を持つ。差し替え先は weight/scales/biases の 3 本になる。
QUANT = [
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj",
    "layers.0.mlp.shared_expert.gate_proj",
    "layers.0.mlp.shared_expert.up_proj",
    "layers.0.mlp.shared_expert.down_proj",
    "layers.0.mlp.shared_expert_gate",
]
# 融合されている routed expert。`[gate; up]` の順で真ん中から割る。
FUSED = "layers.0.mlp.experts.gate_up_proj"
FUSED_TARGETS = ("layers.0.mlp.switch_mlp.gate_proj", "layers.0.mlp.switch_mlp.up_proj")
ROUTED_DOWN = ("layers.0.mlp.experts.down_proj", "layers.0.mlp.switch_mlp.down_proj")


def bf16_from_f32(arr: np.ndarray) -> np.ndarray:
    """float32 → BF16 (round-to-nearest-even)。上位 16 bit を取るだけではない。"""
    bits = np.ascontiguousarray(arr, dtype="<f4").view("<u4")
    rounding = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding) >> 16).astype("<u2")


def widen(raw_u16: np.ndarray) -> np.ndarray:
    """BF16 の生 uint16 を float32 に広げる (無損失)。"""
    return (raw_u16.astype("<u4") << 16).view("<f4")


def stats(x: np.ndarray) -> tuple[float, float, float]:
    """mean / std / 尖度 (Fisher ではない生の 4 次モーメント比)。"""
    x = x.astype(np.float64).ravel()
    mu = x.mean()
    var = x.var()
    kurt = float(((x - mu) ** 4).mean() / (var * var)) if var > 0 else float("nan")
    return float(mu), float(np.sqrt(var)), kurt


DTYPE_NAME = {np.dtype("<u2"): "BF16", np.dtype("<u4"): "U32"}


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict) -> None:
    """BF16 (uint16 で持つ) と U32 を混ぜて 1 枚に書く。"""
    header: dict = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    offset = 0
    order: list[str] = []
    for name, arr in tensors.items():
        dtype = DTYPE_NAME.get(arr.dtype)
        if dtype is None:
            raise ValueError(f"{name}: 書けない dtype {arr.dtype}")
        nbytes = arr.nbytes
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
        order.append(name)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for name in order:
            fh.write(np.ascontiguousarray(tensors[name]).tobytes())


class Donor:
    """供給側 (単一 safetensors) を丸ごと読まずに触る。"""

    def __init__(self, root: Path):
        shards = sorted(root.glob("*.safetensors"))
        if len(shards) != 1:
            raise SystemExit(f"供給側は safetensors 1 枚のはず: {shards}")
        self.path = shards[0]
        header, self.base = read_header(self.path)
        header.pop("__metadata__", None)
        self.spec = header
        self.mm = np.memmap(self.path, dtype="<u1", mode="r")

    def raw(self, name: str) -> np.ndarray:
        spec = self.spec[name]
        if spec["dtype"] != "BF16":
            raise SystemExit(f"{name} が BF16 でない: {spec['dtype']}")
        begin, end = (self.base + spec["data_offsets"][0], self.base + spec["data_offsets"][1])
        return self.mm[begin:end].view("<u2").reshape(tuple(spec["shape"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="差し替え先の MLX チェックポイント (oQ4e-g64)")
    ap.add_argument("donor", help="学習済みヘッド (…-MTP-ONLY)")
    ap.add_argument("output", help="出力先")
    ap.add_argument("--force", action="store_true", help="出力先が既にあっても作り直す")
    ap.add_argument("--json", help="検証結果の書き出し先")
    args = ap.parse_args()

    import mlx.core as mx  # 量子化だけに使う。CPU ストリームで回す

    tgt = Checkpoint(args.target)
    donor = Donor(Path(args.donor).expanduser())
    index = tgt.index()
    if index is None:
        raise SystemExit("model.safetensors.index.json が無い")
    deq_old = Dequantizer(tgt)

    old_mtp = sorted(n for n in tgt.tensors if n.startswith(TARGET_PREFIX))
    if len(old_mtp) != 42:
        raise SystemExit(f"差し替え先の mtp が 42 本でない: {len(old_mtp)}")
    if len(donor.spec) != 19:
        raise SystemExit(f"供給側が 19 本でない: {len(donor.spec)}")

    print(f"## 供給側 {donor.path.name}: {len(donor.spec)} 本 BF16")
    print(f"## 差し替え先 {tgt.root.name}: mtp {len(old_mtp)} 本 / 全 {len(tgt.tensors)} 本")

    out_tensors: dict[str, np.ndarray] = {}
    report: list[dict] = []

    def target_shape(name: str) -> tuple[int, ...]:
        ref = tgt.tensors.get(name)
        if ref is None:
            raise SystemExit(f"差し替え先に無い名前: {name}")
        return ref.shape

    def put_plain(name: str, raw: np.ndarray, kind: str, before: np.ndarray) -> None:
        want = target_shape(name)
        if tuple(raw.shape) != want:
            raise SystemExit(f"{name}: 形が違う {raw.shape} != {want}")
        out_tensors[name] = np.ascontiguousarray(raw)
        new = widen(raw)
        m0, s0, k0 = stats(before)
        m1, s1, k1 = stats(new)
        report.append({"tensor": name, "kind": kind, "shape": list(want),
                       "old": {"mean": m0, "std": s0, "kurtosis": k0},
                       "new": {"mean": m1, "std": s1, "kurtosis": k1}})
        print(f"  {kind:6s} {name[len(TARGET_PREFIX):]:45s} "
              f"std {s0:.5f}→{s1:.5f}  尖度 {k0:6.2f}→{k1:7.2f}")

    # --- 平文 2 本 ---------------------------------------------------------
    for tail in PLAIN:
        name = TARGET_PREFIX + tail
        put_plain(name, donor.raw(DONOR_PREFIX + tail), "plain", tgt.f32(name))

    # --- norm 7 本 (+1) ----------------------------------------------------
    for tail in NORMS:
        name = TARGET_PREFIX + tail
        zero_centered = widen(donor.raw(DONOR_PREFIX + tail))
        absolute = bf16_from_f32(zero_centered + 1.0).reshape(zero_centered.shape)
        put_plain(name, absolute, "norm+1", tgt.f32(name))
        report[-1]["zero_centered_mean"] = float(zero_centered.mean())

    # --- 量子化 11 本 ------------------------------------------------------
    def quantize_into(prefix: str, raw: np.ndarray) -> None:
        spec = deq_old.spec(prefix)
        if spec.get("mode", "affine") != "affine":
            raise SystemExit(f"{prefix}: 未対応の mode {spec}")
        bits, group = int(spec["bits"]), int(spec["group_size"])
        if raw.shape[-1] % group:
            raise SystemExit(f"{prefix}: 入力次元 {raw.shape[-1]} が group {group} で割れない")
        # 往復誤差はここでは測らない。`mx.dequantize` は **BF16 で返す**ので
        # 復元値に BF16 の丸め (相対 0.0052) が乗り、量子化の代償を 1.2 倍に
        # 見せる。本ランタイムは float32/float16 で戻すので、書き出したバイトを
        # numpy の逆量子化器で読み直して測る (下の検算)。
        with mx.stream(mx.cpu):
            w = mx.array(np.ascontiguousarray(raw)).view(mx.bfloat16)
            q, scales, biases = mx.quantize(w, group_size=group, bits=bits)
            mx.eval(q, scales, biases)
        parts = {
            ".weight": np.array(q, copy=True).astype("<u4"),
            ".scales": np.array(scales.view(mx.uint16), copy=True).astype("<u2"),
            ".biases": np.array(biases.view(mx.uint16), copy=True).astype("<u2"),
        }
        for suffix, arr in parts.items():
            name = prefix + suffix
            want = target_shape(name)
            if tuple(arr.shape) != want:
                raise SystemExit(f"{name}: 形が違う {arr.shape} != {want}")
            out_tensors[name] = arr
        old = (deq_old.slice3(prefix, 0) if len(raw.shape) == 3
               else deq_old.matrix(prefix))
        new_ref = widen(raw[0] if len(raw.shape) == 3 else raw)
        m0, s0, k0 = stats(old)
        m1, s1, k1 = stats(new_ref)
        report.append({"tensor": prefix, "kind": f"{bits}b/g{group}",
                       "shape": list(raw.shape),
                       "old": {"mean": m0, "std": s0, "kurtosis": k0},
                       "new": {"mean": m1, "std": s1, "kurtosis": k1}})
        print(f"  {bits}b/g{group:<3d} {prefix[len(TARGET_PREFIX):]:45s} "
              f"std {s0:.5f}→{s1:.5f}  尖度 {k0:6.2f}→{k1:7.2f}")

    for tail in QUANT:
        quantize_into(TARGET_PREFIX + tail, donor.raw(DONOR_PREFIX + tail + ".weight"))

    fused = donor.raw(DONOR_PREFIX + FUSED)
    mid = fused.shape[-2] // 2
    if fused.shape[-2] != 2 * mid:
        raise SystemExit(f"融合の中間次元が偶数でない: {fused.shape}")
    for half, tail in zip((fused[:, :mid, :], fused[:, mid:, :]), FUSED_TARGETS):
        quantize_into(TARGET_PREFIX + tail, np.ascontiguousarray(half))
        del half
    del fused
    quantize_into(TARGET_PREFIX + ROUTED_DOWN[1], donor.raw(DONOR_PREFIX + ROUTED_DOWN[0]))

    if sorted(out_tensors) != old_mtp:
        missing = set(old_mtp) - set(out_tensors)
        extra = set(out_tensors) - set(old_mtp)
        raise SystemExit(f"42 本に写らなかった: 欠 {sorted(missing)} / 余 {sorted(extra)}")

    # --- 出力 --------------------------------------------------------------
    out = Path(args.output).expanduser()
    if out.exists() and not args.force:
        raise SystemExit(f"出力先が既にある: {out} (--force で作り直す)")
    out.mkdir(parents=True, exist_ok=True)

    linked, fallback = 0, []
    for entry in sorted(tgt.root.iterdir()):
        if entry.name in {"model.safetensors.index.json", OVERLAY_SHARD, GRAFT_MANIFEST}:
            continue
        target = out / entry.name
        if target.is_symlink() or target.exists():
            target.unlink()
        try:
            os.link(entry.resolve(), target)
        except OSError:
            os.symlink(entry.resolve(), target)
            fallback.append(entry.name)
        linked += 1

    write_safetensors(out / OVERLAY_SHARD, out_tensors, {
        "grafted_by": "Scripts/qwen35/graft_mtp_head.py",
        "target": str(tgt.root),
        "donor": "shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY",
        "donor_file": donor.path.name,
        "operation": "MTP head replaced (19 BF16 -> 42; norms +1; gate_up split [gate;up])",
    })

    weight_map = dict(index["weight_map"])
    replaced = {n: weight_map[n] for n in out_tensors}
    for name in out_tensors:
        weight_map[name] = OVERLAY_SHARD
    new_index = dict(index)
    new_index["weight_map"] = weight_map
    metadata = dict(index.get("metadata") or {})
    metadata["mtp_head"] = "shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY (12K KL distilled)"
    new_index["metadata"] = metadata
    (out / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2, ensure_ascii=False))

    # --- 読み直して検算 ----------------------------------------------------
    check = Checkpoint(out)
    if len(check.tensors) != len(tgt.tensors):
        raise SystemExit(f"本数が変わった: {len(check.tensors)} != {len(tgt.tensors)}")
    referenced = sum(check.tensors[n].nbytes for n in check.index()["weight_map"])
    if referenced != index["metadata"]["total_size"]:
        raise SystemExit(f"参照バイトが変わった: {referenced}")
    for name in old_mtp:
        if check.tensors[name].shape != tgt.tensors[name].shape:
            raise SystemExit(f"{name}: 形が変わった")
        if check.tensors[name].dtype != tgt.tensors[name].dtype:
            raise SystemExit(f"{name}: dtype が変わった")
        if check.tensors[name].path.name != OVERLAY_SHARD:
            raise SystemExit(f"{name}: 差し替えシャードを指していない")
        if not np.array_equal(check.raw(name), out_tensors[name].view(check.raw(name).dtype)):
            raise SystemExit(f"{name}: 書いたバイトと読み直しが違う")

    # 量子化の代償を、**書き出したバイト**を float32 に戻して測り直す。
    # 3 階テンソルはエキスパートを 1 枚ずつ回して、全 256 枚ぶんを足し込む。
    deq_new = Dequantizer(check)
    worst = 0.0
    for entry in report:
        if "b/g" not in entry["kind"]:
            continue
        prefix = entry["tensor"]
        tail = prefix[len(TARGET_PREFIX):]
        if tail in FUSED_TARGETS:
            src = donor.raw(DONOR_PREFIX + FUSED)
            mid = src.shape[-2] // 2
            src = src[:, :mid, :] if tail == FUSED_TARGETS[0] else src[:, mid:, :]
        elif tail == ROUTED_DOWN[1]:
            src = donor.raw(DONOR_PREFIX + ROUTED_DOWN[0])
        else:
            src = donor.raw(DONOR_PREFIX + tail + ".weight")
        if len(entry["shape"]) == 3:
            num = den = 0.0
            per_expert_std = []
            for e in range(src.shape[0]):
                got = deq_new.slice3(prefix, e).astype(np.float64)
                ref = widen(np.ascontiguousarray(src[e])).astype(np.float64)
                num += float(((got - ref) ** 2).sum())
                den += float((ref ** 2).sum())
                per_expert_std.append(float(got.std()))
            rel = float(np.sqrt(num / den))
            arr = np.asarray(per_expert_std)
            entry["expert_std_cv"] = float(arr.std() / arr.mean())
            old_std = np.asarray([float(deq_old.slice3(prefix, e).std())
                                  for e in range(src.shape[0])])
            entry["expert_std_cv_old"] = float(old_std.std() / old_std.mean())
        else:
            got = deq_new.matrix(prefix).astype(np.float64)
            ref = widen(np.ascontiguousarray(src)).astype(np.float64)
            rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        entry["relative_l2"] = rel
        worst = max(worst, rel)
        cv = entry.get("expert_std_cv")
        extra = "" if cv is None else (f"  expert 間 std の CV "
                                       f"{entry['expert_std_cv_old'] * 100:.2f}%"
                                       f"→{cv * 100:.2f}%")
        print(f"  {entry['kind']:8s} {tail:45s} 往復 L2 {rel:.5f}{extra}")

    (out / GRAFT_MANIFEST).write_text(json.dumps({
        "target": str(tgt.root),
        "donor": str(Path(args.donor).expanduser()),
        "donor_repo": "shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY",
        "overlay_shard": OVERLAY_SHARD,
        "tensors": sorted(out_tensors),
        "replaced_from": replaced,
        "norm_convention": "donor is zero-centered gamma; +1 applied",
        "fused_split": "[gate(0:mid); up(mid:)] on axis -2",
        "tensors_written": len(out_tensors),
        "report": report,
    }, indent=2, ensure_ascii=False))

    print(f"\n  ハードリンク {linked} 本 + 差し替えシャード 1 枚 ("
          f"{(out / OVERLAY_SHARD).stat().st_size / 1e6:.1f} MB) + 書き換えた index")
    if fallback:
        print(f"  ! {len(fallback)} 本は symlink になった (別ファイルシステム)")
    print(f"  読み直し: 42 本すべてバイト一致、参照バイト {referenced:,} は不変")
    print(f"  往復 L2 の最大 {worst:.5f} (4bit の routed expert)")
    print(f"  → {out}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    tgt.close()
    check.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
