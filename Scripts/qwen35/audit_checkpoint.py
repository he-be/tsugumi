#!/usr/bin/env python3
"""MLX 形式の Ornith チェックポイントを、上流 bf16 と突き合わせて監査する。

docs/qwen35moe/04-PHASES.md「次の一手」の 1〜3 を機械化したもの:

  1. norm 規約 (`1+w` が焼かれているか、`linear_attn.norm` は素か)
  2. `conv1d.weight` の軸順 (上流 [C,1,4] / MLX [C,4,1])
  3. router (`mlp.gate`) のビット幅

ついでに (a) 量子化形式の赤リスト、(b) 区画別のバイト、(c) `expertStride` の
16 KiB 整列も見る。**GPU を使わない。**mlx も torch も要らない。

    Scripts/qwen35/audit_checkpoint.py ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 \
        --bf16 ~/LLM/Ornith-1.5-35B-A3B-bf16-partial --json scratch/qwen35/oq4e-g64.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_io import Checkpoint  # noqa: E402

EXPERT_STRIDE = 1_769_472  # 16 KiB × 108 — 01-MODEL.md §5-2
ALLOWED_BITS = {4, 8}
ALLOWED_GROUPS = {32, 64}


def upstream_name(name: str) -> str:
    """MLX 側の名前を、上流 bf16 リポジトリの名前へ写す。"""
    if name.startswith("language_model.model."):
        return "model.language_model." + name[len("language_model.model.") :]
    if name.startswith("language_model.mtp."):
        return "mtp." + name[len("language_model.mtp.") :]
    if name.startswith("language_model.lm_head."):
        return "lm_head." + name[len("language_model.lm_head.") :]
    return name


def section_of(name: str) -> str:
    if ".switch_mlp." in name:
        return "mtp_experts" if ".mtp." in name else "routed_experts"
    if ".mtp." in name:
        return "mtp_core"
    if "visual" in name or "vision" in name:
        return "vision"
    if "embed_tokens" in name or "lm_head" in name:
        return "embed_lm_head"
    return "core"


def quant_groups(ckpt: Checkpoint) -> dict[str, dict]:
    """`.weight` が U32 のものを (weight, scales, biases) の組にまとめる。"""
    out: dict[str, dict] = {}
    for name, ref in ckpt.tensors.items():
        if not name.endswith(".weight") or ref.dtype != "U32":
            continue
        prefix = name[: -len(".weight")]
        scales = ckpt.tensors.get(prefix + ".scales")
        biases = ckpt.tensors.get(prefix + ".biases")
        out[prefix] = {
            "packed_shape": ref.shape,
            "scales_shape": scales.shape if scales else None,
            "scales_dtype": scales.dtype if scales else None,
            "has_biases": biases is not None,
        }
    return out


def quant_config_map(config: dict) -> tuple[dict, dict]:
    q = config.get("quantization") or {}
    default = {
        "bits": q.get("bits"),
        "group_size": q.get("group_size"),
        "mode": q.get("mode"),
    }
    per_tensor = {k: v for k, v in q.items() if isinstance(v, dict)}
    return default, per_tensor


def audit_format(ckpt: Checkpoint, report: dict) -> None:
    default, per_tensor = quant_config_map(ckpt.config)
    groups = quant_groups(ckpt)
    census: Counter = Counter()
    red: list[dict] = []
    mismatch: list[dict] = []
    for prefix, info in sorted(groups.items()):
        spec = per_tensor.get(prefix, default)
        bits, group = spec.get("bits"), spec.get("group_size")
        packed_last = info["packed_shape"][-1]
        scales_last = info["scales_shape"][-1] if info["scales_shape"] else None
        # 形から読める不変量: bits × group = 32 × packed_last / scales_last
        product = None
        if scales_last:
            product = 32 * packed_last / scales_last
        if bits and group and product is not None and abs(product - bits * group) > 1e-9:
            mismatch.append(
                {"tensor": prefix, "config": [bits, group], "from_shape": product}
            )
        census[(bits, group, spec.get("mode"))] += 1
        if bits not in ALLOWED_BITS or group not in ALLOWED_GROUPS:
            red.append({"tensor": prefix, "bits": bits, "group_size": group})
    report["format"] = {
        "default": default,
        "per_tensor_entries": len(per_tensor),
        "quantized_tensors": len(groups),
        "census": {f"{b}bit_g{g}_{m}": n for (b, g, m), n in sorted(census.items(), key=str)},
        "red_list": red,
        "shape_config_mismatch": mismatch,
    }
    print("## 量子化形式")
    print(f"  既定             : {default}")
    print(f"  量子化テンソル   : {len(groups)} 本 (config の個別指定 {len(per_tensor)} 件)")
    for key, count in report["format"]["census"].items():
        print(f"    {key:24s} {count:5d}")
    print(f"  赤リスト (bits∉{{4,8}} or group∉{{32,64}}): {len(red)} 本")
    for item in red[:10]:
        print(f"    - {item}")
    print(f"  形と config の不一致: {len(mismatch)} 本")
    for item in mismatch[:10]:
        print(f"    - {item}")


def audit_inventory(ckpt: Checkpoint, report: dict) -> None:
    by_section: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for name, ref in ckpt.tensors.items():
        slot = by_section[section_of(name)]
        slot[0] += 1
        slot[1] += ref.nbytes
    total = sum(v[1] for v in by_section.values())
    report["inventory"] = {
        "root": str(ckpt.root),
        "shards": [p.name for p in ckpt.shards],
        "tensors": len(ckpt),
        "shadowed_by_index": len(getattr(ckpt, "shadowed", [])),
        "referenced_bytes": total,
        "sections": {k: {"tensors": v[0], "bytes": v[1]} for k, v in sorted(by_section.items())},
    }
    print("## 持ち物")
    print(f"  {ckpt.root}")
    print(f"  シャード {len(ckpt.shards)} 本 / テンソル {len(ckpt)} 本 "
          f"/ 影 {len(getattr(ckpt, 'shadowed', []))} 本")
    print(f"  参照バイト合計 {total:,} B ({total / 1e9:.2f} GB)")
    for key, val in sorted(by_section.items(), key=lambda kv: -kv[1][1]):
        share = 100 * val[1] / total
        print(f"    {key:16s} {val[0]:5d} 本 {val[1]:>15,} B  {share:5.1f}%")


def audit_expert_stride(ckpt: Checkpoint, report: dict) -> None:
    roles = ("gate_proj", "up_proj", "down_proj")
    prefix = "language_model.model.layers.0.mlp.switch_mlp."
    total = 0
    detail = {}
    for role in roles:
        per_role = 0
        for suffix in (".weight", ".scales", ".biases"):
            ref = ckpt.tensors.get(prefix + role + suffix)
            if ref is None:
                continue
            experts = ref.shape[0]
            per_role += ref.nbytes // experts
            detail[role + suffix] = {"shape": list(ref.shape), "bytes": ref.nbytes}
        total += per_role
    report["expert_stride"] = {
        "bytes_per_expert": total,
        "expected": EXPERT_STRIDE,
        "matches": total == EXPERT_STRIDE,
        "page_16kib_remainder": total % 16384,
        "tensors": detail,
    }
    mark = "一致" if total == EXPERT_STRIDE else "不一致"
    print("## expertStride")
    print(f"  エキスパート 1 個 = {total:,} B ({mark}, 期待 {EXPERT_STRIDE:,})"
          f" / 16 KiB 余り {total % 16384}")


NORM_FAMILIES = {
    "input_layernorm": r"^language_model\.model\.layers\.\d+\.input_layernorm\.weight$",
    "post_attention_layernorm": r"^language_model\.model\.layers\.\d+\.post_attention_layernorm\.weight$",
    "self_attn.q_norm": r"^language_model\.model\.layers\.\d+\.self_attn\.q_norm\.weight$",
    "self_attn.k_norm": r"^language_model\.model\.layers\.\d+\.self_attn\.k_norm\.weight$",
    "linear_attn.norm": r"^language_model\.model\.layers\.\d+\.linear_attn\.norm\.weight$",
    "model.norm": r"^language_model\.model\.norm\.weight$",
    "mtp.q_norm": r"^language_model\.mtp\.layers\.\d+\.self_attn\.q_norm\.weight$",
    "mtp.input_layernorm": r"^language_model\.mtp\.layers\.\d+\.input_layernorm\.weight$",
}


def audit_norms(ckpt: Checkpoint, bf16: Checkpoint | None, report: dict) -> None:
    print("## norm 規約 (`1+w` が焼かれているか)")
    header = f"  {'family':26s} {'本':>3s} {'mean':>9s} {'min':>9s} {'max':>9s}"
    if bf16:
        header += f" {'照合':>3s} {'mean 差':>10s} {'判定':>8s}"
    print(header)
    out = {}
    for family, pattern in NORM_FAMILIES.items():
        names = sorted(n for n in ckpt.tensors if re.match(pattern, n))
        if not names:
            continue
        values = [ckpt.f32(n) for n in names]
        means = np.array([float(v.mean()) for v in values])
        row = {
            "tensors": len(names),
            "mean": float(means.mean()),
            "min": float(min(float(v.min()) for v in values)),
            "max": float(max(float(v.max()) for v in values)),
        }
        line = (f"  {family:26s} {len(names):3d} {row['mean']:9.4f} "
                f"{row['min']:9.4f} {row['max']:9.4f}")
        if bf16:
            diffs = []
            for name, val in zip(names, values):
                other = upstream_name(name)
                if other not in bf16:
                    continue
                diffs.append(float(val.mean() - bf16.f32(other).mean()))
            if diffs:
                delta = float(np.mean(diffs))
                # +1 が焼かれていれば差は 1、素のままならビット一致で 0。
                verdict = "焼済み" if abs(delta - 1.0) < 5e-3 else (
                    "素のまま" if abs(delta) < 5e-4 else "不明")
                row.update(compared=len(diffs), mean_delta=delta, verdict=verdict)
                line += f" {len(diffs):3d} {delta:10.6f} {verdict:>8s}"
            else:
                line += f" {0:3d} {'-':>10s} {'照合なし':>8s}"
        print(line)
        out[family] = row
    report["norms"] = out


def audit_conv1d(ckpt: Checkpoint, bf16: Checkpoint | None, report: dict) -> None:
    names = sorted(n for n in ckpt.tensors if n.endswith("linear_attn.conv1d.weight"))
    if not names:
        report["conv1d"] = {"tensors": 0}
        return
    shapes = Counter(tuple(ckpt.tensors[n].shape) for n in names)
    print("## conv1d.weight の軸順")
    for shape, count in shapes.items():
        print(f"  MLX 側 {list(shape)} × {count} 本")
    out = {"tensors": len(names), "shapes": {str(list(s)): c for s, c in shapes.items()}}
    if bf16:
        exact = mismatched = missing = 0
        upstream_shapes: Counter = Counter()
        for name in names:
            other = upstream_name(name)
            if other not in bf16:
                missing += 1
                continue
            upstream_shapes[tuple(bf16.tensors[other].shape)] += 1
            here = ckpt.raw(name).squeeze()
            there = bf16.raw(other).squeeze()
            if here.shape != there.shape:
                mismatched += 1
                continue
            exact += int(np.array_equal(here, there))
        for shape, count in upstream_shapes.items():
            print(f"  上流 bf16  {list(shape)} × {count} 本")
        print(f"  squeeze 後にビット一致: {exact} 本 / 形の不一致 {mismatched} 本"
              f" / 上流に無い {missing} 本")
        out.update(
            upstream_shapes={str(list(s)): c for s, c in upstream_shapes.items()},
            bitwise_equal_after_squeeze=exact,
            shape_mismatch=mismatched,
            missing_upstream=missing,
        )
    report["conv1d"] = out


def audit_router(ckpt: Checkpoint, report: dict) -> None:
    default, per_tensor = quant_config_map(ckpt.config)
    rows: dict[str, Counter] = defaultdict(Counter)
    for name, ref in ckpt.tensors.items():
        for role in ("mlp.gate", "mlp.shared_expert_gate"):
            if not name.endswith(role + ".weight"):
                continue
            prefix = name[: -len(".weight")]
            if ref.dtype != "U32":
                rows[role][f"{ref.dtype} (非量子化)"] += 1
                continue
            spec = per_tensor.get(prefix, default)
            rows[role][f"{spec.get('bits')}bit_g{spec.get('group_size')}"] += 1
    print("## router / shared_expert_gate のビット幅")
    for role, counter in sorted(rows.items()):
        for key, count in sorted(counter.items()):
            print(f"  {role:24s} {key:16s} {count:4d} 本")
    report["router"] = {k: dict(v) for k, v in rows.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="MLX 形式のチェックポイント (ディレクトリ)")
    ap.add_argument("--bf16", help="上流 bf16 の抽出 (照合に使う)")
    ap.add_argument("--json", help="結果を JSON でも書き出す")
    args = ap.parse_args()

    ckpt = Checkpoint(args.checkpoint)
    bf16 = Checkpoint(args.bf16) if args.bf16 else None
    report: dict = {"checkpoint": str(ckpt.root), "bf16": str(bf16.root) if bf16 else None}

    audit_inventory(ckpt, report)
    print()
    audit_format(ckpt, report)
    print()
    audit_expert_stride(ckpt, report)
    print()
    audit_norms(ckpt, bf16, report)
    print()
    audit_conv1d(ckpt, bf16, report)
    print()
    audit_router(ckpt, report)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n→ {path}")
    ckpt.close()
    if bf16:
        bf16.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
