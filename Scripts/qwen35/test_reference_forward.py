"""`reference_forward.py` を**上流実装そのもの**に対して検証する。

PLAN_VISION §6 の教訓 (「カーネルのバグと丸め誤差を分離できる検証系を先に作る」)
をそのまま持ち込む。実物の 35B で数字を取る前に、**小さい乱数モデル**を
`transformers` の `Qwen3_5MoeForCausalLM` (CPU / float32) で 1 回流し、
同じ重みを MLX の並びに写して参照器に流し、logits が一致することを見る。

ここだけは torch と transformers を使う (`scratch/vision-venv`)。
`reference_forward.py` 自体は numpy だけで動く。

    scratch/vision-venv/bin/python Scripts/qwen35/test_reference_forward.py
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference_forward import ReferenceModel, State  # noqa: E402

# `+1` を焼くもの (`Qwen3_5MoeRMSNorm`)。`linear_attn.norm` は `RMSNormGated` なので焼かない。
BAKE_PLUS_ONE = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def save_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        arr = np.ascontiguousarray(array, dtype=np.float32)
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes]}
        offset += arr.nbytes
        blobs.append(arr.tobytes())
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for chunk in blobs:
            fh.write(chunk)


def export(model, config: dict, root: Path) -> None:
    """torch の state_dict を MLX の並び・規約で書き出す。"""
    out: dict[str, np.ndarray] = {}
    for name, param in model.state_dict().items():
        value = param.detach().to(torch.float32).numpy()
        if name.startswith("model."):
            name = "language_model.model." + name[len("model."):]
        elif name.startswith("lm_head."):
            name = "language_model.lm_head." + name[len("lm_head."):]
        if name.endswith(BAKE_PLUS_ONE) or name == "language_model.model.norm.weight":
            value = value + 1.0                      # MLX 変換側が焼くぶん
        if name.endswith("linear_attn.conv1d.weight"):
            value = value.transpose(0, 2, 1)         # [C,1,K] → [C,K,1]
        if name.endswith("mlp.experts.gate_up_proj"):
            mid = value.shape[-2] // 2
            base = name[: -len("experts.gate_up_proj")] + "switch_mlp."
            out[base + "gate_proj.weight"] = value[:, :mid, :]
            out[base + "up_proj.weight"] = value[:, mid:, :]
            continue
        if name.endswith("mlp.experts.down_proj"):
            base = name[: -len("experts.down_proj")] + "switch_mlp."
            out[base + "down_proj.weight"] = value
            continue
        out[name] = value
    save_safetensors(root / "model.safetensors", out)
    (root / "config.json").write_text(json.dumps(config, indent=1))


def tiny_config() -> dict:
    return {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 512,
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "layer_types": ["linear_attention", "linear_attention",
                        "linear_attention", "full_attention"],
        "rope_parameters": {"type": "default", "rope_theta": 10000000.0,
                            "partial_rotary_factor": 0.25},
    }


def main() -> int:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig)
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM)

    torch.manual_seed(20260821)
    raw = tiny_config()
    config = Qwen3_5MoeTextConfig(**raw)
    config._attn_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(config).to(torch.float32).eval()

    # 既定の初期化は norm が 0、experts が空なので、全部を乱数で埋め直す。
    with torch.no_grad():
        for name, param in model.named_parameters():
            scale = 0.5 if "norm" in name or name.endswith("dt_bias") else 0.05
            param.copy_(torch.randn(param.shape, dtype=torch.float32) * scale)
        model.model.layers[0].linear_attn.A_log.copy_(
            torch.log(torch.rand(raw["linear_num_value_heads"]) * 16))

    tokens = torch.tensor([[3, 17, 42, 5, 99, 7, 61, 2, 23, 88, 4, 13]])
    with torch.no_grad():
        want = model(tokens, use_cache=False).logits[0].to(torch.float32).numpy()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        export(model, raw, root)
        reference = ReferenceModel(root)
        got = reference.forward(tokens[0].numpy(), State(reference.cfg),
                                logits_for="all")

    # 2 本目: 状態の持ち越し。prefill 8 + 1 トークンずつ 4 回が、一括と一致するか。
    # conv 状態・再帰状態・KV の 3 つが正しく繰り越されないとここで落ちる。
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        export(model, raw, root)
        reference = ReferenceModel(root)
        state = State(reference.cfg)
        ids = tokens[0].numpy()
        step = reference.forward(ids[:8], state, logits_for="all")
        rows = [step]
        for i in range(8, ids.shape[0]):
            rows.append(reference.forward(ids[i:i + 1], state, logits_for="all"))
        incremental = np.concatenate(rows, axis=0)

    scale = float(np.abs(want).max())
    delta = float(np.abs(got - want).max())
    top1 = float(np.mean(np.argmax(got, -1) == np.argmax(want, -1)))
    print(f"  logits: 最大 |値| = {scale:.4f}  最大 |差| = {delta:.3e}"
          f"  (相対 {delta / scale:.2e})")
    print(f"  top-1 一致率: {top1 * 100:.1f}%")

    # 単精度の行列積を 4 層ぶん積み上げた差。1e-4 は float32 の丸めの範囲。
    split = float(np.abs(incremental - got).max())
    print(f"  状態の持ち越し (8+1×4 対 一括): 最大 |差| = {split:.3e}"
          f"  (相対 {split / scale:.2e})")

    ok = delta / scale < 1e-4 and top1 == 1.0 and split / scale < 1e-4
    print("  判定:", "一致" if ok else "**不一致**")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
