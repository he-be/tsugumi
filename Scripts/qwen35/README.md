# Scripts/qwen35 — Qwen3.5-MoE (Ornith) 用の道具

計画と結果は [docs/qwen35moe](../../docs/qwen35moe/README.md)。
**ここに置くものは GPU を使わない。**numpy だけで動く (mlx / torch は要らない)。

| ファイル | 役割 |
| --- | --- |
| `checkpoint_io.py` | safetensors のヘッダとテンソルを、丸ごと読まずに触る最小の道具。BF16 は無損失で float32 に広げる。打ち直し版で起きる「影のテンソル」は `model.safetensors.index.json` の `weight_map` で解決する |
| `audit_checkpoint.py` | MLX 形式のチェックポイントを上流 bf16 と突き合わせる監査。量子化形式の赤リスト・区画別バイト・`expertStride`・norm 規約・`conv1d` の軸順・router のビット幅 |

```bash
# 例: oQ4e-g64 を上流 bf16 の抽出と突き合わせる (docs/qwen35moe/12 の再現)
scratch/vision-venv/bin/python Scripts/qwen35/audit_checkpoint.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 \
    --bf16 ~/LLM/Ornith-1.5-35B-A3B-bf16-partial \
    --json scratch/qwen35/oq4e-g64-audit.json
```

チェックポイントの置き場は [docs/qwen35moe/02 §2](../../docs/qwen35moe/02-CHECKPOINTS.md)。
`scratch/vision-venv` は Vision 作業で作った venv (numpy があればよい)。
