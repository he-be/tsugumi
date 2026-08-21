# Scripts/qwen35 — Qwen3.5-MoE (Ornith) 用の道具

計画と結果は [docs/qwen35moe](../../docs/qwen35moe/README.md)。
**ここに置くものは GPU を使わない。**numpy だけで動く (mlx / torch は要らない)。
唯一の例外は `test_reference_forward.py` で、参照器を上流実装と突き合わせるために
`transformers` と torch を使う (`scratch/vision-venv` に入っている)。

| ファイル | 役割 |
| --- | --- |
| `checkpoint_io.py` | safetensors のヘッダとテンソルを、丸ごと読まずに触る最小の道具。BF16 は無損失で float32 に広げる。打ち直し版で起きる「影のテンソル」は `model.safetensors.index.json` の `weight_map` で解決する |
| `audit_checkpoint.py` | MLX 形式のチェックポイントを上流 bf16 と突き合わせる監査。量子化形式の赤リスト・区画別バイト・`expertStride`・norm 規約・`conv1d` の軸順・router のビット幅。`bake_manifest.json` があれば係数を割り戻して照合する |
| `bake_snapshot.py` | `q_norm × head_dim**-0.5` の焼き込み。元の 21.9 GB は触らず、差分シャード 1 枚 + 書き換えた `index.json` + 元ファイルへの symlink を出す (212 KB)。2 のべき乗なので無損失で、2 経路のビット一致を毎回検査する |
| `mlx_quant.py` | MLX affine の逆量子化 (bits 4/8、group 32/64、テンソルごとの上書き対応)。上流 bf16 と突き合わせて検証済み ([docs/qwen35moe/14 §2](../../docs/qwen35moe/14-REFERENCE.md)) |
| `reference_forward.py` | **float32 の参照実装。**層を 1 つずつ開いて捨てるので、19.5〜21.9 GB のモデルを **18 GB の機械でピーク 3 GB** で流せる。fixtures の書き出しと greedy 生成もここ |
| `nll_texts.py` | 候補比較用の文章 4 本 (日本語散文 / 日本語技術文 / 英語技術文 / コード) をトークン ID にして落とす。**両チェックポイントのトークナイザが同じ ID を出すことを確かめてから**書く ([docs/qwen35moe/16 §1](../../docs/qwen35moe/16-QUALITY.md)) |
| `in_proj_a_real.py` | 減衰ゲートの量子化感度を**実活性**で測る。参照は上流 bf16 の抽出 ([docs/qwen35moe/16 §2](../../docs/qwen35moe/16-QUALITY.md)) |
| `test_reference_forward.py` | 参照器の検証。小さい乱数モデルを `transformers` の `Qwen3_5MoeForCausalLM` (CPU/float32) と参照器の両方に流し、logits が一致することを見る。**状態の持ち越し** (prefill + 1 トークンずつ) も同じ物差しで見る |

```bash
# 例: oQ4e-g64 を上流 bf16 の抽出と突き合わせる (docs/qwen35moe/12 の再現)
scratch/vision-venv/bin/python Scripts/qwen35/audit_checkpoint.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 \
    --bf16 ~/LLM/Ornith-1.5-35B-A3B-bf16-partial \
    --json scratch/qwen35/oq4e-g64-audit.json
```

```bash
# 例: q_norm に 1/16 を焼く (docs/qwen35moe/12 §5)
scratch/vision-venv/bin/python Scripts/qwen35/bake_snapshot.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-baked
```

```bash
# 例: 参照器を上流実装に対して検証する (これだけ torch を使う)
scratch/vision-venv/bin/python Scripts/qwen35/test_reference_forward.py
```

```bash
# 例: 実物を 1 回流して fixtures を落とす
scratch/vision-venv/bin/python Scripts/qwen35/reference_forward.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 --text "日本の首都は" \
    --dump scratch/qwen35/fixtures-oq4e-g64.npz
```

```bash
# 例: 2 候補の平均 NLL を同じ文章で取る (1 本あたり 210〜266 s)
scratch/vision-venv/bin/python Scripts/qwen35/nll_texts.py
scratch/vision-venv/bin/python Scripts/qwen35/reference_forward.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 --tokens "$(...)" --nll
```

チェックポイントの置き場は [docs/qwen35moe/02 §2](../../docs/qwen35moe/02-CHECKPOINTS.md)。
`scratch/vision-venv` は Vision 作業で作った venv (numpy があればよい)。
