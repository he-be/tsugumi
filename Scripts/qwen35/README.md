# Scripts/qwen35 — Qwen3.5-MoE (Ornith) 用の道具

計画と結果は [docs/qwen35moe](../../docs/qwen35moe/README.md)。
**ここに置くものは GPU を使わない。**numpy だけで動く (mlx / torch は要らない)。
例外は 2 本: `test_reference_forward.py` は参照器を上流実装と突き合わせるために
`transformers` と torch を使い、`graft_mtp_head.py` は量子化に mlx を使う
(どちらも CPU。mlx は `scratch/mtp-venv`、torch は `scratch/vision-venv`)。

| ファイル | 役割 |
| --- | --- |
| `checkpoint_io.py` | safetensors のヘッダとテンソルを、丸ごと読まずに触る最小の道具。BF16 は無損失で float32 に広げる。打ち直し版で起きる「影のテンソル」は `model.safetensors.index.json` の `weight_map` で解決する |
| `audit_checkpoint.py` | MLX 形式のチェックポイントを上流 bf16 と突き合わせる監査。量子化形式の赤リスト・区画別バイト・`expertStride`・norm 規約・`conv1d` の軸順・router のビット幅。`bake_manifest.json` があれば係数を割り戻して照合する |
| `graft_mtp_head.py` | **出荷版の MTP ヘッド 42 本を、学習済みのヘッドで差し替える。**供給側の BF16 19 本を、norm は +1、融合 `gate_up_proj` は `[gate; up]` で分割、量子化は差し替え先のテンソルごとの規約 (8b/g64 と 4b/g64) で 42 本に写す。**ここだけ mlx が要る** (`mx.quantize`、CPU ストリーム)。出力は差し替えシャード 1 枚 + 書き換えた `index.json` + 元ファイルへのハードリンク。書いたバイトを読み直してバイト一致・形・参照バイト・往復 L2・expert 間 std の CV まで検算する ([docs/qwen35moe/30 §6](../../docs/qwen35moe/30-MTP-HEAD-GRAFT.md)) |
| `mtp_acceptance.py` | **受理率 P1 と受理長 a を CPU で測る** (Phase 7 の M0)。実生成のトークン列を本体に教師強制で 1 回通し、その hidden に MTP ヘッドを深さ 3 まで鎖で当てて本体 top-1 との一致を数える。pre/post-norm と鎖の作法の両方を引ける。**差し替え済み・焼き込み前**のチェックポイントしか受け取らない ([docs/qwen35moe/33](../../docs/qwen35moe/33-MTP-ACCEPTANCE.md)) |
| `verify_fused_split.py` | 融合 `gate_up_proj` の**分割順**を上流の実物で引き直す。上流 layer-0 expert-0 (4.2 MB) を HTTP Range で取り、3 仮説に相関を当てる。**ネットワークが要る** |
| `mtp_history_ablation.py` | **MTP 自身の注意履歴をどこから始めるか**で深さ 1 の受理率がどれだけ動くかを、`mtp_acceptance.py` が残した hidden の npz だけで引く (**モデルの再実行なし**)。`full` (教師強制と同じ) / `gen` (生成開始から = 実機) / `last` (履歴なし)。実機に MTP の prompt prefill を書くかどうかの判断材料 ([docs/qwen35moe/36 §2-2](../../docs/qwen35moe/36-MTP-DECODE.md)) |
| `build_mtp_sidecar.py` | 差し替え済み MTP ヘッド 42 本を、ランタイムが mmap できる **sidecar** (`mtp_core.bin` 50 MB + `mtp_experts.bin` 453 MB + 目録) に書き出す。エキスパートの blob は**本体の `packed_experts` とバイト並びが同じ**で、`--moepack` を渡すと `expertStride` を照合する。`.moepack` の repack は要らない ([docs/qwen35moe/36 §1](../../docs/qwen35moe/36-MTP-DECODE.md)) |
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
# 例: MTP ヘッドを差し替えて、そのうえで q_norm を焼く (docs/qwen35moe/30 §6)
#     mlx が要るのはこの 1 本だけなので venv が違う
scratch/mtp-venv/bin/python Scripts/qwen35/graft_mtp_head.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 ~/LLM/Ornith-1.5-35B-A3B-MTP-ONLY \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa
scratch/vision-venv/bin/python Scripts/qwen35/bake_snapshot.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa-baked

# 分割順を上流の実物で引き直す (ネットワーク)
scratch/vision-venv/bin/python Scripts/qwen35/verify_fused_split.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64
```

```bash
# 例: 受理長 a を測る (docs/qwen35moe/33)。トークン id はランタイムに書かせる
.build/release/TsugumiCLI --model scratch/ornith-oq4e-g64.moepack \
    --messages-file bench/qwen35/t2-code.json --temperature 0 \
    --repetition-penalty 1 --thinking off --max-new 192 \
    --verification trusted-install \
    --dump-tokens scratch/qwen35/mtp-a/t2-code.tokens.json > /dev/null
scratch/vision-venv/bin/python Scripts/qwen35/mtp_acceptance.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa \
    --tokens scratch/qwen35/mtp-a/t2-code.tokens.json \
    --messages bench/qwen35/t2-code.json
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
**MTP ヘッドを読むものは `~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa-baked` だけを使う**
— 出荷版の `mtp.*` は乱数初期化で、差し替え済み
([docs/qwen35moe/30](../../docs/qwen35moe/30-MTP-HEAD-GRAFT.md))。
`scratch/vision-venv` は Vision 作業で作った venv (numpy があればよい)。
