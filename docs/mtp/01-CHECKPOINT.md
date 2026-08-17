# 01. ドラフターの事実と、未解決の仕様質問

リモート側の事実は 2026-08-17 に HF API / Range 取得 / 参照ソース取得で直接読んだもの
(**実測**)。revision とハッシュは上流で変わりうる (M1 で再確認)。

---

## 1. 使うチェックポイント

| | |
| --- | --- |
| 変換元 | `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` @ `9537141506fe…` (bf16、48 テンソル、839,427,840 B) |
| 採用候補 | `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` @ `bb94eae1b70a…` (int4 affine group 64、236,124,704 B) |
| 系譜の確証 | 量子化されない BF16 テンソル 3 本 (`model.norm` / `layers.0.input_layernorm` / `layers.3.q_norm`) の SHA-256 が両者で**バイト一致** → MLX 版は変換元の忠実な変換物 |
| 前提条件 | Google のモデルカード: 「MTP を使うとき assistant も**同じ精度の QAT チェックポイント**であること」。本ドラフターは QAT ターゲット向けに訓練・分割されている |

本リポジトリのテキスト側は `qat-q4_0-mlx-aligned` (lattice、group 32)、
ドラフターは通常の min/max 誘導 affine (group 64)。**方式の違いは受理率にのみ効く**
(ドラフターは提案専用で、出力はターゲットの検証が決める)。

## 2. 形 (設計に効く分だけ)

- 4 層。層 0–2 が sliding (head_dim 256)、層 3 が full (head_dim 512)。hidden 1024、
  intermediate 8192、q_norm あり (k_norm なし)、`layer_scalar` あり、sandwich 残差、
  gelu_pytorch_tanh、rms_norm_eps 1e-6。
- **k_proj / v_proj が 1 本もない。**全層がターゲットの**層 28 (最終 sliding) と
  層 29 (最終 full)** の K/V を共有する。ドラフターは自前 KV を持たない
  (`num_kv_shared_layers = 4`)。
- `pre_projection` [1024, 5632] — 入力は `concat(embed(tok) × √2816, last_hidden)`。
- `post_projection` [2816, 1024] — 次ステップに渡す hidden。
- lm head は埋め込み [262144, 1024] に tie (`tie_word_embeddings: true`)。
- centroid (ordered embeddings) は **26B では未使用** (`use_ordered_embeddings: false`)。
  config に残る `num_centroids` / `centroid_intermediate_top_k` は死んだ項目。

## 3. 1 ラウンドの流れ (block_size = bs)

```
ターゲットが通常 decode を 1 回 → bonus トークン + その位置の最終 hidden + 共有 KV

ドラフト (bs − 1 回、自己回帰、q_len = 1):
  in  = concat(embed(tok) × √2816, last_hidden)      # [1, 5632]
  h   = pre_projection(in)                            # [1, 1024]
  h   = 4 層 (共有 KV に attend、自前 KV なし)
  h   = norm(h)
  last_hidden = post_projection(h)                    # [1, 2816]
  logits = embed_tokens.as_linear(h)
  tok = sample(logits)
  ※ RoPE 位置は全ステップで bonus の絶対位置に固定

検証: ターゲットが bs − 1 個のドラフトを 1 回の forward で処理し、
      各位置の logits とドラフトを比較 → 受理接頭辞を採用 → KV を受理長まで巻き戻す
```

**マスクは不要** (**導出**、根拠は参照実装): 参照実装は将来の一括ドラフト用に
「双方向マスク」を組むが、q_len=1 の自己回帰では full 層で通常の causal full attention に、
sliding 層で通常の SWA decode attention に潰れる。既存の decode attention を
共有 KV に対して呼ぶだけで意味が一致する。mlx-vlm 側も
`kv_len <= sliding_window` ではマスクを short-circuit する。

## 4. ターゲットとの整合 (**実測**)

`ArchConfig.gemma4_26B_A4B` (`Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift:78-107`)
と突き合わせた。ドラフターの要求はすべて既存の形に収まる。

| 項目 | ターゲット | ドラフターの要求 | 判定 |
| --- | --- | --- | --- |
| `numLayers` / full 層マスク | 30 / `stride(from:5, to:30, by:6)` = 5,11,17,23,29 | 共有 KV は**最終 full (29) と最終 sliding (28)** | 整合 |
| `numHeads` / `headDim` | 16 / 256 | sliding 層 q 16×256 | 整合 |
| `numKVHeads` | 8 | sliding 共有 KV 8×256 (GQA 2:1) | 整合 |
| `numFullKVHeads` / `fullHeadDim` | 2 / 512 | full 層 q 16×512、共有 KV 2×512 (GQA 8:1) | 整合 |
| `slidingWindow` | 1024 | 1024 | 整合 |
| `ropeTheta` / `fullRopeTheta` / `partialRotaryFactor` | 1e4 / 1e6 / 0.25 | 同一 | 整合 |
| `hiddenSize` | 2816 | `backbone_hidden_size` 2816 | 整合 |
| `tieWordEmbeddings` / `attentionKEqV` | true / true | 同一 | 整合 |

→ **ドラフターの attention は既存 decode attention カーネルの形とそのまま合う。**

## 5. 未解決の仕様質問

実装前に**参照実装を読んで確定させる** (`scratch/mtp-ref/` に取得済み:
transformers `gemma4_assistant` の configuration/modeling と、mlx-vlm の
`speculative/drafters/gemma4_assistant` 一式 + README)。

| # | 質問 | なぜ効くか | 決着 |
| --- | --- | --- | --- |
| Q1 | ドラフターの logits に **final logit softcap 30.0** を掛けるか | ターゲット側は `ArchConfig.finalLogitSoftcap = 30.0` で `Sampler` も softcap 前提。掛ける/掛けないで提案分布が変わり**受理率が動く** | M0 / M2 |
| Q2 | ドラフターに渡す hidden は `model.norm` **後**で確定か | 本ランタイムは post-norm hidden を materialize していない (02 §3-N3)。どちらかで追加カーネルの位置が変わる | M0 / M2 |
| Q3 | ドラフト内部の sampling は greedy 固定か、ターゲットと同じ温度か | 受理率に直結。同温度なら「同じ分布から 2 回引く」ぶん一致確率が下がる (**導出**) | M0 |
| Q4 | 共有 KV は層 28/29 の K/V をそのまま使うか、追加の norm/RoPE があるか | ターゲット側 KV は k_norm + RoPE 済みでキャッシュされている | M2 |
| Q5 | `bs` (block size) の推奨値 | 上流の自己申告は best bs = 3 (batch 4/8、M3 Max)。B=1 での最適は不明 | M5 |
| Q6 | 埋め込みスケール √2816 はドラフター側でも同じか | mlx-vlm の `bind()` がターゲットの `embed_scale` を読む (**実測**)。再確認のみ | M2 |

## 6. 明示的に対象外

- **DFlash / unified assistant** (`gemma4_dflash` 系)。別アルゴリズム・別ループ。
- **centroid (ordered embeddings)。**26B では未使用。
- **E2B / E4B / 31B 向けドラフターへの一般化。**
- **ドラフターの再量子化** (bf16 → 独自 int4)。int4 版をそのまま使い、
  受理率が足りない場合にだけ検討する。
