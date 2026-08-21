# PLAN_QWEN35 — Qwen3.5-MoE (Ornith-1.5-35B-A3B) 対応

作成: 2026-08-21
M3 Pro 18GB / macOS 15.7.5 / ブランチ `macos15-support`
対象: [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
(`architectures: ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: qwen3_5_moe`)

表記は PLAN.md / PLAN_QAT.md / PLAN_VISION.md と同じ:
**実測** / **導出** / **未確認**

ただし本 PLAN の **実測** は 2 種類ある。混ぜないために区別する:

| 記号 | 意味 |
| --- | --- |
| **実測(上流)** | 上流リポジトリの実体を取得して確認した事実。`config.json`、`model.safetensors.index.json`、各シャードの safetensors ヘッダ (HTTP range で先頭のみ取得)、`tokenizer_config.json`、`chat_template.jinja`、`transformers` の `modeling_qwen3_5_moe.py` |
| **実測(手元)** | この機械で数字を取ったもの |

**GPU はまだ 1 度も回していない。**「実測(手元)」は §16 と §17 にあるが、
すべて**ファイルを読む / CPU で量子化する**までの数字である。
速度・ヒット率・TTFT に関する数字はすべて **導出** か **未確認** のままである。
運用点 (スロット数・チャンク幅) は既存の Gemma 4 の値をそのまま持ち越せない —
持ち越せない理由は §7 に、測り直しの手順は §11 に書く。

> **本 PLAN 作成後の追記 (2026-08-21): §15 を読んでから本文を読むこと。**
> MLX 4-bit 量子化済みの公開チェックポイントが存在することが分かり、
> **§4 (重み変換) と §5-6 (Gated DeltaNet カーネル) の結論が変わった。**
> 食い違ったときは **§15-7 の差し戻し表が正**である。
>
> **さらに追記 (2026-08-21 夜): 現在地は §17 にある。**候補チェックポイントが
> 2 本になり、§16-7 が付けた順位は保留になった。§15-7 / §16-7 と食い違ったら **§17 が正**。

方針は既存 PLAN と同じ: **汎用性を捨てる。**この 1 台 (M3 Pro / 18GB /
macOS 15.7.5) で速いことだけを目的にし、互換性・移植性・他アーキテクチャへの
一般化は最初から狙わない。

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | これは何か | **Qwen3.5-MoE。ただの「Qwen 版 Gemma」ではない。**40 層のうち **30 層が線形注意 (Gated DeltaNet)**、10 層だけが full attention。SWA は 1 層も無い (**実測(上流)**、§1) |
| 2 | 一番大きい実装 | **Gated DeltaNet カーネルの新規実装。**本ランタイムに相当物が 1 個も無い。KV キャッシュではなく**固定サイズの再帰状態**を持つ層という概念自体が無い (§6-2) |
| 3 | 二番目に大きい実装 | **重み変換の入口。**本リポジトリには量子化器が無い。既存の repack は「MLX が量子化済みのチェックポイント」を**並べ替えるだけ**である (**実測(上流)** = `SupportedModelSource.repoID` が `mlx-community/…-4bit`)。Ornith は bf16 しか無いので、**bf16 → MLX affine 4-bit の変換器を新規に書く**のが Phase 1 の主題 (§4) |
| 4 | MoE の形は乗るか | **乗る。ただし `numExperts <= 256` の precondition にちょうど乗る (余裕ゼロ)。**top-8 は一致、`D=2048` は 64 の倍数、prefill router のスクラッチは既に 256 で確保済み。**decode/prefill の MoE カーネルは無改造で正しく動く**見込み (専用化 PSO から汎用 PSO に落ちるだけ) (§8) |
| 5 | エキスパート 1 個のバイト数 | **1,769,472 B = 16 KiB × 108 ちょうど。パディング 0 バイト** (**導出**、§3-2)。Gemma は 205 ページ中 13,312 B が捨て札なので、そこは改善 |
| 6 | 1 トークンあたりのバイト | **導出で Gemma の 0.78 倍** (4K 文脈、全ヒット時 2.41 GB → 1.89 GB)。**decode は Gemma より速くなり得る**。ただしヒット率が落ちる要因が別にある (#7) |
| 7 | 一番大きい性能リスク | **エキスパート母集団が 3,840 → 10,240 に増える。**32 スロットが覆う割合は 1 層あたり 25% → 12.5% に半減する。スロット 1 本の値段は 96.1 → 67.5 MiB に下がるので**バイト等価なら 45 本**だが、`allowedExpertCacheSlots` は `[8,16,24,32]` で頭打ち (§7-1) |
| 8 | 二番目に大きい性能リスク | **prefill の GEMM 占有率が半減する。**チャンク 2048 でエキスパート 1 個あたり平均 128 行 → 64 行。64 行ブロックがちょうど 1 個しか埋まらない (§7-2) |
| 9 | 一番大きい正しさリスク | **partial RoPE のペアの取り方が Gemma と違う。**本ランタイムは `(i, HD/2+i)` を回し周波数の分母に `HD` を使う。Qwen は `(i, 32+i)` を回し分母は `rotary_dim=64`。**既存カーネルを流用すると静かに間違う** (§5-2) |
| 10 | ただで貰えるもの | (a) `attention_prefill_causal_qblock_d256` は head_dim だけで選ばれるので**そのまま当たる** (b) RMSNorm は Qwen も `1+w` 規約なので既存カーネルのまま (c) 文脈長は 10 層ぶんしか KV が要らず、線形層の状態は**文脈長に依らず 62.8 MiB 固定** (§3-3) |
| 11 | MTP | **本モデルは MTP ヘッドを同梱している** (`mtp.*`、1 層、805M のエキスパートつき)。**専用ドラフターを別リポジトリから取ってくる必要が無い。**しかも 4-bit で 453 MB なので**全 256 エキスパートを常駐にできる** = ドラフトの I/O がゼロになる (§10) |
| 12 | サーバーへの波及 | **prompt cache と投機デコードの前提が壊れる。**再帰状態は「途中を捨てる」「巻き戻す」ができない。SPEC の LCP 再利用と MTP の verify ブロックの両方に設計変更が要る (§9) |
| 13 | Vision | 後回しでよい。**tower の形は Gemma と偶然ほぼ同じ** (1152 / 27 層 / 16 head / 4304) だが、**位置符号化とマージが別物** (2×2 patch merger MLP + mRoPE 対 3×3 平均プーリング + 加算テーブル + 2D RoPE)。カーネルは書き直し (§12) |

---

## 0-A. 何を確認し、何を確認していないか

**確認した (実測(上流)):**

- `config.json` 全文。`layer_types` の 40 要素、`linear_*` 一式、`mtp_num_hidden_layers`、`vision_config`
- `model.safetensors.index.json` の **1,811 本**のテンソル名 (16 シャード、`total_size = 71,903,645,408 B`)
- シャード 1 / 2 / 16 の safetensors ヘッダを HTTP range で取得し、**全テンソルの shape と dtype を確定**した (全部 BF16)
- `tokenizer_config.json` の追加トークン 33 本と ID、`chat_template.jinja` 全文、`generation_config.json`、`preprocessor_config.json`
- `transformers` の `modeling_qwen3_5_moe.py` (97 KB) を読み、`Qwen3_5MoeGatedDeltaNet` / `Qwen3_5MoeAttention` / `Qwen3_5MoeTopKRouter` / `Qwen3_5MoeSparseMoeBlock` / `Qwen3_5MoeRMSNorm` / `apply_rotary_pos_emb` / `torch_recurrent_gated_delta_rule` の**算式を確定**した

**確認していない (未確認、Phase 0 で潰す):**

1. `mlp.experts.gate_up_proj` `[256, 1024, 2048]` の行 0..511 が gate で 512..1023 が up か (**連結**か**交互**か)。`Qwen3_5MoeExperts.forward` の `.chunk(2, dim=-1)` は `F.linear` 後の出力に対してなので**連結が濃厚だが、バイトで確かめていない**
2. 4-bit RTN を通したときの品質。Gemma 側は QAT チェックポイントという逃げ道があったが、Ornith には無い
3. この機械の SSD が 1.69 MiB の読み出しをどれだけ並べられるか (Gemma の 3.19 MiB とは別の点で測っている)
4. ルーターの偏り。Gemma で効いた LFU・先読みの前提が 256-way でも成り立つか

---

## 1. 相手の正体 (実測(上流))

### 1-1. text_config

```
hidden_size            2048          num_hidden_layers      40
num_attention_heads    16            num_key_value_heads    2
head_dim               256           attn_output_gate       true
partial_rotary_factor  0.25          rope_theta             10,000,000
rope: mrope_interleaved=true, mrope_section=[11,11,10], rope_type=default
num_experts            256           num_experts_per_tok    8
moe_intermediate_size  512           shared_expert_intermediate_size 512
hidden_act             silu          rms_norm_eps           1e-6
vocab_size             248,320       tie_word_embeddings    false
max_position_embeddings 262,144      full_attention_interval 4
linear_num_key_heads   16            linear_key_head_dim    128
linear_num_value_heads 32            linear_value_head_dim  128
linear_conv_kernel_dim 4             mamba_ssm_dtype        float32
mtp_num_hidden_layers  1             mtp_use_dedicated_embeddings false
```

`layer_types` は `[linear, linear, linear, full] × 10`。**full attention は層 3, 7, 11, …, 39 の 10 層。**
sliding window は無い (フィールドそのものが無い)。`final_logit_softcapping` も無い。

### 1-2. テンソルの実体 (shape はすべて実測(上流)、dtype は全部 BF16)

| 名前 (`model.` を剥がした後) | shape | 本数 |
| --- | --- | ---: |
| `language_model.embed_tokens.weight` | `[248320, 2048]` | 1 |
| `lm_head.weight` (先頭に `model.` は付かない) | `[248320, 2048]` | 1 |
| `language_model.norm.weight` | `[2048]` | 1 |
| `…layers.{i}.input_layernorm.weight` | `[2048]` | 40 |
| `…layers.{i}.post_attention_layernorm.weight` | `[2048]` | 40 |
| **線形注意層 (30)** | | |
| `…linear_attn.in_proj_qkv.weight` | `[8192, 2048]` | 30 |
| `…linear_attn.in_proj_z.weight` | `[4096, 2048]` | 30 |
| `…linear_attn.in_proj_a.weight` | `[32, 2048]` | 30 |
| `…linear_attn.in_proj_b.weight` | `[32, 2048]` | 30 |
| `…linear_attn.conv1d.weight` | `[8192, 1, 4]` | 30 |
| `…linear_attn.A_log` / `.dt_bias` | `[32]` | 60 |
| `…linear_attn.norm.weight` | `[128]` | 30 |
| `…linear_attn.out_proj.weight` | `[2048, 4096]` | 30 |
| **full attention 層 (10)** | | |
| `…self_attn.q_proj.weight` | `[8192, 2048]` | 10 |
| `…self_attn.k_proj.weight` / `.v_proj.weight` | `[512, 2048]` | 20 |
| `…self_attn.o_proj.weight` | `[2048, 4096]` | 10 |
| `…self_attn.q_norm.weight` / `.k_norm.weight` | `[256]` | 20 |
| **MoE (全 40 層)** | | |
| `…mlp.gate.weight` (router) | `[256, 2048]` | 40 |
| `…mlp.experts.gate_up_proj` | `[256, 1024, 2048]` | 40 |
| `…mlp.experts.down_proj` | `[256, 2048, 512]` | 40 |
| `…mlp.shared_expert.{gate,up}_proj.weight` | `[512, 2048]` | 80 |
| `…mlp.shared_expert.down_proj.weight` | `[2048, 512]` | 40 |
| `…mlp.shared_expert_gate.weight` | `[1, 2048]` | 40 |
| **MTP (1 層)** | | |
| `mtp.fc.weight` | `[2048, 4096]` | 1 |
| `mtp.pre_fc_norm_embedding.weight` / `_hidden.weight` / `mtp.norm.weight` | `[2048]` | 3 |
| `mtp.layers.0.self_attn.*` | full attention 層と同形 | 6 |
| `mtp.layers.0.mlp.experts.{e}.{gate,up}_proj.weight` | `[512, 2048]` | 512 |
| `mtp.layers.0.mlp.experts.{e}.down_proj.weight` | `[2048, 512]` | 256 |
| `mtp.layers.0.mlp.{gate,shared_expert*}` | 本体と同形 | 5 |
| **Vision (27 ブロック)** | | 111 |

**注意すべき非対称:** 本体の routed expert は**融合 3D テンソル 2 本** (`gate_up_proj` / `down_proj`)
なのに、**MTP 側はエキスパートごとにバラの 768 本**である (**実測(上流)**)。変換器は両方の形を扱う。

### 1-3. 算式 (`modeling_qwen3_5_moe.py` を読んで確定、実測(上流))

**RMSNorm — Gemma と同じ `1 + w` 規約:**

```python
output = x * rsqrt(mean(x^2) + eps)
output = output * (1.0 + self.weight.float())     # Qwen3_5MoeRMSNorm
```

つまり `input_layernorm` / `post_attention_layernorm` / `q_norm` / `k_norm` / `norm` /
`mtp.*_norm` は**変換時に +1 を焼き込めば既存の `rmsnorm_bf16w` がそのまま使える**。
**ただし `linear_attn.norm` だけは別物** (`Qwen3_5MoeRMSNormGated`) で `+1` しない:

```python
h = h * rsqrt(mean(h^2)+eps);  h = self.weight * h;  h = h * silu(gate.float())
```

**+1 する norm と しない norm を取り違えると静かに壊れる。**

**full attention:**

```python
q_all = q_proj(x).view(..., -1, head_dim*2)          # [.., 16, 512]
q, gate = chunk(q_all, 2, dim=-1)                     # 各 [.., 16, 256]
q = q_norm(q);  k = k_norm(k_proj(x));  v = v_proj(x)
q, k = rope(q, k)                                     # partial 0.25 (下記)
o = attention(q, k, v, scale = head_dim**-0.5)        # GQA 16/2, causal
o = o * sigmoid(gate)                                 # ← attn_output_gate
o = o_proj(o)
```

**partial RoPE (§0 #9 の中身):**

```python
dim      = head_dim * 0.25 = 64
inv_freq = 1 / (theta ** (arange(0, 64, 2) / 64))     # 32 本、分母は 64
q_rot, q_pass = q[..., :64], q[..., 64:]
q_out = cat([q_rot*cos + rotate_half(q_rot)*sin, q_pass])
# rotate_half は 64 次元の中で (i, 32+i) を組む
```

本ランタイムの `fused_rope_neox_pair` は `(pair, HD/2 + pair)` を組み `pow(theta, -2*pair/HD)`
を使う。**組も分母も違う。**新カーネルが要る (§5-2)。

mRoPE は `mrope_section=[11,11,10]` を t/h/w に交互配置するが、**テキストのみなら
3 軸の position_ids が全部同じ**になるので `apply_interleaved_mrope` は恒等写像に潰れる。
→ **Phase 1〜8 は mRoPE を一切実装しなくてよい。Vision (§12) でだけ要る。**

**router (`Qwen3_5MoeTopKRouter`):**

```python
probs = softmax(x @ W.T, dim=-1)      # 256 全体で softmax
v, idx = topk(probs, 8)
v = v / v.sum(-1, keepdim=True)       # 選ばれた 8 本で再正規化
```

本ランタイムの `router_topk_select_k8` は「top-8 に限った softmax」を計算する。
**数学的に同一** (全体 softmax → top-k → 再正規化 は分母が消える)。**そのまま使える。**
Gemma 由来の `per_expert_scale` は Qwen に対応物が無いので **1.0 を書く**。

**MoE ブロック:**

```python
shared = shared_expert(x)                             # silu-gated, 幅 512
shared = sigmoid(shared_expert_gate(x)) * shared      # ← Gemma に無い
routed = Σ_k w_k * down_k(silu(gate_k(x)) * up_k(x))
y = routed + shared
```

**decoder layer — Gemma よりずっと単純:**

```python
h = h + mixer(input_layernorm(h))       # mixer = linear_attn か self_attn
h = h + moe(post_attention_layernorm(h))
```

sandwich norm も `layer_scalar` も無い。**norm は層あたり 2 本だけ** (Gemma は 6 本 + scalar)。

**Gated DeltaNet (30 層ぶんの本体):**

```python
qkv = conv1d_causal(in_proj_qkv(x))        # depthwise, kernel 4, groups=8192
qkv = silu(qkv)
q, k, v = split(qkv, [2048, 2048, 4096])   # q,k: 16 head × 128 / v: 32 head × 128
q, k = repeat_interleave(q, 2), repeat_interleave(k, 2)   # → 32 head に揃える
q, k = l2norm(q), l2norm(k)                # eps 1e-6
q = q / sqrt(128)
beta  = sigmoid(in_proj_b(x))                         # [32]
g     = -exp(A_log) * softplus(in_proj_a(x) + dt_bias) # [32]
# 1 トークンぶんの再帰 (S: [32, 128, 128], fp32)
S      = S * exp(g)
kv_mem = einsum('hkv,hk->hv', S, k)
delta  = (v - kv_mem) * beta
S      = S + einsum('hk,hv->hkv', k, delta)
o      = einsum('hkv,hk->hv', S, q)
o = rmsnorm_gated(o, z = in_proj_z(x))     # +1 しない norm、silu(z) で乗算
y = out_proj(o.reshape(-1, 4096))
```

---

## 2. Gemma 4 26B-A4B との差分 (一覧)

| 項目 | Gemma 4 26B-A4B | Ornith / Qwen3.5-MoE | 影響 |
| --- | --- | --- | --- |
| 層構成 | 30 層、SWA×25 + full×5 | **40 層、linear×30 + full×10** | **最大。層種別が 2 値 → 3 値** |
| hidden | 2816 | 2048 | 専用化 PSO が外れる |
| head 数 | 16 / KV 8 (SWA) / KV 2 (full) | 16 / KV 2 (full のみ) | GQA 比 8。SWA 用 GQA カーネルは使わない |
| head_dim | 256 (SWA) / 512 (full) | **256 (full)** | prefill の `_d256` がそのまま当たる |
| attention 出力ゲート | 無し | **有り** (`sigmoid(gate) *`) | 小カーネル 1 個 |
| K == V | full 層で `v_proj = k_proj` | **常に別** | `isFull ? k : vProj` の分岐を消す |
| RoPE | 2 θ (1e4 / 1e6)、SWA は全回転 / full は partial 0.25、ペアは `(i, HD/2+i)` | **1 θ (1e7)、full のみ partial 0.25、ペアは `(i, 32+i)`、分母 64** | **新カーネル (§5-2)** |
| logit softcap | 30.0 | **無し** | 無条件適用を切る |
| 埋め込み | tie する (262144×2816 が LM head 兼用) | **tie しない** (embed と lm_head が別、各 248320×2048) | 常駐 +286 MB |
| 活性化 | `gelu_pytorch_tanh` | **`silu`** | **SiLU カーネルが 1 個も無い** |
| norm | sandwich 6 本 + `layer_scalar` | **2 本だけ** | `fused_post_attn_setup` / `fused_layer_tail` は簡略版に |
| norm 規約 | `1+w` を変換時に焼く | **同じ** (ただし `linear_attn.norm` は例外) | 既存カーネル流用可 |
| expert | 128/層、top-8、幅 704、hidden 2816 | **256/層**、top-8、幅 512、hidden 2048 | `numExperts<=256` にちょうど乗る |
| expert 1 個 | 3,345,408 B → 205 ページ (13,312 B 捨て) | **1,769,472 B → 108 ページ (捨て 0)** | ページ効率は改善 |
| shared expert | 幅 2112 (= 3×704)、ゲート無し | 幅 512 (= 1×512)、**sigmoid ゲート有り** | 小カーネル 1 個 |
| MoE の置き場所 | 全 30 層 | **全 40 層** (線形注意層にも有る) | 事前計算ループの前提は保たれる |
| KV | FP16、SWA はリング / full は maxContext | **full 10 層のみ。線形 30 層は固定 62.8 MiB の状態** | **KVCacheManager に第 3 の層種別** |
| トークナイザ | SentencePiece 系 (metaspace + ByteFallback + Fuse)、262144 | **Qwen2 byte-level BPE**、248,320 | `verifyDecoderConfiguration` が弾く |
| チャット形式 | `<|turn|>` + `<|channel|>thought` | **ChatML + `<think>`/`<tool_call>` XML** | テンプレートとパーサを別系統で |
| ドラフター | 別リポジトリの assistant 4 層 | **本体同梱の MTP 1 層** | 取得が要らない (§10) |
| 上流の量子化 | mlx-community が 4-bit 版を配っている | **bf16 しか無い** | **量子化器を書く (§4)** |

---

## 3. 数量 (すべて 導出、根拠は §1-2 の shape)

### 3-1. パラメータ数

| 区画 | パラメータ数 |
| --- | ---: |
| embed_tokens | 508,559,360 |
| lm_head | 508,559,360 |
| 線形注意 30 層 | 1,011,553,920 |
| full attention 10 層 | 272,634,880 |
| MoE の非 routed 部 (router + shared + ゲート + norm) × 40 | 147,046,400 |
| **常駐コア 小計** | **2,448,355,968** |
| routed experts 40 × 256 × 3,145,728 | **32,212,254,720** |
| MTP 1 層 (うち expert 805,306,368) | 844,443,648 |
| Vision tower | 445,896,000 前後 |
| **合計** | **約 35.95 B** (bf16 71.9 GB と一致、**実測(上流)** の `total_size` で検算済み) |

### 3-2. エキスパート 1 個 = 108 ページちょうど

MLX affine 4-bit / group 64 の 1 パラメータあたり: `4 bit + (16+16) bit / 64 = 4.5 bit`。

```
gate  [512, 2048] = 1,048,576 param → 524,288 (w) + 32,768 (scale) + 32,768 (bias) = 589,824
up    同上                                                                          = 589,824
down  [2048, 512] = 1,048,576 param                                                 = 589,824
                                                              合計 = 1,769,472 B
1,769,472 / 16,384 = 108.0   ← 端数ゼロ
```

対称 (`sym`) なら bias が落ちて `1,671,168 B = 16,384 × 102` で**これも端数ゼロ**。
Gemma の `expertStride = 3,358,720` は 205 ページ中 13,312 B が捨て札 (0.40%) なので、
**Ornith は affine でも sym でも 16 KiB 境界にぴったり乗る**。

| | affine | sym |
| --- | ---: | ---: |
| expert 1 個 | 1,769,472 B (108 ページ) | 1,671,168 B (102 ページ) |
| 1 層ファイル (256 本) | 452,984,832 B (432 MiB) | 427,819,008 B (408 MiB) |
| 40 層合計 | **18,119,393,280 B (16.88 GiB)** | 17,112,760,320 B (15.94 GiB) |
| スロット 1 本 (40 層) | **70,778,880 B (67.5 MiB)** | 66,846,720 B (63.8 MiB) |

Gemma のスロット 1 本は `30 × 3,358,720 = 100,761,600 B (96.1 MiB)`。
**Ornith のスロットは 0.70 倍の値段。**

### 3-3. インストール容量と常駐

| 区画 | 量子化 | バイト |
| --- | --- | ---: |
| routed experts | affine int4 g64 | 18.12 GB |
| embed + lm_head | int4 | 572 MB |
| 注意系 (q/k/v/o, in_proj_qkv/z, out_proj) | int4 | 720 MB |
| shared expert | int4 | 71 MB |
| router (`mlp.gate`) 40 本 | int8 | 22 MB |
| 小テンソル (A_log, dt_bias, conv1d, in_proj_a/b, 全 norm) | fp16/fp32 | 約 11 MB |
| **テキストのみ 合計** | | **約 19.5 GB** |
| MTP (うち expert 453 MB) | int4 | 475 MB |
| Vision tower | bf16 | 892 MB |
| **全部入り** | | **約 20.9 GB** |

Gemma 4 の 14.3 GB より **+5 GB**。ディスク要件は上がる。
`model_weights.bin` (常駐) は **約 1.40 GB**。Gemma の 1.35 GB とほぼ同じ桁だが、
**tie しないぶん lm_head の 286 MB が丸ごと上乗せ**されている。

**変換の入力は bf16 71.9 GB。**丸ごと落とすと 71.9 + 19.5 = 91 GB 要る。
シャード単位 (16 本、1 本 4.5 GB 前後) で「落とす → 量子化 → 書く → 消す」を回せば
**ピーク約 25 GB** で済む (§4-3)。

### 3-4. 文脈と状態

| | 1 トークンあたり | 4K | 32K | 128K |
| --- | ---: | ---: | ---: | ---: |
| KV (full 10 層、FP16) | 20,480 B | 82 MB | 655 MB | 2.62 GB |
| 再帰状態 (線形 30 層、fp32) | **0 (固定)** | 62.8 MiB | 62.8 MiB | 62.8 MiB |
| 参考: Gemma 4 (SWA リング + global) | — | 0.71 GB | 1.30 GB | 3.31 GB |

再帰状態の内訳 (1 層): `S[32,128,128] fp32 = 2,097,152 B` + `conv 状態 8192×3 fp32 = 98,304 B`
= `2,195,456 B`。× 30 = **65,863,680 B = 62.8 MiB。文脈長に依らない。**

**注意:** full attention 層の KV は Gemma の global 層とバイト単価が同じ (20 KB/token) なので、
**長文脈が劇的に安くなるわけではない。**得なのは (a) SWA リング 0.63 GB が消えること
(b) SWA の KV 再読み 210 MiB/token が状態往復 126 MiB/token に置き換わること。

### 3-5. decode 1 トークンのバイト予算 (全ヒット時、導出)

| 区画 | Gemma 4 | Ornith |
| --- | ---: | ---: |
| 常駐コア (lm_head 込み、embed は 1 行) | 1.35 GB | 1.11 GB |
| routed experts (top-8 × 層数) | 0.765 GB (240 本) | 0.566 GB (**320 本**) |
| KV 読み (4K 文脈) | 0.30 GB | 0.084 GB |
| 再帰状態 往復 | — | 0.126 GB |
| **合計** | **2.41 GB** | **1.89 GB (0.78×)** |

M3 Pro の帯域を 150 GB/s とすると、**バイトだけの天井は Gemma 62 tok/s に対し Ornith 79 tok/s**
(**導出**、効率 100% の非現実的な上限)。**decode は Gemma より速くなり得る、というのが本 PLAN の
性能側の主張である。**ただし読み出しの**本数**は 240 → 320 に増える (§7-3)。

### 3-6. prefill の計算量 (2048 トークン、導出)

| | Gemma 4 | Ornith |
| --- | ---: | ---: |
| MoE (routed + shared) | 8.03 TFLOP | **4.63 TFLOP (0.58×)** |
| 注意系 | 1.20 TFLOP | **0.54 TFLOP (0.45×)** |
| routed expert I/O の最悪値 | 12.01 GiB | **16.88 GiB (1.41×)** |

**計算は軽くなり I/O は重くなる。**prefill は Gemma 以上に I/O 律速に寄る (§7-2)。

---

## 4. 重み変換 — ここが Phase 1 の主題

### 4-1. 現状の repack は量子化しない (実測 = ソース)

`SupportedModelSource.repoID = "mlx-community/gemma-4-26b-a4b-it-4bit"`。
`RepackPlanner` は「dtype が `u32` で名前が `.weight` で終わるものは 4-bit パック済み、
`.scales` / `.biases` が対になっている」という MLX の規約を前提に、**並べ替えとページ揃えだけ**をする。
`bits=4 / group_size=64 / mode=affine` は上流 `config.json` の `quantization` から読む。

**Ornith には MLX 量子化版が無い。**したがって選択肢は 2 つ:

> **この表は「量子化器をどこに置くか」の話である。**§15-3 の「どのチェックポイントを
> 入力にするか」とは別の軸なので、混ぜないこと (以前はどちらも `案 A` と呼んでいた)。

| 案 | 中身 | 判断 |
| --- | --- | --- |
| **案「Python 変換器」** | **Python で bf16 → MLX 規約の safetensors を書く変換器を新規に作り、`--source-snapshot` で既存 repack に食わせる** | **採用。**Swift 側の量子化まわりを 1 行も触らずに済む。`SymmetricProbe` も `--source-snapshot` 経路でしか動かないので、そこも自然に噛む |
| 案「Swift 側に量子化器」 | Swift の repack に量子化器を足す | 却下。ストリーミング install の設計 (range copy + checkpoint + fingerprint) が「バイトをそのまま運ぶ」ことに依存していて、途中で値を作る余地が無い |

案「Python 変換器」は Vision / MTP の取得スクリプト (`Scripts/vision/fetch_vision_weights.py`,
`Scripts/mtp/fetch_draft_weights.py`) と同じ立て付けなので、リポジトリの慣習にも合う。

### 4-2. 変換器 `Scripts/qwen35/convert_mlx.py` の仕様

**出力は MLX 規約の safetensors + `config.json` + `model.safetensors.index.json`。**
既存の `IndexLoader` / `RepackPlanner` / `ArchInfo` が読める形にする。

**(a) 量子化 (affine 4-bit, group 64, 入力次元方向にグループ):**

```
行 r の入力次元 K を 64 ずつに切る。各グループで
  lo, hi = min(w), max(w);  scale = (hi-lo)/15;  bias = lo
  q = clip(round((w - bias)/scale), 0, 15)
uint32 に 8 個ずつリトルエンディアンで詰める (MLX の並び)。
scales / biases は bf16。
```

`K % 64 == 0` は全対象で成立 (2048, 4096, 512, 1024 のいずれか)。**実測(上流) の shape で確認済み。**

**(b) 量子化する / しない の割り当て:**

| テンソル | 扱い | 理由 |
| --- | --- | --- |
| `embed_tokens`, `lm_head` | int4 g64 | Gemma と同じ。ただし **tie していないので 2 本ぶん** |
| `q/k/v/o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj` | int4 g64 | |
| `shared_expert.{gate,up,down}_proj` | int4 g64 | |
| `experts.gate_up_proj`, `experts.down_proj` | int4 g64 | |
| `mlp.gate` (router) | **int8 g64** | Gemma と同じ方針。40 本で 22 MB |
| `in_proj_a`, `in_proj_b` (`[32,2048]`) | **fp16** | 出力 32 次元。softplus の中身なので誤差が指数に効く |
| `A_log`, `dt_bias` (`[32]`) | **fp32** | 同上。`exp(A_log)` を取る |
| `conv1d.weight` (`[8192,4]`) | **fp16** | 32,768 個。量子化する価値が無い |
| すべての norm | **bf16** | 既存と同じ |
| `shared_expert_gate` (`[1,2048]`) | **fp32** | 1 行 |

**(c) 変換時に焼き込む (カーネルを増やさないための前処理):**

1. **`Qwen3_5MoeRMSNorm` 系の weight に `+1.0` を足す。**`input_layernorm` /
   `post_attention_layernorm` / `q_norm` / `k_norm` / `language_model.norm` /
   `mtp.norm` / `mtp.pre_fc_norm_*`。→ 既存 `rmsnorm_bf16w` がそのまま使える
2. **`linear_attn.norm.weight` には足さない。**`RMSNormGated` は `1+w` 規約ではない
3. **`q_norm.weight` に `head_dim ** -0.5 = 1/16` を掛ける。**q_norm は RoPE の直前で、
   RoPE は回転 (ノルム保存) なので順序を入れ替えてよい。→ attention の `scale` を **1.0** に
   固定でき、`attention_prefill_full_tensorops_2d_validity_v2` の `scale==1.0` ゲートを
   将来使える余地が残る (§8-3)
4. **`experts.gate_up_proj` を gate / up に切る。**`[256, 1024, 2048]` の行 0..511 と
   512..1023 に分け、Gemma と同じ 3 ロール (`gate` / `up` / `down`) で出す。
   → `RepackPlanner.planLayerFile` の「ロール 3 本」ループが**無改造で動く**
   (**未確認 §0-A #1**: 連結順序は Phase 0 で必ず確かめる)

**(d) 名前:** `RepackPlanner.classify` は `language_model.` 接頭辞を要求する。
Ornith は `model.language_model.…` なので `model.` を剥がす (Vision / Draft と同じ
`strippedNamePrefix` の考え方)。routed expert は
`language_model.layers.{i}.mlp.experts.switch_glu.{gate,up,down}_proj.weight` に**寄せる**
(既存の `routedExpertRole` が `.experts.switch_glu.` を見ているため)。
`lm_head.weight` は tie していないので `language_model.lm_head.weight` として常駐に入れる。

> 寄せるか、`RepackPlanner` 側に Qwen の名前を足すかは趣味の問題だが、
> **寄せるほうが Swift の差分がゼロになる。**本 PLAN は寄せる案を採る。

### 4-3. ディスクとネットワーク

```
for shard in 1..16:
    range GET でシャードを落とす (約 4.5 GB)
    ヘッダを読み、テンソルごとに量子化して出力シャードに追記
    入力シャードを消す
最後に index.json / config.json を書く
```

ピーク使用量 = 入力 1 シャード (4.5 GB) + 出力累計 (最大 19.5 GB) ≒ **25 GB**。
`DiskSpaceChecker` の予約と合わせて、**空き 30 GB を install の門にする。**

---

## 5. 新規に書くカーネル

**方針: 新規は全部 `Sources/TurboFieldfare/Metal/Qwen/qwen.metal` に置く。**
既存 `.metal` を触らない (Gemma の実測値を凍結したままにするため)。
SiLU だけは共有ヘッダに置きたくなるが、`gelu_pytorch_tanh` が 4 箇所に重複定義されている
現状に合わせて **`qwen.metal` にローカル定義する**。

### 5-1. `qwen_silu_mul` — SiLU

`grep -rni silu Sources/TurboFieldfare/Metal` は **0 件** (実測 = ソース)。
`gelu_mul_fp16` (`utility.metal`) の SiLU 版を書く: `y = x * sigmoid(x) * up`。
shared expert (int4) と routed expert (phase1) の両方から呼ぶ。

### 5-2. `qwen_rope_partial` — Qwen 規約の partial RoPE

```
rotary_dim = 64,  half = 32
pair < half:  x0 = h[pair], x1 = h[half + pair]
              freq = pow(theta, -2*pair / rotary_dim)      // 分母は 64
              回転
pair >= half: そのまま (h[64..255] は無変更)
```

既存 `fused_rope_neox_pair` との違いは §1-3。**流用禁止。**
`q_norm` / `k_norm` と融合して `qwen_qkv_epilogue` にまとめる (Gemma の
`fused_qkv_epilogue` と同じ構造: q 16 head + k 2 head + v 2 head = 20 threadgroup)。
**v には norm も RoPE も掛からない** (Gemma は v に no-scale norm を掛けていたので違う)。

### 5-3. `qwen_attn_output_gate` — 出力ゲート

`o[h,d] *= sigmoid(gate[h,d])`。gate は `q_proj` 出力の後半 4096 次元。
`o_proj` の直前に 1 dispatch。decode では `fused` に畳んでよい (4096 要素の elementwise)。

### 5-4. `qwen_moe_shared_gate` — shared expert の sigmoid ゲート

`shared_out *= sigmoid(dot(shared_expert_gate, x))`。1 行 GEMV + スカラー乗算。
`SharedExpertInt4` の後に追加する。

### 5-5. `qwen_delta_conv_step` / `qwen_delta_conv_chunk` — 因果 depthwise conv

チャネル 8192、カーネル 4、groups=8192、bias 無し、直後に SiLU。
decode は状態 `[8192, 3]` を持ち回して 1 step 更新。prefill は §5-6 の中に畳む。

### 5-6. `qwen_delta_rule` — 本命

**設計 (M3 Pro 向けに 1 案に決め打つ):**

- **threadgroup = (head h, v ブロック)。**`v_head_dim=128` を 32 ずつ 4 分割 →
  `32 head × 4 = 128 threadgroup`
- 各 threadgroup が `S[128 (k), 32 (v)] fp32 = 16,384 B` を **threadgroup memory に常駐**させる
  (Apple の上限 32 KiB に対し半分。1 コアに 2 TG 乗る)
- k 方向 (128) の縮約は threadgroup 内で完結する。v は threadgroup ごとに独立
- **チャンクのトークンをカーネル内部のループで順に舐める。**`S` は 1 度もメモリに出ない。
  最後に状態バッファへ書き戻すだけ
- 1 dispatch = 1 層ぶんの prefill チャンク全部。decode は同じカーネルに `T=1` で入る

**なぜ chunkwise (WY / UT 変換) を最初から書かないか:**

prefill 2048 トークンの線形注意の総計算量は **193 GFLOP** (**導出**、§3-6)。
MoE の 4.63 TFLOP に対して **4%** しかない。逐次形の弱点は FLOP ではなく
「トークンごとの threadgroup barrier が 2048 回直列に並ぶ」ことだが、
1 層あたり **導出で 0.4 ms 前後**、30 層で **12 ms** の見込み。
prefill 全体が秒オーダーなので**最初は逐次で足りる。**

**中止線: 逐次形が 30 層合計で 150 ms を超えたら chunkwise に切り替える** (§13)。

### 5-7. `qwen_delta_norm_gate` — RMSNormGated

head_v_dim=128 の RMSNorm (**`+1` しない**) → `* silu(z)`。
`z` は `in_proj_z` の出力 4096 を 32×128 に見たもの。

### 5-8. `qwen_lm_head_greedy` — LM head

`LMHeadChainInt4` は `realDecodeD=2816` / `realDecodeVocab=262144` を関数定数で焼いている
(実測 = ソース)。**`D=2048` / `vocab=248320` の specialization を足す。**
`vocab_size=248320` に対し実際に使われる ID は 248,076 までなので、
**248,076..248,319 の 244 行は学習されていない。argmax / sampling で -inf にマスクする**
(コストはほぼゼロ、事故は防げる)。

### 5-9. 要らないもの

- **logit softcap** — Qwen には無い。`LogitOutput` / `Sampler` の無条件適用を切る
- **埋め込みの `sqrt(hidden)` スケール** — Gemma だけの規約。`embedInt4` が融合しているので
  Qwen 経路では 1.0 を渡す (または専用パスを足す)
- **mRoPE** — テキストのみなら恒等 (§1-3)。Vision まで不要

---

## 6. ランタイムの構造変更

### 6-1. アーキテクチャの選択を実体化する

現状は `Model.load(… expecting: ArchConfig = .gemma4_26B_A4B)` の**デフォルト引数 1 個**が
唯一の選択点で、CLI / Server / App はどれも引数を渡していない (実測 = ソース)。

```swift
public enum ModelFamily: String, Sendable { case gemma4, qwen35moe }
```

1. `manifest.arch.family` を新設 (無ければ `gemma4`)。`versionMinor` を **3** に上げる
2. `Model.load` は `expecting:` を捨て、**manifest の family でベースラインを選ぶ**
3. `ArchConfig.qwen35_35B_A3B` を `ModelTypes.swift` に追加
4. `fullAttentionLayerMask: [UInt8]` を **`layerKinds: [UInt8]` (0=sliding, 1=full, 2=linear)** に
   改名。gemma4 の manifest を読むときは旧フィールドから写す

### 6-2. forward runner は分ける

**`RealForwardRunner` (3,300 行) は触らない。`QwenForwardRunner` を新設する。**

理由: `RealForwardRunner` の decode ループは Gemma 専用に手で畳んだ
コマンドバッファ 3 本のパイプライン (CB1 → shared/routed を跨いで次の層と重ねる) で、
そこに第 3 の層種別を差し込むと **Gemma 側の実測値が動く。**この
リポジトリの資産は「測った数字」なので、動かさないほうが安い。

共有するもの: `Model` / 各 streamer / `ExpertCacheBudget` / `MoE` / `SharedExpertInt4` /
`Sampler` / `LMHeadChainInt4` / prefill の GEMM 一式 / `Attention` の汎用 PSO。

`QwenForwardRunner` の 1 層 (線形注意層、decode):

```
h_in = rmsnorm_bf16w(h, input_layernorm)
qkv  = int4_gemv(in_proj_qkv, h_in)                 # 8192
z    = int4_gemv(in_proj_z,  h_in)                  # 4096
a,b  = fp16_gemv(in_proj_a/b, h_in)                 # 32 ずつ
qkv  = qwen_delta_conv_step(qkv, convState)         # + silu
o    = qwen_delta_rule(qkv, a, b, A_log, dt_bias, S)
o    = qwen_delta_norm_gate(o, z, norm)
h   += int4_gemv(out_proj, o)
h_r  = rmsnorm_bf16w(h, post_attention_layernorm)
… MoE (既存の router / shared / routed をそのまま) …
h   += moe_out
```

full attention 層は `qkv` が `q_proj/k_proj/v_proj` になり、
`qwen_qkv_epilogue` → `Attention.encodeFull` → `qwen_attn_output_gate` → `o_proj` に変わる。

**MoE 部分は Gemma と同一の呼び出し**にできる (§8-1)。ここが本 PLAN 最大の再利用点。

### 6-3. 状態の置き場所 — `RecurrentStateManager` 新設

`KVCacheManager.LayerKind` は `{swa, full}` の 2 値で、
`kSlot` / `vSlot` / `stride` / `capacity` がすべて「トークン添字のキャッシュ」を前提にしている
(実測 = ソース)。**線形注意層には居場所が無い。**

```swift
final class RecurrentStateManager {
    // 層 → S:[32,128,128] fp32 と conv:[8192,3] fp32
    // 62.8 MiB を 1 本の MTLBuffer に連結、層ごとにオフセット
    func reset()                       // 会話境界。ゼロ埋め
    func snapshot(into: Handle)        // 62.8 MiB のコピー
    func restore(from: Handle)
}
```

`KVCacheManager` 側は `layerKinds[l] == .linear` の層に**バッファを 1 バイトも確保しない**。
`ExpertCacheBudget.kvCacheByteEstimate` も同じ規則で数え直す
(**線形層を勘定に入れると 18 GB 機で false negative が出る**)。

### 6-4. 巻き戻せないという性質 (重要)

再帰状態は**途中のトークンを捨てられないし、巻き戻せない。**
影響するのは 3 箇所:

| 場所 | 現状 | Qwen での扱い |
| --- | --- | --- |
| `KVCacheManager.maximumSafeRewind` | SWA リングの容量から巻き戻し可能量を出す | 線形層は **0**。巻き戻すならスナップショットが要る |
| 投機デコードの verify ブロック | k 行流して j 行だけ採用、KV は j でカーソルを戻す | **状態は k 行ぶん進んでしまう。**§10-3 に対策 |
| サーバーの prompt cache (LCP 再利用) | 共通接頭辞ぶんの KV をそのまま使う | **接頭辞末尾の状態スナップショット (62.8 MiB) を slot ごとに持つ**。§9 |

---

## 7. 性能リスク (すべて 導出 / 未確認、§11 で測る)

### 7-1. エキスパート母集団が 2.67 倍になる

| | Gemma 4 | Ornith |
| --- | ---: | ---: |
| エキスパート総数 | 30 × 128 = 3,840 | **40 × 256 = 10,240** |
| 1 層で 32 スロットが覆う割合 | 25% | **12.5%** |
| スロット 1 本の値段 | 96.1 MiB | **67.5 MiB** |
| 1 本のエキスパートが引かれる確率 | 8/128 = 6.25% | **8/256 = 3.13%** |
| 32 スロットのバイト | 3.00 GiB | **2.11 GiB** |
| **バイト等価なスロット数** | 32 | **45** |

`RuntimeConfiguration.allowedExpertCacheSlots = [8,16,24,32]` は
**Gemma の測定に基づくユーザー確定値** (2026-08-20)。Ornith の 32 スロットは
Gemma の 32 スロットと**同じ意味ではない** — バイトで 0.70 倍、母集団の被覆率で 0.5 倍。

**本 PLAN は「上限を上げろ」とは言わない。**運用点はユーザーの判断である。
本 PLAN が用意するのは判断材料だけ:

- `ExpertTelemetry.startTrace` の TSV を使えば、**モデルを回さずに**別スロット数の
  ヒット率を再計算できる (実測 = ソース)。Phase 6 でまずこれを取る
- 提案するのは `[8,16,24,32,48]` への**候補追加**であって既定変更ではない。
  48 スロット = 3.16 GiB で、Gemma の 32 スロット (3.00 GiB) とほぼ同じ footprint

### 7-2. prefill の GEMM 占有率が半減する

`PrefillRoutedGEMMPlanner` は「エキスパート 1 本の重みタイルを 64 行のトークンで使い回す」
ことを効率の根拠にしている (実測 = ソース)。

| チャンク幅 | Gemma のエキスパート 1 本あたり平均行数 | Ornith |
| ---: | ---: | ---: |
| 512 | 32 | 16 |
| 1024 | 64 | 32 |
| **2048 (既定)** | **128** | **64** |

既定の 2048 でも Ornith は**平均でちょうど 64 行 = ブロック 1 個ぶん**にしかならない。
分布は当然ばらつくので、**半端ブロックが Gemma より確実に増える。**

`allowedPrefillChunkTokens` の上限は 2048。4096 を候補に足せば平均 128 行に戻るが、
KV リングと prefill スクラッチが増える (Ornith は SWA リングが無いので、増えるのは
scratch と full 層の KV だけ = **導出で 4096 幅なら +84 MB**)。
**これも候補追加の提案に留める。**

### 7-3. 読み出しの本数が増える

decode 1 トークンあたりの routed expert fetch は `numLayers × topK`:
**240 本 → 320 本 (+33%)**。1 本あたりは 3.19 MiB → 1.69 MiB (**0.53×**)。
総バイトは 0.74 倍だが、**小さい読み出しが増えると NVMe の実効帯域は落ちる。**

- pread 腕は `concurrentPerform(iterations: misses.count)` なのでキュー深度は
  ミス数まで自然に伸びる (実測 = ソース)
- mmap 腕 (既定) は `requestResidency()` 1 発なので深度 1。だから `F_RDADVISE` を
  足してある。`RDAdviceAdaptivePolicyConfig.conservative` の `byteCap: 384 MiB` /
  `slowCallNanos: 1ms` は **Gemma の 3.19 MiB 単位で調律した値**なので、
  1.69 MiB 単位では**再調律が要る**

### 7-4. 先読みとルーター相関の前提が崩れる

`RouterPreviewProbe` の「層 L の hidden に層 L+1 の router を当てると 66% 当たる」は
**128-way の測定**。256-way ではランダム基準線が 6.25% → 3.13% に下がるので、
**「相関が有るか」の結論から取り直す。**`ExpertPrefetch` の `topN=1 / distance=1` も同様。

---

## 8. 既存資産の再利用可否

### 8-1. そのまま使える (差分ゼロ or ほぼゼロ)

| 資産 | 根拠 |
| --- | --- |
| `router_gemv_gemma4_r4` / `_bf16_r4` | `D` / `num_experts` は関数定数。int8 router 対応済み |
| `router_topk_select_k8` | top-8 固定、Qwen も top-8。softmax の式も等価 (§1-3) |
| `moe_phase1_gate_up_act_u16load` / `_subset` | `D` / `F` / `top_k` は関数定数。**活性化だけ SiLU に差し替え** |
| `moe_phase2_down_reduce_k8` | top-8 = 8 simdgroup 固定。一致 |
| 期待値の事前確保 | `maxRouterRows * 256 * Float` と `kPrefillRouterMaxExperts=256` が**既に 256** |
| `PreadExpertStreamer` / `MmapExpertMapping` / `ExpertCacheBudget` | `expertStride` / `expertsPerLayer` を layout.json から読む。次元のハードコード無し |
| `GTurboPackedExpertsLayoutV1` | 16 KiB 揃え。Ornith は 108 ページちょうど |
| `rmsnorm_bf16w` / `rmsnorm_no_scale` | `+1` を変換時に焼くので規約が一致 (§4-2c) |
| `attention_prefill_causal_qblock_d256` | **head_dim だけで選ばれる。**Qwen の full 層 (256) がそのまま当たる |
| `MPPPrefillInt4QMM` (Metal 4 tensor ops) | `K % 64 == 0` と group 64 のみが条件。2048 / 4096 / 512 すべて満たす |
| `dequant_int4_gemv_simd` 一式 | 次元は実行時引数 |
| `Attention` の汎用 PSO | `(256,16,2)` は専用化が無いので汎用に落ちる。**正しさは保たれる** |

### 8-2. 手を入れる

| 資産 | 内容 |
| --- | --- |
| `MoE.swift` の `realDecode*` 定数 | `(2048, 512, 8, 256)` の specialization を追加。無くても汎用 PSO で動く (**性能だけの話**) |
| `FusedQKVGEMV` | Gemma の `(4096,2048,2816)` / `(8192,1024,2816)` に対し Qwen は `(8192,512,2048)`。specialization 追加 |
| `Attention` | `(256,16,2)` の専用 PSO を足す。`kAttnMaxQPerKV=2` は SWA 用 GQA カーネル専用で、full 経路 (`attention_decode_partial`) は通らないので**触らない** |
| `LMHeadChainInt4` | `D=2048 / vocab=248320` の specialization (§5-8) |
| `LogitOutput` / `Sampler` | softcap を条件化 (Qwen は 0.0 = 無効) |
| `KVCacheManager` / `ExpertCacheBudget` | 層種別 3 値、線形層は KV を確保しない (§6-3) |
| `ManifestReader` | `family` で期待ベースラインを選ぶ |
| `RDAdvice` のポリシー定数 | 1.69 MiB 単位で再調律 (§7-3) |

### 8-3. 将来の伸びしろ (Phase 6 以降)

`attention_prefill_full_tensorops_2d_validity_v2` は
`kPrefillTensorOpsHeadDim=512` / `kPrefillTensorOpsOutputs=8` / `kPrefillTensorOpsKeys=64` を
焼いていて、Swift 側が `headDim==512 && numQHeads==16 && numKVHeads==2 && scale==1.0` で門を閉めている。

**Qwen の full 層は `16/2 = 8` で `kPrefillTensorOpsOutputs` が一致する。**
`scale` も §4-2c の q_norm 焼き込みで 1.0 にできる。
**`kPrefillTensorOpsHeadDim=256` の兄弟カーネルを作るだけで Metal 4 tensor op 経路が開く**
可能性がある。差分が小さいわりに効く候補として記録しておく (**未確認**)。

---

## 9. サーバーへの波及 (SPEC / CONFORMANCE との関係)

**本 PLAN はサーバー仕様を書き換えない。**`docs/serving/SPEC.md` が唯一の規範である。
ここに書くのは「Qwen を載せると SPEC のどの不変条件が前提を失うか」だけ。

| 論点 | 現状 | Qwen |
| --- | --- | --- |
| prompt cache の LCP 再利用 | 共通接頭辞ぶんの KV をそのまま残す | **線形層の状態は「接頭辞の終端でのスナップショット」でしか復元できない。**slot ごとに 62.8 MiB を持つか、共通接頭辞から再計算するかの二択 |
| 文脈シフト (古いトークンを捨てる) | SWA リングが自然に回る | **不可能。**線形層の状態から過去を引き算できない。溢れたら状態リセット + 再 prefill |
| `maximumSafeRewind` | リング容量 − window | 線形層は 0 |
| reasoning 分離 | `<|channel|>thought` の 3 状態機械 | `<think>` / `</think>` の**トークン ID 248068 / 248069** で切る別実装 (§10-4 ではなく §11-5) |

**推奨:** slot あたり 62.8 MiB のスナップショットを持つ。32 slot なら 2.0 GB。
18 GB 機では重い。**Phase 8 で「slot 数 × 62.8 MiB」を `ExpertCacheBudget` の勘定に入れ、
入らない構成は起動時に断る**という、既存と同じ作法にする。

---

## 10. MTP — 本モデル最大の追い風

### 10-1. ドラフターが同梱されている

Gemma 側は `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` を別途 pin して
取ってきていた (実測 = ソース)。**Ornith は `mtp.*` として同じチェックポイントに入っている。**
`mtp_use_dedicated_embeddings: false` なので `embed_tokens` と `lm_head` は本体と共用。

### 10-2. 全エキスパート常駐という手

MTP 層の routed expert は 256 本 × 1.6875 MiB = **453 MB**。
**これを丸ごと常駐にすれば、ドラフト 1 トークンの SSD I/O はゼロになる。**

Gemma のドラフターは 236 MB を常駐させていた (`DraftModelSource.expectedPayloadBytes`、
実測 = ソース) ので、453 MB は同じ桁の判断で通る。**推奨: `draft/draft_weights.bin` に
エキスパートも含めて全部入れ、resident index で持つ。**

### 10-3. verify ブロックと再帰状態 (§6-4 の続き)

k 行の verify ブロックを流して j 行だけ採用するとき、線形層の状態は k 行ぶん進んでいる。

| 案 | コスト | 判断 |
| --- | --- | --- |
| ブロック前にスナップショット、j < k なら復元して j 行を replay | 復元 126 MiB + **replay 1 トークンにつき線形層の重み 570 MB 再読み** | 却下。replay が高すぎる |
| **行ごとに状態を書き出し、j が決まったら j 番目を採用** | 追加書き出し `(k-1) × 62.8 MiB`。k=4 なら **188 MiB / ブロック = 導出 1.25 ms** | **採用** |
| 状態を fp16 で保存 | バイト半減 | 却下。採用された状態が丸め済みになり誤差が蓄積する |

ブロック 4 で平均 3 トークン採用なら **オーバーヘッドは 0.42 ms/token** (**導出**)。
1 トークン 15〜20 ms の予算に対して 2〜3%。**許容範囲だが、`DraftAcceptanceProbe` の
計器に「状態書き出しのナノ秒」を足して実測できるようにする。**

### 10-4. `DraftForward` は書き直し

Gemma のドラフターは (a) 4 層 (b) 本体の K/V を共有 (c) `k_proj`/`v_proj` を持たない
(d) sandwich norm (実測 = ソース)。**Ornith の MTP は全部違う** — 1 層、自前の
`k_proj`/`v_proj` (= 自前の KV キャッシュ 2 KiB/token が要る)、norm 2 本、MoE 付き。

```
h_mtp = mtp.fc( concat( pre_fc_norm_embedding(embed(x_{t+1})),
                        pre_fc_norm_hidden(h_t) ) )
h_mtp = decoder_layer(h_mtp)          # full attention + MoE-256
logits = lm_head(mtp.norm(h_mtp))     # lm_head は本体と共用
```

`GTurboManifestDraftV1` は `tieWordEmbeddings == true` を**無条件に要求**し、
`sharedSlidingKVLayer` / `sharedFullKVLayer` を必須にしている (実測 = ソース)。
**Qwen の draft セクションは別スキーマにする** (`draftFamily` で分岐)。

---

## 11. 段階計画

各 Phase は「出口条件」を満たすまで次へ行かない。**GPU を使う Phase は 3 以降。**

> **進捗 (2026-08-21 夜) — 詳細は §17:**
>
> | Phase | 状態 |
> | --- | --- |
> | Phase 0 事実確定 | **大半済み。**§16-3 / §16-5 で `gate_up` の順序・RMSNorm の `+1`・`conv1d` の軸順・`in_proj_a` の感度が確定。**残: 実活性での再測 (合成入力でしか測っていない)、`oQ4e-g64` 側の norm 規約の照合** |
> | Phase 1 変換 | **入力が MLX 形式で揃った** (候補 2 本、どちらも赤リスト 0 本)。**残: `q_norm` への `1/16` の焼き込み、名前寄せ、`.gturbo` への repack、`--verify-install`** |
> | Phase 2 以降 | 未着手。**GPU はまだ 0 回** |

### Phase 0 — 事実確定 (GPU 不要)

1. `Scripts/qwen35/dump_reference.py`: CPU / float32 の `transformers` で 8 トークンの
   プロンプトを 1 回流し、**層ごとの入出力・線形注意の状態・router の top-8・logits** を
   `.npy` で落とす (`Scripts/vision/dump_vision_fixtures.py` と同じ立て付け)
2. **`gate_up_proj` の連結順を確定する** (§0-A #1)。行 0..511 を gate として
   組み直した MoE が参照と一致するかで判定
3. `A_log` / `dt_bias` / `in_proj_a` の値域を出す。fp16 で足りるかの判断材料
4. 4-bit RTN を numpy で当てて、**層ごとの相対誤差と最終 logits の KL** を出す。
   affine と sym の両方

**出口:** fixtures がリポジトリに入り、`gate_up` の順序が確定し、
「4-bit で行けるか」に数字が付いている。

**中止線:** 4-bit affine で logits の top-1 一致率が参照に対し 95% を切ったら、
attention と `in_proj_*` を int8 に上げる案 (+約 580 MB) を先に評価する。

### Phase 1 — 変換 (GPU 不要)

`Scripts/qwen35/convert_mlx.py` (§4) → `TurboFieldfareRepack --source-snapshot` →
`.gturbo`。`ArchInfo.load` に Qwen の `config.json` パーサを足し、
`GTurboManifestArchV1` に `family` / `layerKinds` / `linearAttention` セクションを足す。

**出口:** `--verify-install` が緑。ファイルサイズが §3-3 の導出表と一致する
(**ここが導出を実測に変える最初の点**)。`expertStride == 1_769_472`。

### Phase 2 — カーネル (GPU を使うが、モデルは載せない)

§5 の各カーネルを `TurboFieldfareKernelCheck` で Phase 0 の fixtures に対して検証。
FP16 の誤差床は `Scripts/vision/fp16_error_floor.py` と同じ手続きで先に測る
(**カーネルのバグと丸め誤差を分離できる検証系を先に作る** — PLAN_VISION §6 の教訓)。

**出口:** 全カーネルが誤差床の中。特に `qwen_delta_rule` は
**2048 トークン流した後の状態**が参照と一致すること (1 トークンだけでは足りない)。

### Phase 3 — decode 結線

`QwenForwardRunner` の decode 経路のみ。prefill は off、投機も off、
`--temp 0` の greedy。

**出口:** 固定プロンプトから **64 トークン、参照 (CPU float32) と完全一致**。
一致しないなら層ごとの hidden を突き合わせて発散点を特定する。

**中止線:** 発散点が `qwen_delta_rule` の数値的な蓄積 (fp32 でも合わない) なら、
状態を fp64 相当 (2×fp32 の compensated summation) にする案を検討。それでも
合わなければ chunkwise 形へ (誤差の出方が変わる)。

### Phase 4 — prefill

チャンク幅 512 / 1024 / 2048。`qwen_delta_rule` の T>1 経路。

**出口:** prefill 経由の greedy 64 トークンが Phase 3 と一致。
**線形注意の 30 層合計が 150 ms 以内** (§5-6 の中止線)。

### Phase 5 — トークナイザ / テンプレート / CLI

- `verifyDecoderConfiguration` に **ByteLevel** を許す分岐 (現状は metaspace +
  ByteFallback + Fuse の 3 段固定を要求している)
- `GemmaDecoding` の兄弟として `ByteLevelDecoding` (GPT-2 の byte↔unicode 表)
- `GrammarVocabulary` の piece 復元をデコーダ種別で切り替える
- `vocabSize = 262_144` のリテラルを tokenizer から読むように
- 特殊トークン: `<|im_start|>` 248045 / `<|im_end|>` 248046 / `<|endoftext|>` 248044 /
  `<think>` 248068 / `</think>` 248069 / `<tool_call>` 248058。
  **停止トークンは `[248046, 248044]`** (`generation_config.json`、実測(上流))
- チャットテンプレートは**上流の `chat_template.jinja` をそのまま同梱する**。
  Gemma 側は「上流に chat_template が無い」から Swift で書いていた (実測 = ソース) が、
  Ornith は持っている。サーバーの redraw 不変条件用の兄弟 jinja は Phase 8 で
- ツール呼び出しは `<tool_call><function=名前><parameter=名前>値</parameter></function></tool_call>`
  という **XML 形** (実測(上流))。Gemma の `call:name{...}` とは別物なので
  パーサと GBNF ビルダを新設

**出口:** `TurboFieldfareCLI --model … --prompt …` が日本語と英語で通る。
`<think>` ブロックが `reasoning_content` に分離される。

### Phase 6 — 計測と運用点

`bench.sh` の作法をそのまま踏襲する。**temp 1.0 のまま、クールダウン 20 秒**
(採点は temp 0 / 10 秒)。GPU は 1 個だけ。

1. `ExpertTelemetry.startTrace` の TSV を 1 回だけ取り、**オフラインで**
   8/16/24/32/48 スロット × LFU/LRU のヒット率を再計算する (モデル再実行不要)
2. その結果を持って、運用点の候補をユーザーに出す。**既定は変えない**
3. チャンク幅 512/1024/2048 (と 4096 を評価するなら候補追加の可否を含めて) の A/B
4. `RDAdvice` のバイト上限を 1.69 MiB 単位で調律
5. `RouterPreviewProbe` を 256-way の基準線 3.13% に対して取り直す

**注意:** 反復 3 未満のセルには解釈を書かない。数字だけ置く。

### Phase 7 — MTP

§10。全エキスパート常駐 + 行ごと状態書き出し。

**出口:** `RESULTS_MTP.md` と同じ様式で tok/s / TTFT / peak の 3 点。
**受入は「非投機と greedy でバイト一致」** (Gemma 側の D5 不変条件と同じ)。

### Phase 8 — サーバー

§9。prompt cache のスナップショット方針を決め、`ExpertCacheBudget` に勘定を足す。
`docs/serving/SPEC.md` に Qwen 固有の行を足すかは、そこで別途判断する。

### Phase 9 — Vision

§12。

---

## 12. Vision (後回しの根拠と、やるときの中身)

**tower の形は Gemma と偶然ほぼ同じ** (どちらも SigLIP2-so400m-patch16 系):

| | Gemma 4 vision | Ornith vision |
| --- | ---: | ---: |
| hidden / 層 / head / FFN | 1152 / 27 / 16 / 4304 | **1152 / 27 / 16 / 4304** |
| patch | 16 | 16 |
| head_dim | 72 | 72 |

**しかし中身は別物である:**

| 項目 | Gemma 4 | Ornith |
| --- | --- | --- |
| norm | RMSNorm (bias 無し) | **LayerNorm (bias 有り)** — `norm1.bias` 等が実在 (実測(上流)) |
| qkv | 3 本に分離、bias 無し | **融合 1 本 `[3456,1152]` + bias** |
| 位置 | 加算テーブル `[2,10240,1152]` **かつ** 2D RoPE (θ=100) | **`pos_embed [2304,1152]` の補間 + mRoPE** |
| マージ | **3×3 平均プーリング** + 標準化 | **2×2 patch merger MLP** (`merger.linear_fc1 [4608,4608]` → `fc2 [2048,4608]`) |
| リサイズ | アスペクト比保存、48 の倍数 | **面積境界** (`shortest_edge 65536` = 256², `longest_edge 16777216` = 4096²) の smart resize |
| 時間軸 | 無し | `temporal_patch_size 2` (静止画は同じフレームを 2 枚) |
| LM 側の位置 | 1 次元 | **mRoPE (t,h,w)。ここで初めて §1-3 の interleaved mrope が要る** |

`VisionImagePreprocessor` には時間軸の概念が無く、`vision_pool_std_block` は 3×3 固定、
`vision_qk_norm_rope2d_block` は加算テーブル + 2D RoPE を前提にしている (実測 = ソース)。
**カーネルもパイプラインも書き直しになる。**再利用できるのは
`vision_bf16_qmm_f16` と `vision_attention_full_seg_d72` くらい。

**画像 1 枚のトークン数の上限に注意:** `longest_edge 16777216` px を素直に取ると
`16777216 / 16² / 2² = 65,536` トークンになる。**上限を切る** (Gemma は 280 だった)。

---

## 13. リスクと中止線

| # | リスク | 検知 | 中止線 / 代替 |
| --- | --- | --- | --- |
| 1 | 4-bit RTN の品質が足りない | Phase 0 #4 の KL / top-1 一致率 | attention と `in_proj_*` を int8 へ (+580 MB)。それでも駄目なら routed expert のみ int4、残り全部 int8 (+1.2 GB) |
| 2 | `qwen_delta_rule` 逐次形が遅い | Phase 4 の 30 層合計 | **150 ms 超で chunkwise (WY) へ。**その場合 Phase 4 の工数が 2 倍になる |
| 3 | 32 スロットでヒット率が落ちすぎる | Phase 6 #1 のオフライン再計算 | 候補スロット追加をユーザーに提案。それが通らないなら **decode の I/O 律速を受け入れて数字を出す** |
| 4 | prefill の半端ブロックで TTFT が悪化 | Phase 6 #3 | チャンク 4096 の候補追加。または `PrefillMoEGrouping.tileExpertCount` (現在 16) の見直し |
| 5 | 状態スナップショットで 18 GB を食い潰す | Phase 7 / 8 の `peak` | slot 数を絞る。サーバーは起動時に断る (既存の作法) |
| 6 | `gate_up_proj` が交互配置だった | Phase 0 #2 | 変換器で de-interleave するだけ。**工数は小さいが、見落とすと Phase 3 まで発覚しない** |
| 7 | `numExperts <= 256` の precondition に余裕が無い | — | 今回は一致するので通る。**将来 512 エキスパートのモデルを載せる余地は無い**と明記しておく |
| 8 | Gemma 側の実測値が動く | `Scripts/test.sh` と `bench.sh` | **`RealForwardRunner` を触らない**方針 (§6-2) を守る。触ったら Gemma のベンチを取り直す |

---

## 14. やらないこと

- **動画・音声。**`<|video_pad|>` / `<|audio_pad|>` は**明示的に拒否する** (黙って壊れない)
- **YaRN による 1M 文脈拡張。**`rope_type: default` のまま
- **Gemma 4 経路の性能変更。**共有コードに手を入れるときは Gemma のベンチを取り直す
- **2 つのモデルの同時ロード。**`.gturbo` は 1 プロセス 1 モデルのまま
- **ANE への退避** (`aneSharedExpert` フラグは今も死んでいる)
- **他の Qwen3.5 サイズへの一般化。**35B-A3B のこの形だけを見る
- **Mac アプリの UI 対応。**ビルドが壊れないことだけ守る (PLAN_VISION と同じ扱い)

---

## 15. oQ / oMLX から取り入れるもの (2026-08-21 追記)

入力: [`jundot/omlx`](https://github.com/jundot/omlx) (Apache-2.0) と、そこで作られた
[`scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp`](https://huggingface.co/scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp)。
以下はすべて **実測(上流)** — 公開チェックポイントの `config.json` / `index.json` /
5 シャードの safetensors ヘッダ / `oq_imatrix_report.json` を取得し、
`omlx/oq.py` と `omlx/custom_kernels/qwen35_prefill/gdn.py` を読んで確定した。

### 15-0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | §4 の前提 (「MLX 量子化版が無いので変換器を書く」) | **崩れた。**MLX 4-bit 版が既に存在する |
| 2 | 一番大きい収穫 | **routed expert 18,119,393,280 B が「4-bit affine group 64、gate/up/down の 3 ロールに分割済み、U32 + BF16 scales + BF16 biases」で、本リポジトリの `RepackPlanner.planLayerFile` が食う形と 1 バイトも違わない。**しかも §3-2 で導出した数字とバイト一致した |
| 3 | 二番目 | **その 18 GB は imatrix (校正データによる重要度重み) 付きで量子化されている。**素の RTN より確実に良い。**形式が同じなので、ランタイムは 1 行も変えずに品質だけ貰える** |
| 4 | 三番目 | **`gate_up_proj` の連結順の未確認 (§0-A #1) が事実上消えた。**MLX 側が既に `gate_proj` / `up_proj` に分割している |
| 5 | 四番目 | **MTP も vision も入っている。**`language_model.mtp.*` は switch_mlp が 4-bit g64 で 503 MB、vision は bf16 で 893 MB。§10-2 の「MTP エキスパート全常駐」が**そのまま成立する** |
| 6 | そのままは乗らない部分 | **248 本のテンソル (730M param / 533 MB) が group_size 128 か 5/6-bit。**本ランタイムは (a) アフィン group size がシェーダライブラリ全体の**コンパイル時定数** (32 か 64) で**モデル内混在が原理的に不可能** (b) 5-bit / 6-bit のカーネルが無い |
| 7 | 一番大きい設計上の収穫 | **Gated DeltaNet の prefill は「chunkwise (WY) より blocked-sequential のほうが速い」**というのが omlx の実装上の結論。§5-6 の判断が裏付けられ、**さらに良い並列化 (状態をレジスタに置き simd_shuffle で縮約、barrier ゼロ) の実物が読める** |
| 8 | ライセンス | omlx は **Apache-2.0**。本リポジトリと同じ。**設計を読んで写せる** |

### 15-1. oQ とは何をするものか

`docs/oQ_Quantization.md` と `omlx/oq.py` (8,389 行) から:

1. **校正推論で感度を測る (imatrix)。**
   `sensitivity = MSE(float_output, quantized_output) / mean(float_output²)`。
   出力の大きさで正規化するので、残差が積み上がる後段の層が不当に敏感に見えない。
   今回のアップロードは `oqe_code_multilingual` を 128 サンプル × 512 トークンで収集
   (`oq_imatrix_report.json`)。`Linear` 511 本 + `SwitchLinear` 123 本を計装している
2. **感度で bit を配る。**最大感度比 ≥50% → base+4 bit、≥20% → base+2、それ未満 → base+1。
   bpw 予算に入らなければ 8 → 6 → 5 と落とす貪欲割り当て
3. **固定の保護。**`lm_head` 8-bit / MoE router / vision / SSM パラメータ
4. **routed expert は base のまま。**「バイトあたりの品質改善効率が悪い」から昇格させない
5. **oQ+ は GPTQ を掛ける。**列ごとに丸め、逆ヘッセ行列で残り列を補正する。
   MoE 向けに**層内の全エキスパートで Hessian を共有したバッチ GPTQ** を持ち、
   Qwen3.5-35B-A3B で 90 分 → 6 分と書いてある

**本 PLAN にとって決定的なのは 1 と 5 の組み合わせである。**
「重要度で重みを付けた scale/bias 探索」も「GPTQ 補正」も、**書き出すバイトの値を変えるだけで
フォーマットを変えない。**4-bit g64 affine は 4-bit g64 affine のままである。

### 15-2. 公開チェックポイントの実体

`total_size = 21,613,054,816 B (21.61 GB)`、テンソル 2,052 本、5 シャード。

| 区画 | バイト | 割合 | 量子化 | 本ランタイム |
| --- | ---: | ---: | --- | --- |
| routed experts (`mlp.switch_mlp.{gate,up,down}_proj`) | **18,119,393,280** | 83.8% | **4-bit affine g64** | **そのまま乗る** |
| embed_tokens + lm_head | 1,080,688,640 | 5.0% | 8-bit affine g64 | 乗る (int8 経路あり) |
| core (attn / linear_attn / shared / router / norm) | 1,016,785,152 | 4.7% | **混在 (下記)** | **一部乗らない** |
| vision_tower | 893,142,496 | 4.1% | BF16 | 乗る (現行 tower も bf16) |
| MTP (`language_model.mtp.*`) | 503,045,248 | 2.3% | switch_mlp は 4-bit g64 | 乗る |

**routed expert 1 本のバイト配置** (実測(上流) のヘッダから):

```
gate_proj  weight U32  [256, 512, 256]   scales/biases BF16 [256, 512, 32]
up_proj    weight U32  [256, 512, 256]   scales/biases BF16 [256, 512, 32]
down_proj  weight U32  [256, 2048, 64]   scales/biases BF16 [256, 2048, 8]
→ 1 エキスパート = 589,824 × 3 = 1,769,472 B = 16 KiB × 108
```

**§3-2 で導出した 1,769,472 B / 108 ページと完全一致した。**
40 層合計 18,119,393,280 B も導出値と一致。**§3-2 は導出から実測(上流)に格上げできる。**

**乗らない 248 本** (`config.json → quantization` の per-tensor override と
ヘッダの shape から確定):

| テンソル | 本数 | 指定 |
| --- | ---: | --- |
| `linear_attn.out_proj` | 30 | 5-bit **g128** |
| `linear_attn.in_proj_z` | 29 | 5-bit **g128** |
| `linear_attn.in_proj_a` / `in_proj_b` | 21+21 / 8+8 | 5-bit **g128** / 6-bit g64 |
| `linear_attn.in_proj_qkv` | 3 / 1 | 5-bit **g128** / 6-bit g64 |
| `mlp.shared_expert.{gate,up,down}_proj` | 120 | 8-bit **g128** |
| `self_attn.{q,k,v}_proj` | 3 | 6-bit g64 / 6-bit **g128** |
| **合計** | **248** | **730,464,256 param / 533,434,368 B** |

なぜ g128 が混ざるか: `oq.py` の `gs()` が **`num_experts >= 150` なら 128 を返す**。
Ornith は 256 なので、routed expert 以外は自動的に g128 側に倒れている。
routed expert だけは予算計画側の経路が base (`group_size 64`) を保っている。

### 15-3. 取り込み方 — 3 案

> **この表は「どのチェックポイントを入力にするか」の話である。**
> §4-1 の「量子化器をどこに置くか」とは別の軸。**以降この 3 つは記号ではなく名前で呼ぶ**
> (旧表記との対応: 案「oQ を自分で回す」= 旧 A″、案「公開版を打ち直す」= 旧 A‴、
> 案「素の RTN を書く」= 旧 A)。

| 案 | 中身 | ダウンロード | 工数 | 品質 |
| --- | --- | ---: | --- | --- |
| **案「oQ を自分で回す」** | **oQ を制約付きでローカル実行する。**`gs()` を 64 固定にし、許可 bit を {4, 8} に狭めた上で imatrix を有効にする | bf16 71.9 GB | 中 (oq.py へのパッチ + MLX を回す) | **最良** |
| **案「公開版を打ち直す」** | **公開チェックポイントを食い、非互換 248 本だけ打ち直す。**5/6/8-bit g128 を bf16 に戻して **8-bit g64** で再量子化する | **21.6 GB** | **小** | 良 (18 GB 側は imatrix そのまま。248 本は 5-bit 由来を 8-bit に上げるので、再量子化の損失はほぼ無い) |
| 案「素の RTN を書く」(§4) | numpy で素の RTN 変換器を書く | 71.9 GB | 小 | 素の RTN。imatrix 無し |

> **実行結果 (2026-08-21 夜): 案「公開版を打ち直す」は実行済み。**§17-2 を見ること。
> 打ち直しの入力は「5/6/8-bit g128 を bf16 に戻す」のではなく
> **上流 bf16 の原本を HTTP Range で取った**ので、逆量子化の往復が無くなった。

**推奨は 案「公開版を打ち直す」を先に、案「oQ を自分で回す」を後に。**理由:

- 案「公開版を打ち直す」は**ダウンロードが 1/3 になり、GPU も要らず、18 GB の imatrix 品質をそのまま貰える。**
  Phase 1 を最短で通せる
- 打ち直す 248 本は **730M param、全体の 2.0%** にすぎない。
  8-bit g64 で **776 MB**、oQ の現状 533 MB に対し **+243 MB**。
  install 合計は **21.61 → 約 21.86 GB** (**導出**)
- 4-bit g64 に落とす手もあり **411 MB (−123 MB)** になるが、
  **oQ がわざわざ 5〜8 bit に昇格させた 248 本を 4-bit に落とすのは、
  感度測定の結論を捨てることに等しい。**上げる側に倒す
- 案「oQ を自分で回す」は「校正データを差し替えたい」「GPTQ も掛けたい」となったときに戻ってくる道。
  imatrix 収集はモデルを回す = GPU を使うので、**Phase 6 以降の作業**

**§4-2 の変換器は捨てない。**案「公開版を打ち直す」の「248 本を打ち直す」部分がそのまま §4-2(a) の量子化器であり、
§4-2(c) の焼き込み (RMSNorm の +1 / `q_norm` に 1/16 / `linear_attn.norm` は +1 しない) は
**どの案でも必要**である。MLX 側は焼き込みをしないので、`.gturbo` に書く前に本リポジトリ側でやる。

**名前の対応** (§4-2(d) を上書きする):

| oQ の名前 | 本リポジトリ |
| --- | --- |
| `language_model.model.layers.{i}.…` | `classify` の `language_model.` 接頭辞に**そのまま合致する** |
| `…mlp.switch_mlp.{gate,up,down}_proj` | `routedExpertRole` が見る `.experts.switch_glu.` を **`.mlp.switch_mlp.` にも当てる** (1 文字列) |
| `vision_tower.…` | `isMultimodalTensorName` の既存接頭辞と**一致する** |
| `language_model.mtp.…` | draft セクションへ (§10-4) |
| `language_model.lm_head` | tie しないので常駐に入れる |

### 15-4. Gated DeltaNet カーネル — omlx の実装を写す

`omlx/custom_kernels/qwen35_prefill/gdn.py` は **2 本**持っている:

- `gated_delta_chunked_metal` — chunkwise (WY)、`CHUNK = 64`
- `gated_delta_blocked_seq` — **blocked-sequential (厳密な逐次漸化式)**

ソースのコメントが判断を書いている: **「chunked 再定式化なしの厳密漸化式 = WY 経路の半分の FLOP」**、
そのうえで Apple GPU 向けに組み直したのが後者。**§5-6 で採った「まず逐次」は正しかった。**
ただし並列化の切り方は omlx のほうが良い。**そちらを写す。**

**omlx の Kernel S の幾何 (実測(上流) = ソース):**

```
TB = 32 (時間ブロック),  DB = 32 (threadgroup が持つ dv 行数),  256 スレッド
grid = (Dv/DB = 4, Hv = 32, B = 1)  →  128 threadgroup
thread → dv = tid/8 (0..31),  seg = tid%8,  d0 = seg*16
状態は【レジスタ】: float4 st[4] = 16 float/thread  (256 × 16 = 32 × 128 ✓)
threadgroup memory: k_s[32][Dk+8], q_s[32][Dk+8], v_s[32][DB+8], g_s[32], b_s[32]
                    ≒ 20 KB (上限 32 KiB に収まる。+8 はバンク衝突回避)
```

肝は 3 つ:

1. **状態をレジスタに置く。**私が §5-6 で書いた「threadgroup memory に 16 KiB」より良い。
   `float4 st[4]` は 16 レジスタで、占有率をほとんど落とさない
2. **縮約は `simd_shuffle_down(4→2→1)` + `simd_shuffle` の同報。
   ホットループに `threadgroup_barrier` が 1 個も無い。**barrier は時間ブロックの境界だけ
3. **k/q/v を時間ブロック単位で threadgroup memory に協調ロードする。**
   コメント曰く、素の mlx_lm カーネルは dv を Dv/4 で切るので同じ k/q 行を **32 回**読み直し、
   16k トークンの 1 層で **約 13 GB** の冗長トラフィックになる。
   Dv/16 で切って staging すると **8 分の 1** になる。
   **本 PLAN の §3-5 は「状態往復 126 MiB/token」しか数えていなかったが、
   切り方を間違えると k/q の読み直しがそれを桁で超える。**

さらに小技: **減衰 `st *= gt` を `kv_mem` の縮約と同じループで掛ける** (2 パスにしない)。
出力書き出しは `seg == 0` のスレッドだけ。

**§5-6 の設計を上記で置き換える。**時間ブロックは `(16, 32, 48)` が可変になっているので、
本機では 3 通り測る (Phase 4)。

**§13 の中止線 #2 (「150 ms 超で chunkwise へ」) は据え置く**が、
omlx の記述どおりなら **chunkwise は FLOP が 2 倍で、逃げ道としての魅力は薄い。**
その場合の代替は「TB を振る」「Dv の切り方を変える」を先に試すこと。

### 15-5. その他、確認できた細部

| 事実 (実測(上流)) | 本 PLAN への影響 |
| --- | --- |
| `mlp.gate` (router) は **BF16 のまま量子化されていない** | 既存の `router_gemv_gemma4_bf16_r4` 経路 (QAT 用に用意されたもの) が**そのまま当たる**。§4-2(b) の「router を int8」は**不要**。22 MB → 84 MB になるが、カーネルが 1 個減る |
| `A_log` / `dt_bias` / `conv1d.weight` / 全 norm が BF16 | §4-2(b) の割り当てと一致 |
| `in_proj_a` / `in_proj_b` は 5〜8 bit に量子化されている | §4-2(b) で fp16 にした判断は保守側。**そのままでよい** (3.93M param = 7.9 MB) |
| `mtp.fc.weight` は BF16 (16.8 MB) | 量子化しない |
| `conv1d.weight` の shape が `[8192, 4, 1]` (上流 bf16 は `[8192, 1, 4]`) | **MLX 変換で軸が入れ替わっている。**読み込み側で必ず確認する (**新しい落とし穴**) |
| imatrix レポートの `expert_coverage`: 123 モジュール × 256 = 31,488 エキスパート、**全部が校正で発火**、`min_count 32` / `median 1678` / `max 29349` | **ルーターの偏りの一次資料。**median の 17 倍が max。§7-4 の「256-way でも相関があるか」を測る前の事前分布として使える |
| omlx は ANE prefill 経路 (`qwen35_ane.metal` / `.mm` 123 KB) を持つ | **本 PLAN では扱わない** (§14)。存在だけ記録 |
| omlx は quantized KV (`turboquant_kv.py`) を持つ | 本ランタイムは **TurboQuant KV を削除済み**で manifest 側で拒否している。**逆方向。追わない** |

### 15-6. 取り入れないもの

- **mxfp4 / mxfp8 モード** — `_bytes_per_group` にあるが、今回のチェックポイントは
  全部 `affine`。int4/int8 affine 以外のカーネルは持たない
- **oQ2〜oQ3 の低ビット帯** — 18 GB を 3.5 bpw に落とせば 14 GB になるが、
  本ランタイムに 2/3 bit のカーネルが無い。**候補としてだけ記録**
- **oMLX の連続バッチ / 階層 KV / マルチモデル配信** — 本リポジトリの設計思想と別物
- **per-expert の bit 混在** — `expertStride` が manifest 単一値で、
  `expertOffset = layer*expertsPerLayer*stride + expert*stride` の等間隔前提が壊れる。
  **原理的に不可能。**oQ 側も routed expert は一律 base bit なので衝突しない

### 15-7. 本文への差し戻し

| 節 | 変更 |
| --- | --- |
| §0 #3 | 「量子化器を書くのが Phase 1 の主題」→ **「公開 oQ チェックポイントを食い、非互換 248 本だけ打ち直すのが Phase 1 の主題」** |
| §0-A 未確認 #1 (`gate_up` の順序) | **事実上解決。**MLX が分割済み。ただし Phase 0 で 1 度だけ突き合わせる |
| §0-A 未確認 #2 (4-bit の品質) | **imatrix 付きが手に入るので前提が良くなった。**それでも測る |
| §3-2 / §3-3 | 導出値が実測(上流)とバイト一致。install は **約 21.9 GB** (案「公開版を打ち直す」の場合) に更新 |
| §4 | 案「Python 変換器」の上に **案「公開版を打ち直す」を第一候補**として置く |
| §4-2(b) | router は **BF16 のまま**でよい (int8 にしない) |
| §5-6 | **omlx の blocked-sequential の幾何で置き換え** (§15-4) |
| §10-2 | MTP 全常駐 = **453 MB** が実在のバイトで裏付けられた |
| §11 Phase 1 | ダウンロード 71.9 GB → **21.6 GB**、ディスク門 30 GB → **26 GB** |

---

## 16. ローカル取得済みチェックポイントの検証 (2026-08-21)

`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` をダウンロードし、全 4 シャードの safetensors
ヘッダをローカルで解析した。**ここから先は 実測(手元) である**(GPU は使っていない。
ファイルを読んだだけ)。

### 16-1. pin 情報

```
repoID            ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit
revision          19504d912fa8fc7622bf6b1de3db5d5d890b1f02
sourceIndexSHA256 c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36
local             ~/LLM/Ornith-1.5-35B-A3B-MLX-4bit
```

完全性: リモートのファイル一覧と差分なし。シャード 4 本、テンソル 1,757 本、
`index.metadata.total_size = 19,508,787,456 B` に対し実ファイル合計 19,509,024,201 B
(差はヘッダぶん)。

### 16-2. 形式 — **打ち直し 0 本**

| | 本数 |
| --- | ---: |
| 4-bit affine **group 64** (scales/biases つき) | 432 |
| 8-bit affine **group 64** | 80 |
| **group ≠ 64 または bits ∉ {4,8}** | **0** |

**§15-3 の案「公開版を打ち直す」で必要だった「248 本の打ち直し」が、この版では 1 本も要らない。**
8-bit の 80 本は `mlp.gate` (router) 40 本と `mlp.shared_expert_gate` 40 本ちょうど。

routed expert 1 本 = `gate/up/down × (weight 524,288 + scales 32,768 + biases 32,768)`
= **1,769,472 B = 16 KiB × 108**。§3-2 の導出と**バイト一致**(3 回目の一致)。
40 層 × 256 本 = **18,119,393,280 B**、スロット 1 本 = **67.5 MiB**。

`.gturbo` に落としたときの導出サイズ: `packed_experts` 18.12 GB +
`model_weights.bin` **1.39 GB** = **19.51 GB**。§3-3 の修正値 (1.40 GB) と一致。

### 16-3. 確定した事実 (本文の未確認・落とし穴を潰したもの)

| 論点 | 結果 |
| --- | --- |
| §0-A #1 `gate_up_proj` の連結順 | **解決。**MLX が `switch_mlp.gate_proj` / `up_proj` に分割済み。3 ロールのまま `RepackPlanner.planLayerFile` に入る |
| §4-2(c)#1 RMSNorm の `+1` | **既に焼かれている。**`input_layernorm` の平均 +1.031 (min +0.922)、`model.norm` の平均 +2.640。Qwen の生の重みは 0 中心なので、MLX 変換側が `1+w` を畳んでいる。**こちらで足してはいけない** |
| §4-2(c)#2 `linear_attn.norm` は `+1` しない | **正しい。**平均 +0.884 (min +0.523) と 1 未満に散っており、`RMSNormGated` の素の `w` のまま |
| §15-5 `conv1d` の軸順 | **`[8192, 4, 1]`。**上流 bf16 の `[8192, 1, 4]` から入れ替わっている。落とし穴として確定 |
| router のビット幅 | **8-bit g64。**既存の `router_gemv_gemma4_r4` (int8) が既定でそのまま当たる。**§15-5 の「router は BF16 なので int8 不要」は oQ 版の話で、この版には当てはまらない** |
| 非量子化のまま残るもの | 全 layernorm / `q_norm` / `k_norm` / `linear_attn.norm` / `A_log` / `dt_bias` / `conv1d` (すべて BF16)。§4-2(b) の割り当てと一致 |
| tokenizer | `decoder.type = ByteLevel`。**`verifyDecoderConfiguration` が要求する `Sequence[Replace(▁→" "), ByteFallback, Fuse]` と一致しないので確実に弾かれる** (§11 Phase 5 の裏付け)。BPE 語彙 248,044 + added 33 = 248,077 に対し `vocab_size` は 248,320 なので**末尾 243 行は未学習** (§5-8 のマスクは要る) |
| vision / MTP | **入っていない** (0 本)。テキスト専用の変換 |
| tie_word_embeddings | false。`lm_head` は独立して存在 |

### 16-4. 品質面の懸念 — この版だけでは足りない可能性がある

**形式は完璧だが、ビットの配り方は攻めている。**oQ が感度測定にもとづいて保護した
テンソルのいくつかが、この版では base の 4-bit のままである:

| テンソル | この版 | oQ4e | 懸念 |
| --- | --- | --- | --- |
| `linear_attn.in_proj_a` / `in_proj_b` | **4-bit g64** | 5-bit g128 / 6-bit g64 / 8-bit g64 | **最も危ない。**`g = -exp(A_log) · softplus(a + dt_bias)` で、実測 `A_log ∈ [-3.97, +4.25]` → `exp(A_log)` は最大 **70.1**、`dt_bias ∈ [-7.31, +15.56]`。**`a` の相対誤差が 70 倍に増幅されて減衰項の指数に入る** |
| `embed_tokens` / `lm_head` | 4-bit g64 (572 MB) | 8-bit g64 (1.08 GB) | logits の質。投機デコードの受理率に直結する (§10) |
| `mlp.shared_expert.{gate,up,down}` | 4-bit g64 | 8-bit g128 | 全トークンが通る経路。oQ は明示的に「精度の床」として保護している |

`in_proj_a` / `in_proj_b` は **30 層 × 2 本 = 3.93M param しかない。bf16 で持っても 7.9 MB。**
4-bit から bf16 に戻しても**失われた精度は戻らない**ので、直すなら供給源が要る:

- **上流 bf16 (71.9 GB) から該当 60 本だけ取り出す** — 最良。range 取得で数 MB で済む
- **oQ4e (21.6 GB) の 5/6/8-bit を使う** — 次善

**→ 測定した。結果は §16-5。8-bit への打ち直しを決定した (費用 +2 MB)。**
供給源の bf16 は §16-5 で範囲取得済みなので、上流 bf16 の全体は要らない (§16-6)。

### 16-5. bf16 からの範囲取得と、減衰ゲートの実測 (2026-08-21)

`ornith-ai/Ornith-1.5-35B-A3B` (bf16) から **421 + 30 本 / 2.30 GB (全体の 3.20%)** を
HTTP range で抜き出した。80 レンジに結合して転送し、無駄 0.0%。**実測(手元)。**

```
~/LLM/Ornith-1.5-35B-A3B-bf16-partial/model.safetensors        421 本 2.30 GB
~/LLM/Ornith-1.5-35B-A3B-bf16-partial/model-extra.safetensors   30 本 7,680 B (linear_attn.norm)
  A 減衰ゲート系 (in_proj_a/b, A_log, dt_bias, conv1d)  150 本    9.8 MB
  B shared expert (gate/up/down + shared_expert_gate)   164 本  258.1 MB
  C embed_tokens / lm_head                                2 本 2034.2 MB
  D norm 一式 (照合用)                                   135 本    0.4 MB
```

#### 規約の直接照合 — 2 件とも確定

| 照合 | 上流 bf16 | MLX-4bit | 差 |
| --- | ---: | ---: | --- |
| `input_layernorm` 平均 | +0.0311 | +1.0311 | **+1.000001** |
| `post_attention_layernorm` 平均 | +0.1516 | +1.1516 | **+1.000011** |
| `q_norm` 平均 | +0.3255 | +1.3253 | **+0.999846** |
| `linear_attn.norm` 平均 | +0.8842 | +0.8842 | **0.000000 (ビット一致)** |

**`Qwen3_5MoeRMSNorm` は MLX が `1+w` を焼き済み、`RMSNormGated` は素のまま。**
残差 3.91e-03 は `1+w` を BF16 に丸めた量そのもの。§4-2(c)#1 / #2 は**完全に決着**した。

`conv1d.weight` は上流 `(8192,1,4)` / MLX `(8192,4,1)` で、**squeeze 後の値はビット一致**。
軸だけが入れ替わっている。

#### ★ `in_proj_a` の 4-bit が減衰ゲートに与える害 (導出)

α = exp(g) は再帰状態の減衰係数で、`Δg = Δlog α` は T トークンで
`Πα` に `exp(T · mean Δg)` として**累積する**。したがって **`mean Δg` の偏りが本質**であり、
α が 0 近傍の領域の相対誤差は意味を持たない (どちらも「全部忘れる」)。
入力は RMSNorm 後を模した `x ~ N(0,1)^2048` を 8,192 本 — **合成入力なので方向と桁の指標であり、
実活性ではない**(実活性での再測は Phase 0)。

| 層 | \|dW\|/\|W\| | α>0.5 での相対誤差 中央値 / p95 / max | mean Δg (偏り) | **1000 tok の Πα 誤差** |
| ---: | ---: | ---: | ---: | ---: |
| L0 | 9.17% | 0.006% / 0.32% / 3.9% | −1.14e-03 | **×0.321** |
| L1 | 9.65% | 0.107% / 3.43% / 16.7% | −2.17e-04 | ×0.805 |
| L2 | 10.19% | 0.326% / 3.65% / 28.4% | −1.53e-04 | ×0.858 |
| L4 | 10.07% | 0.334% / 3.41% / 19.5% | −2.15e-04 | ×0.806 |
| L8 | 10.13% | 0.032% / 1.34% / 15.3% | −1.22e-05 | ×0.988 |
| L16 | 11.07% | 0.085% / 1.44% / 10.3% | −6.74e-05 | ×0.935 |
| L28 | 10.99% | 0.327% / 3.00% / 17.2% | −8.32e-05 | ×0.920 |

**偏りは全層で負である。**つまり 4-bit の `in_proj_a` は**系統的に「忘れっぽく」する。**
L0 では 1000 トークンで保持量が **0.32 倍**になる。
これらの層は実際に長期記憶を担っている (α 中央値: L0 0.997 / L2 0.865 / L8 0.990 / L28 0.917、
α>0.5 の割合は L28 で 96.8%) ので、**短いプロンプトでは見えず、長文脈で効く形の劣化**になる。

#### 8-bit g64 に打ち直した場合 (同じ指標、導出)

| 層 | \|dW\|/\|W\| | 中央値 / p95 / max | mean Δg | 1000 tok |
| ---: | ---: | ---: | ---: | ---: |
| L0 | **0.53%** | 0.000% / 0.019% / 0.23% | −2.77e-05 | **×0.973** |
| L2 | 0.59% | 0.019% / 0.213% / 1.29% | −8.25e-06 | ×0.992 |
| L8 | 0.59% | 0.002% / 0.078% / 0.82% | −4.82e-06 | ×0.995 |
| L28 | 0.63% | 0.019% / 0.173% / 0.79% | −2.75e-06 | ×0.997 |

**誤差が 17 倍改善し、累積の偏りが消える。**
`in_proj_a` / `in_proj_b` は 30 層 × 2 本 = **3.93M param しかない。
4-bit 2.21 MB → 8-bit 4.18 MB、差は +2 MB。**

**→ 決定: `in_proj_a` / `in_proj_b` は bf16 抽出から 8-bit g64 で打ち直す。**
費用 2 MB、効果は「長文脈で系統的に忘れっぽくなる」の除去。議論の余地が無い。

#### 参考: 他の 4-bit 箇所の重み誤差 (実測(手元))

| | \|dW\|/\|W\| | 8-bit g64 にした場合の増分 |
| --- | ---: | ---: |
| `shared_expert.gate_proj` L0 | 10.26% | 125.8M param → **+67 MB** |
| `shared_expert.down_proj` L20 | 9.89% | (同上に含む) |
| `lm_head` (先頭 4096 行) | 9.09% | 508.6M param → **+286 MB** |

shared expert は全トークンが通り、lm_head は logits に直結する (投機の受理率、§10)。
**どちらも bf16 抽出を持っているので、Phase 0 の結果を見てから 8-bit に上げられる。**
3 つ全部上げても install は 19.51 → 約 **19.87 GB**。

### 16-6. 上流 bf16 (71.9 GB) を全部引く必要はあるか — **無い**

| 用途 | 供給源 | 状態 |
| --- | --- | --- |
| Phase 0〜4 の本体 (テキスト) | `MLX-4bit` 19.51 GB | **取得済み** |
| 攻めすぎ箇所の打ち直し (in_proj_a/b, shared expert, embed/lm_head) | **bf16 部分抽出 2.30 GB** | **取得済み** |
| カーネル検証の参照 | `MLX-4bit` を脱量子化して fp32 で回す | 手元で足りる |
| Vision (Phase 9) | `oQ4e-mtp` の `vision_tower.*` (bf16 893 MB) | **取得済み (§17-1)** |
| MTP (Phase 7) | `oQ4e-mtp` の `language_model.mtp.*` (switch_mlp は 4-bit g64、503 MB) | **取得済み (§17-1)** |
| imatrix の効きの対照 | `oQ4e-mtp` | **取得済み (§17-1)** |
| **oQ / GPTQ を自分で回す (案「oQ を自分で回す」)** | **bf16 全体** | **これだけが 71.9 GB を要求する** |

**Phase 0〜9 は上流 bf16 の全体を必要としない。**残る唯一の用途は、18 GB の routed expert を
**自分で** imatrix + GPTQ で量子化し直す案「oQ を自分で回す」である。それは Phase 6 以降の選択肢であり、
やると決めてから引けばよい。

**次に引くべきは 上流 bf16 ではなく `scottlowry/…-oQ4e-mtp` (21.6 GB)** — vision と MTP の
唯一の供給源であり、かつ oQ を回さない限り再現できない。
**→ 2026-08-21 夜に取得した (§17-1)。**

> 転送速度の実測: 単一ストリームの range GET で **11.3 MB/s (約 90 Mbps)** だった。
> 500 Mbps は出ていない。`hf download --max-workers 8` の並列取得は
> これより速いはずなので、oQ4e の見積もりにこの数字を使わないこと。
> **実績: `hf download` (hf_transfer 有効) で oQ4e 21.6 GB が約 6 分だった (§17-1)。**

### 16-7. §15-7 への差し戻し

| 節 | 更新 |
| --- | --- |
| §15-3 の案の順位 | **`ornith-ai/…-MLX-4bit` を第一候補に繰り上げ。**打ち直し 0 本で、案「公開版を打ち直す」の 248 本作業が消える。oQ4e は「imatrix の効きを測る対照」と「vision / MTP の供給源」に役割変更 |
| §4-2(c)#1 / #2 | **不要になった** (MLX が焼き済み)。ただし `q_norm` への `1/16` の焼き込み (#3) は**依然として要る** |
| §11 Phase 1 | 入力はローカルの `~/LLM/Ornith-1.5-35B-A3B-MLX-4bit`。ダウンロード不要 |
| §11 Phase 0 | 最優先の測定を §16-4 の `in_proj_a`/`b` 感度に差し替え |
| §13 リスク #6 (`gate_up` 交互配置) | **消滅** |
| 新規リスク | **`in_proj_a`/`in_proj_b` の 4-bit** (§16-4)。中止線は「bf16 との相対誤差が線形注意層の出力で 1% を超えたら上流 bf16 から差し替える」 |

---

## 17. 現在地 (2026-08-21 夜、実測(手元))

**GPU はまだ 0 回。**この節の数字はすべて「ファイルを読む」「CPU で量子化する」までで出ている。

### 17-1. 手元にあるもの

| 置き場 | 中身 | 実効サイズ | 状態 |
| --- | --- | ---: | --- |
| `~/LLM/Ornith-1.5-35B-A3B-MLX-4bit` | 公式 MLX 4-bit。テキスト専用 (vision / MTP なし)、打ち直し 0 本 | 19.51 GB | §16 で検証済み |
| `~/LLM/Ornith-1.5-35B-A3B-oQ4e-mtp` | oQ4e。**MTP 42 本 + vision 333 本つき**、imatrix つき。非互換 248 本あり | 21.61 GB | **新規取得・検証済み** |
| `~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64` | 上を打ち直したもの。**非互換 0 本** | 21.86 GB | **新規作成** (§17-2) |
| `~/LLM/Ornith-1.5-35B-A3B-bf16-partial` | 上流 bf16 の部分抽出。**518 本 3.50 GB** (§16-5 の 451 本 2.30 GB に 67 本 1.195 GB を追加) | 3.50 GB | 拡張済み |

**oQ4e の完全性 (実測(手元)):** テンソル 2,052 本、`index.json` のキー集合と 5 シャードの
safetensors ヘッダが完全一致。テンソルデータ **21,613,054,816 B** は
**§15-2 が上流ヘッダから測った値と 1 バイトも違わない** (差の 270,763 B はヘッダぶん)。
`oq_imatrix_report.json` も同梱されている。

**取得速度:** `hf download` (hf_transfer 有効) で 21.6 GB が約 6 分。
§16-6 の脚注が心配した「単一ストリーム 11.3 MB/s」は当たらなかった。

### 17-2. 案「公開版を打ち直す」を実行した (§15-3)

`config.json` の per-tensor override から
**`group_size ≠ 64` または `bits ∉ {4,8}`** を機械的に拾うと **ちょうど 248 本**。
bf16 換算 **1,460,928,512 B = 730,464,256 param** で、**§15-2 の値と完全一致**した。

**入力は「5/6/8-bit を bf16 に戻したもの」ではなく、上流 bf16 の原本である。**
§16-5 で作った部分抽出に **181 本が既に入っていた**ので、
HTTP Range で新たに取ったのは残り **67 本 (1.195 GB)** だけ:
`linear_attn.out_proj` 30 / `in_proj_z` 29 / `in_proj_qkv` 4 / `self_attn.{q,k,v}_proj` 4。
**これで打ち直し対象 248/248 の bf16 原本が揃った。逆量子化の往復は 1 本も無い。**

打ち直しは `mx.quantize(w.astype(bfloat16), group_size=64, bits=8)`、CPU デバイス。

| | 予測 (§15-3) | 実際 |
| --- | ---: | ---: |
| 248 本のバイト | 533 MB → 776 MB (+243 MB) | **533,434,368 → 776,118,272 (+242,683,904)** |
| install 合計 | 約 21.86 GB | **21.86 GB** |
| メニュー外テンソル | 0 本 | **0 本** |

`config.json` の per-tensor override は 314 本すべて `8b/g64/affine`、base は `4b/g64/affine`。
`Quantization.supportedGroupSizes = [32,64]` と「routed expert は int4 のみ」の制約に対して
**赤リストが空になった。**

**打ち直しの効き目 (上流 bf16 に対する相対 L2 誤差、実測(手元)):**

| テンソル | oQ4e | 打ち直し後 |
| --- | ---: | ---: |
| `layers.0.linear_attn.out_proj` (5b/g128) | 0.06628 | **0.00765** |
| `layers.3.self_attn.q_proj` (5b/g128) | 0.05647 | **0.00743** |
| `layers.20.linear_attn.in_proj_z` (5b/g128) | 0.05939 | **0.00800** |
| `layers.0.mlp.shared_expert.down_proj` (8b/g128) | 0.00856 | **0.00781** |

5-bit だったものは 8 倍近く改善して 8-bit の床 (0.0075 前後) に張り付いた。
**§15-3 の「oQ が 5〜8 bit に昇格させた 248 本を 4-bit に落とすのは感度測定の結論を
捨てることに等しい、上げる側に倒す」という判断が、数字の側から支持された。**

**ディスクの実装:** シャード 1〜5 は `oQ4e-mtp` への hardlink。新規実体は
打ち直した 744 本 (248 × weight/scales/biases) が入った
`model-00006-of-00006.safetensors` (740.3 MiB) だけ。増えたディスクは 740 MiB で、
元リポジトリは無傷。シャード 1〜5 に残る旧 248 本の 508.7 MiB は index から参照されない。
`metadata.total_size` は上流の流儀 (シャードのファイルサイズ和) だと死にバイトを含んで
嘘になるので、**index が参照するテンソルデータのバイト和**に変えてある
(`total_size_convention` に明記)。経緯は同ディレクトリの `README.md`。

### 17-3. 候補が 2 本になった — §16-7 の順位は保留

§16-7 は「打ち直し 0 本」を理由に公式 MLX-4bit を第一候補に繰り上げたが、
**打ち直しが実際には 1 回の CPU 作業で終わったので、その理由は弱くなった。**
いま比べるべきは次の 2 本である:

| | **公式 MLX-4bit** (+ §16-5 の打ち直し) | **oQ4e-g64** (§17-2) |
| --- | --- | --- |
| 実効サイズ | 19.51 GB (打ち直し込みで約 19.87 GB) | **21.86 GB** |
| MTP | **無い。**oQ4e から移植が要る | **入っている** (42 本、`switch_mlp` 積み済み 503 MB) |
| vision | **無い。**oQ4e から移植が要る | **入っている** (333 本 bf16 893 MB) |
| routed expert の校正 | **素の RTN** | **imatrix つき** (`oqe_code_multilingual` 128×512) |
| `in_proj_a`/`b` | 4-bit → **§16-5 で 8-bit 打ち直しを決定済み** | **8-bit g64** (打ち直し後) |
| `embed`/`lm_head` | 4-bit (§16-4 の懸念) | **8-bit g64** |
| shared expert | 4-bit (§16-4 の懸念) | **8-bit g64** (打ち直し後) |
| RMSNorm の `+1` | **MLX が焼き済み** (§16-5) | **未照合。§17-4 の最初の一手** |
| 赤リスト | 0 本 | 0 本 |

**差は 2.35 GB で、その対価が「imatrix + MTP + vision + 高ビットの保護箇所」である。**
どちらを本線にするかは**まだ決めていない。**決める前に §17-4 の照合が要る。

> **注意 (§16-3 との食い違い):** 公式 MLX-4bit では `mlp.gate` (router) が **8-bit g64** で、
> §15-5 が oQ 版について書いた「router は BF16 なので int8 不要」は当てはまらなかった。
> **oQ4e-g64 側の router がどちらなのかは未確認。**§17-4 で見る。

### 17-4. 次の一手

**GPU 不要 (先にこれを全部やる):**

1. **`oQ4e-g64` の norm 規約を §16-5 と同じやり方で照合する。**
   `input_layernorm` / `post_attention_layernorm` / `q_norm` の平均を上流 bf16 と比べ、
   `1+w` が焼かれているか、`linear_attn.norm` は素のままかを確定する。
   **MLX 変換器が違えば結論も違う。公式版の結果を流用してはいけない**
2. **`conv1d.weight` の軸順を確認する** (公式版は `[8192,4,1]`、上流は `[8192,1,4]`)
3. **router のビット幅を確認する** (§17-3 の注意)
4. **`q_norm` への `1/16` の焼き込み** (§4-2(c)#3)。どちらの候補でも要る
5. **名前寄せ** (§4-2(d)) — `.mlp.switch_mlp.` を `routedExpertRole` に当てる 1 文字列
6. `.gturbo` への repack を通す

**GPU が要る (ここから先は指示待ち):**

7. **生成スモーク。`oQ4e-g64` はまだ 1 度も推論を通していない。**
   相対 L2 誤差を測っただけで、文が出るかは未確認
8. §16-5 の `in_proj_a` 感度を**合成入力ではなく実活性**で測り直す (Phase 0 の宿題)
9. 2 候補の品質差を測る。**ここで初めて「imatrix に 2.35 GB の価値があるか」が決まる**

### 17-5. まだ手を付けていない前提

- **tokenizer は確実に弾かれる** (§16-3)。`decoder.type = ByteLevel` が
  `verifyDecoderConfiguration` の要求と合わない。Phase 5 の作業として据え置き
- Gated DeltaNet カーネル (§5-6 / §15-4) は 1 行も書いていない
- 運用点 (スロット数・チャンク幅) は Gemma 4 の値のままで、Ornith 用には何も測っていない

---

## 18. 参照

- 上流 (bf16): [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
- 上流 (MLX 4-bit, oQ): [`scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp`](https://huggingface.co/scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp)
- 量子化ツール: [`jundot/omlx`](https://github.com/jundot/omlx) (Apache-2.0) —
  `docs/oQ_Quantization.md` / `omlx/oq.py` / `omlx/custom_kernels/qwen35_prefill/gdn.py`
- `transformers` `models/qwen3_5_moe/modeling_qwen3_5_moe.py`
- 本リポジトリ: [PLAN.md](PLAN.md) / [PLAN_QAT.md](PLAN_QAT.md) / [PLAN_VISION.md](PLAN_VISION.md) /
  [docs/mtp/README.md](docs/mtp/README.md) / [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) /
  [docs/serving/SPEC.md](docs/serving/SPEC.md)
