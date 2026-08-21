# 01. 相手の正体と数量

上流の事実は 2026-08-21 に HF API / HTTP range / 参照ソース取得で直接読んだもの
(**実測(上流)**)。`config.json` 全文、`model.safetensors.index.json` の 1,811 本
(16 シャード、`total_size = 71,903,645,408 B`)、シャード 1 / 2 / 16 の safetensors
ヘッダ (全テンソルの shape と dtype を確定、全部 BF16)、`tokenizer_config.json`、
`chat_template.jinja`、`generation_config.json`、`preprocessor_config.json`、
`transformers` の `modeling_qwen3_5_moe.py` (97 KB)。

---

## 1. text_config

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

## 2. テンソルの実体 (shape はすべて実測(上流)、dtype は全部 BF16)

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

**注意すべき非対称:** 本体の routed expert は**融合 3D テンソル 2 本** (`gate_up_proj` /
`down_proj`) なのに、**MTP 側はエキスパートごとにバラの 768 本**である (**実測(上流)**)。
(MLX 変換済みチェックポイントでは両方とも gate/up/down の 3 ロールに分割・積層済み — [02 §5](02-CHECKPOINTS.md)。)

## 3. 算式 (`modeling_qwen3_5_moe.py` を読んで確定、実測(上流))

### 3-1. RMSNorm — Gemma と同じ `1 + w` 規約

```python
output = x * rsqrt(mean(x^2) + eps)
output = output * (1.0 + self.weight.float())     # Qwen3_5MoeRMSNorm
```

つまり `input_layernorm` / `post_attention_layernorm` / `q_norm` / `k_norm` / `norm` /
`mtp.*_norm` は `1+w` を焼き込めば既存の `rmsnorm_bf16w` がそのまま使える
(公式 MLX-4bit は**変換側が焼き済み** — [10 §4](10-MLX4BIT-AUDIT.md)。こちらで足してはいけない)。
**ただし `linear_attn.norm` だけは別物** (`Qwen3_5MoeRMSNormGated`) で `+1` しない:

```python
h = h * rsqrt(mean(h^2)+eps);  h = self.weight * h;  h = h * silu(gate.float())
```

**+1 する norm と しない norm を取り違えると静かに壊れる。**

### 3-2. full attention

```python
q_all = q_proj(x).view(..., -1, head_dim*2)          # [.., 16, 512]
q, gate = chunk(q_all, 2, dim=-1)                     # 各 [.., 16, 256]
q = q_norm(q);  k = k_norm(k_proj(x));  v = v_proj(x)
q, k = rope(q, k)                                     # partial 0.25 (下記)
o = attention(q, k, v, scale = head_dim**-0.5)        # GQA 16/2, causal
o = o * sigmoid(gate)                                 # ← attn_output_gate
o = o_proj(o)
```

**partial RoPE (README #10 の中身):**

```python
dim      = head_dim * 0.25 = 64
inv_freq = 1 / (theta ** (arange(0, 64, 2) / 64))     # 32 本、分母は 64
q_rot, q_pass = q[..., :64], q[..., 64:]
q_out = cat([q_rot*cos + rotate_half(q_rot)*sin, q_pass])
# rotate_half は 64 次元の中で (i, 32+i) を組む
```

本ランタイムの `fused_rope_neox_pair` は `(pair, HD/2 + pair)` を組み `pow(theta, -2*pair/HD)`
を使う。**組も分母も違う。**新カーネルが要る ([03 §2-2](03-DESIGN.md))。

mRoPE は `mrope_section=[11,11,10]` を t/h/w に交互配置するが、**テキストのみなら
3 軸の position_ids が全部同じ**になるので `apply_interleaved_mrope` は恒等写像に潰れる。
→ **Phase 1〜8 は mRoPE を一切実装しなくてよい。Vision ([04 §11](04-PHASES.md)) でだけ要る。**

### 3-3. router (`Qwen3_5MoeTopKRouter`)

```python
probs = softmax(x @ W.T, dim=-1)      # 256 全体で softmax
v, idx = topk(probs, 8)
v = v / v.sum(-1, keepdim=True)       # 選ばれた 8 本で再正規化
```

本ランタイムの `router_topk_select_k8` は「top-8 に限った softmax」を計算する。
**数学的に同一** (全体 softmax → top-k → 再正規化 は分母が消える)。**そのまま使える。**
Gemma 由来の `per_expert_scale` は Qwen に対応物が無いので **1.0 を書く**。

### 3-4. MoE ブロック

```python
shared = shared_expert(x)                             # silu-gated, 幅 512
shared = sigmoid(shared_expert_gate(x)) * shared      # ← Gemma に無い
routed = Σ_k w_k * down_k(silu(gate_k(x)) * up_k(x))
y = routed + shared
```

### 3-5. decoder layer — Gemma よりずっと単純

```python
h = h + mixer(input_layernorm(h))       # mixer = linear_attn か self_attn
h = h + moe(post_attention_layernorm(h))
```

sandwich norm も `layer_scalar` も無い。**norm は層あたり 2 本だけ** (Gemma は 6 本 + scalar)。

### 3-6. Gated DeltaNet (30 層ぶんの本体)

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

## 4. Gemma 4 26B-A4B との差分 (一覧)

| 項目 | Gemma 4 26B-A4B | Ornith / Qwen3.5-MoE | 影響 |
| --- | --- | --- | --- |
| 層構成 | 30 層、SWA×25 + full×5 | **40 層、linear×30 + full×10** | **最大。層種別が 2 値 → 3 値** |
| hidden | 2816 | 2048 | 専用化 PSO が外れる |
| head 数 | 16 / KV 8 (SWA) / KV 2 (full) | 16 / KV 2 (full のみ) | GQA 比 8。SWA 用 GQA カーネルは使わない |
| head_dim | 256 (SWA) / 512 (full) | **256 (full)** | prefill の `_d256` がそのまま当たる |
| attention 出力ゲート | 無し | **有り** (`sigmoid(gate) *`) | 小カーネル 1 個 |
| K == V | full 層で `v_proj = k_proj` | **常に別** | `isFull ? k : vProj` の分岐を消す |
| RoPE | 2 θ (1e4 / 1e6)、SWA は全回転 / full は partial 0.25、ペアは `(i, HD/2+i)` | **1 θ (1e7)、full のみ partial 0.25、ペアは `(i, 32+i)`、分母 64** | **新カーネル ([03 §2-2](03-DESIGN.md))** |
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
| チャット形式 | `<\|turn\|>` + `<\|channel\|>thought` | **ChatML + `<think>`/`<tool_call>` XML** | テンプレートとパーサを別系統で |
| ドラフター | 別リポジトリの assistant 4 層 | **本体同梱の MTP 1 層** | 取得が要らない ([03 §6](03-DESIGN.md)) |
| 上流の量子化 | mlx-community が 4-bit 版を配っている | 上流本体は bf16 のみだが、**公式 MLX-4bit と oQ4e が存在** | 量子化器は不要になった ([02 §3](02-CHECKPOINTS.md)) |

---

## 5. 数量 (導出。§5-2 / §5-3 はのちに実物とバイト一致し実測に格上げ)

### 5-1. パラメータ数

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

### 5-2. エキスパート 1 個 = 108 ページちょうど

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

affine の列は oQ4e のヘッダ ([02 §5](02-CHECKPOINTS.md))・公式 MLX-4bit の実ファイル
([10 §2](10-MLX4BIT-AUDIT.md)) と**バイト一致**しており、**実測(上流) / 実測(手元) に格上げ済み。**

### 5-3. インストール容量と常駐

実際の候補の実効サイズ: **公式 MLX-4bit 19.51 GB / oQ4e-g64 21.86 GB**
([02 §1](02-CHECKPOINTS.md))。以下は区画ごとの内訳 (int4 一律の導出):

| 区画 | 量子化 | バイト |
| --- | --- | ---: |
| routed experts | affine int4 g64 | 18.12 GB |
| embed + lm_head | int4 | 572 MB |
| 注意系 (q/k/v/o, in_proj_qkv/z, out_proj) | int4 | 720 MB |
| shared expert | int4 | 71 MB |
| router (`mlp.gate`) 40 本 | int8 | 22 MB |
| 小テンソル (A_log, dt_bias, conv1d, in_proj_a/b, 全 norm) | fp16/fp32 | 約 11 MB |
| **テキストのみ 合計** | | **約 19.5 GB** |
| MTP (うち expert 453 MB) | int4 | 475〜503 MB |
| Vision tower | bf16 | 892 MB |
| **全部入り** | | **約 20.9〜21.9 GB** |

Gemma 4 の 14.3 GB より **+5〜7 GB**。ディスク要件は上がる。
`model_weights.bin` (常駐) は **約 1.39〜1.40 GB**。Gemma の 1.35 GB とほぼ同じ桁だが、
**tie しないぶん lm_head の 286 MB が丸ごと上乗せ**されている。

### 5-4. 文脈と状態

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

### 5-5. decode 1 トークンのバイト予算 (全ヒット時、導出)

| 区画 | Gemma 4 | Ornith |
| --- | ---: | ---: |
| 常駐コア (lm_head 込み、embed は 1 行) | 1.35 GB | 1.11 GB |
| routed experts (top-8 × 層数) | 0.765 GB (240 本) | 0.566 GB (**320 本**) |
| KV 読み (4K 文脈) | 0.30 GB | 0.084 GB |
| 再帰状態 往復 | — | 0.126 GB |
| **合計** | **2.41 GB** | **1.89 GB (0.78×)** |

M3 Pro の帯域を 150 GB/s とすると、**バイトだけの天井は Gemma 62 tok/s に対し Ornith 79 tok/s**
(**導出**、効率 100% の非現実的な上限)。**decode は Gemma より速くなり得る、というのが
本計画の性能側の主張である。**ただし読み出しの**本数**は 240 → 320 に増える
([05 §1-3](05-RISKS.md))。

### 5-6. prefill の計算量 (2048 トークン、導出)

| | Gemma 4 | Ornith |
| --- | ---: | ---: |
| MoE (routed + shared) | 8.03 TFLOP | **4.63 TFLOP (0.58×)** |
| 注意系 | 1.20 TFLOP | **0.54 TFLOP (0.45×)** |
| routed expert I/O の最悪値 | 12.01 GiB | **16.88 GiB (1.41×)** |

**計算は軽くなり I/O は重くなる。**prefill は Gemma 以上に I/O 律速に寄る
([05 §1-2](05-RISKS.md))。
