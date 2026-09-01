# 20. Phase 3 の結線 — decode が参照と一致した (実測(手元)、2026-08-21)

[04-PHASES.md](04-PHASES.md) 次の一手 **#14**。Phase 2 で書けたカーネルを
実物の重みにつなぎ、**固定プロンプトから greedy でトークンを出して CPU float32
参照と突き合わせた**。GPU を使い、**この計画で初めてモデルを載せて動かした**。

```
swift run -c release TsugumiKernelCheck \
  --qwen-decode scratch/ornith-oq4e-g64.moepack
```

| | |
| --- | --- |
| 結果 | **41 トークンすべてが参照と一致** (§5)。参照は [14 §6](14-REFERENCE.md) の生成スモーク |
| 決めたこと | **`QwenForwardRunner` を新設し、直列に書く** (§1)。`RealForwardRunner` は 1 行も触っていない |
| 実物で分かったこと | **3 か所が「Phase 2 のカーネルのままでは静かに壊れる」形だった** (§2) |
| 検査 | `--qwen-decode` の**正例 1 + 負例 5**。パッケージテストに 9 本 (`RecurrentStateTests`) |
| 副産物 | `RecurrentStateManager`、`LayerKind.linear`、`qwen_embed_lookup_int8`、`qwen_moe_shared_gate_logit`、`qwen_residual_add` |

---

## 1. `QwenForwardRunner` — 直列に書いた

[03 §3-2](03-DESIGN.md) の方針どおり `RealForwardRunner` (3,319 行) には触らず、
別のランナーを立てた。**共有したのは輪の下にあるもの全部** — `Model`、
ストリーマー、router と routed expert のカーネル、`Attention`、`RMSNorm`、
逆量子化 GEMV。輪そのものだけが新しい。

**意図的に直列にしてある。**1 層 = コマンドバッファ 2 本 (router まで / MoE の尾)、
それぞれ待ってから次を積む。Phase 3 の出口条件は「参照と一致」であって速さでは
なく、**食い違ったときに層と段を名指しできる形**が要る。`RealForwardRunner` の
3 本パイプライン (層の routed I/O と次の層の attention を重ねる) は、それを
測った数字ごと持っているので、写すのは結線が済んでからでよい。

1 層はこの順に流れる (参照器 `forward` と同じ):

```
normed = rmsnorm(hidden, input_layernorm)
hidden += 再帰(normed)  または  attention(normed)
normed = rmsnorm(hidden, post_attention_layernorm)
hidden += MoE(normed)
```

Gemma の sandwich norm 4 本も `layer_scalar` も logit softcap も無い。
だから `fused_post_attn_setup` / `fused_layer_tail` に**対応物が無い**。
残差は素の足し算 2 本で、`qwen_residual_add` を足した (2048 要素 × 2 dispatch / 層)。
**足した直後は必ず RMSNorm なので 1 本に畳める**が、畳むのは Phase 6 に回す。

### 1-1. 状態の置き場所 — `RecurrentStateManager`

[03 §3-3](03-DESIGN.md) のとおり新設した。再帰層 30 本ぶんを 2 本の
`MTLBuffer` に連結し、層ごとにオフセットで渡す。

| テンソル | 形 | 1 層 | 30 層 |
| --- | --- | ---: | ---: |
| `S` (再帰状態) | FP32 `[Hv=32, Dv=128, Dk=128]` | 2 MiB | 60 MiB |
| `conv` (因果窓の過去) | FP16 `[K-1=3, C=8192]` | 48 KiB | 1.41 MiB |

**文脈長に依らない。**これが [01 §3-4](01-MODEL.md) の「線形層の状態は 62.8 MiB 固定」
の実体である。`reset()` は本当にゼロで埋める (`KVCacheManager.reset` の
`MADV_DONTNEED` とは違う。あちらはゼロにしたカーソルの下だけが読まれるが、
こちらは次のトークンで丸ごと読まれる)。

### 1-2. `KVCacheManager` の 3 値目

`LayerKind` に `.linear` を足し、**その層には 1 バイトも確保しない**
(Metal が長さ 0 のバッファを作らないので 1 バイトずつ置き、読む道は全部
`.linear` で先に落とす)。`ExpertCacheBudget` も同じ規則で数え直し、
`recurrentStateBytes` を足した。maxContext 4096 で 40 層ぶん確保すると
**K/V を読まない 30 層に 252 MB** を積むことになり、18 GB 機では
「触りもしないバイト」で構成を断る側に回る。

`maximumSafeRewind` は再帰層が 1 本でもあれば **0** ([03 §3-4](03-DESIGN.md))。
リング容量から出した数は「K/V の行がまだ在る」ことしか言っておらず、
再帰状態には引き算が無い。

---

## 2. 実物で分かった 3 つの食い違い

Phase 2 は合成入力でカーネルを検査した ([17](17-PHASE2-KERNELS.md))。
**実物をつないで初めて見えたのは「カーネルは正しいが、そのカーネルに
実物を渡せない」型の食い違い**で、3 か所あった。3 つとも
**落ちずに走り、それらしいトークンを出す**側の間違いである。

| # | 場所 | 何が違ったか | どうしたか |
| --- | --- | --- | --- |
| 1 | 埋め込み | 本線の `embed_tokens` は **8-bit** ([18 §3](18-MIXED-BITS.md))。`embed_lookup_int4` は nibble の展開が行の幾何そのもので、幅は引数ではない | `qwen_embed_lookup_int8` を書いた。**`out_scale` は持たせない** — Gemma だけの `sqrt(hidden)` を 1.0 で渡す形は、渡し忘れが静かに動く |
| 2 | shared expert のゲート | `shared_expert_gate.weight` も **8-bit**。`qwen_moe_shared_gate` は BF16 を直に読む ([17](17-PHASE2-KERNELS.md) の検査は合成の BF16 で通っていた) | 内積を汎用 GEMV (M=1) に出し、`qwen_moe_shared_gate_logit` は sigmoid と掛け算だけにした。融合版は BF16 の checkpoint 用に残す。**どちらを呼ぶかは索引が言う幅で決める** ([18 §1](18-MIXED-BITS.md) の「索引から導く」がそのまま効いた) |
| 3 | routed expert の活性化 | `moe_phase1_gate_up_act_u16load` は `gelu_pytorch_tanh` を焼いていた。Qwen は SiLU | **関数定数 `FC_MOE_ACT_SILU` (index 4)** で分岐。**Gemma は定数を定義しないので、その PSO は今までと同じコードを吐く**。`.metal` を触ったが、Gemma の腕は 1 命令も動いていない |

3 つとも [03 §4-1](03-DESIGN.md) の「そのまま使える」表に載っていた資産である。
表が間違っていたわけではなく、**「そのまま使える」の粒度が資産単位で、
テンソルの幅は資産の外にあった**。

---

## 3. `q_proj` の 2 倍幅と attention の間

`qwen_qkv_epilogue` は 2 倍幅 `[NQ, 2*HD]` を **その場で**正規化・回転させ、
後半のゲートには触らない ([03 §2-2](03-DESIGN.md))。一方 attention の
カーネルは q を `headDim` 刻みの連続として読む。**間に詰め直しが要る。**

**blit にした。**decode は 1 トークンなので、16 ヘッド × 512 B のコピーが
16 本。エピローグに「詰めた q」用の第 2 出力を足す道もあったが、そうすると
出力ゲートを持ち回るのが**忘れやすい方**になる (ゲートは attention の後で
`qwen_attn_output_gate` が同じ 2 倍幅バッファから読む)。
prefill (Phase 4) では T 倍になるので、そこで測り直す。

---

## 4. router は単位ベクトルで Gemma のカーネルに乗る

[03 §4-1](03-DESIGN.md) の見立てどおりだった。Gemma の
`router_gemv_gemma4_bf16_r4` は入力に `effective_scale[i]`
(= `router.scale` × `1/sqrt(D)`)、`router_topk_select_k8` は top-k の重みに
`per_expert_scale[e]` を掛ける。**Qwen にはどちらも無い**ので、
両方 BF16 の 1.0 を並べたバッファを渡す。それで残るのは
`logits = W·x` → top-8 → softmax で、参照器の

```python
probs = softmax(logits); idx = top_k(probs); weights = vals / sum(vals)
```

と**恒等的に等しい** (全体 softmax の正規化定数が比で消える)。
カーネルを 1 本も足していない。

---

## 5. 検査 — 一致 41 本と負例 5 本

### 5-1. 正例

参照は [14 §6](14-REFERENCE.md) の生成スモーク: 上流トークナイザで包んだ
19 トークンのプロンプト (`日本の首都はどこですか。一文で答えてください。`) に対する
float32・greedy の 41 トークン。**41 本すべて一致**した。

**比べるのはトークン ID で、活性ではない。**理由は 2 つ。参照は 47.5 s/トークン
なので層ごとの hidden を全ステップぶん落とす fixtures は作るのに何時間もかかり、
プロンプトを変えるたびに作り直しになる。そして FP16 の活性が float32 参照と
ビット一致することはそもそも無いのに対し、**248,077 行の argmax は一致するか
しないかのどちらか**で、出口条件が問うているのはそちらである。

### 5-2. 負例 5 本 — 物差しが働いていることの証明

正例が通っただけでは、**この比較が間違いを見つけられることの証拠にならない**
(PLAN_VISION.md §6-3、[17 §2](17-PHASE2-KERNELS.md) と同じ作法)。
カーネル検査と違って**誤差の床が無い**ので、校正の手段は「**このランナーが
実際に取りかけた道**を通した模型を、同じ比較が落とすこと」しかない。
`QwenForwardRunner.DecodeFault` の 5 本は全部、書いている途中に本当に
在り得た形である。5 本とも**確保するバイトも走る時間も正しく、
それらしいトークンを出す。**

| 負例 | 何を間違える | 離れた位置 |
| --- | --- | ---: |
| `shared-gate-bf16` | 8-bit のゲートを BF16 のカーネルで読む (§2 の #2 を直さなかった場合) | step 0 |
| `shared-gate-skipped` | shared expert の sigmoid ゲートを落とす | step 0 |
| `routed-gelu` | routed expert の活性化を Gemma の gelu のままにする (§2 の #3) | **step 8** |
| `uncompacted-query` | 詰め直さない q をそのまま attention に渡す (§3) | step 7 |
| `forget-state` | 再帰状態をトークンごとにゼロにする (30 層が直前の 1 トークンしか見なくなる) | step 0 |

**`routed-gelu` は 8 トークン生き延びた。**これはこの検査自身の分解能に
ついての事実で、既定の負例の予算を **16 トークン**にした理由でもある
(`--qwen-decode-fault-tokens` で広げられる)。gelu と silu は形が近く、
routed 側は 2 本ある枝の片方なので、**トークン単位の比較で見えるまでに
数トークンかかる**。8 本しか回さない読み方をしていたら、
「関数定数が効いていない」と「効いているが差が出ていない」を取り違えていた。

`shared-gate-bf16` が出したトークンは `-1` = `0xFFFFFFFF`、
すなわち `qwen_lm_head_greedy_int8_rows_reduce` の番兵である。
8-bit のバイト列を BF16 として読むと値が桁違いになり、logits が有限でなくなる。
**離れることの証拠としては十分だが、「もっともらしく間違う」側の例ではない。**

### 5-3. パッケージテスト

`RecurrentStateTests` を 9 本足した (状態の大きさが文脈長に依らないこと、
オフセットが重ならず密であること、`reset` が両方をゼロにすること、
`layerCount` とマスクの食い違いを断ること、`convKernelDim == 1` を断ること、
再帰層に K/V を確保しないこと、`maximumSafeRewind` が 0 になること、
**Gemma の `maximumSafeRewind` が変わっていないこと**、予算が K/V の代わりに
状態を数えること)。`swift test --no-parallel` で **1,282 件すべて緑**
(直前は 1,273 — [18 §7](18-MIXED-BITS.md) と同じく、並列だと remote install 系が落ちる)。

既存のカーネル検査も動かしていない: `--qwen` 39 本、`--gdn` 15 本、
既定の 69 本 (`moe.metal` に手を入れたので、gelu の腕はここが見張っている)。

---

## 6. 数字 (**運用値ではない**)

n=1。直列で、prefill 経路が無く、何も重ねていない状態のもの。
**Phase 6 で測り直すまで、ここから何かを読み取らないこと。**

| 見るもの | 値 |
| --- | ---: |
| `Model.load` | 1,259 ms |
| step 0 (19 トークンのプロンプトを 1 トークンずつ + エキスパートキャッシュが冷たい) | 9,845 ms |
| step 1〜40 の合計 | 3,788 ms |
| 同、1 トークンあたり | 94.7 ms |

---

## 7. この Phase が動かした結論

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) Phase 3 | **41 トークンで一致。**出口条件の 64 トークンは参照の続きを取ってから (§8) |
| [04](04-PHASES.md) 次の一手 #14 | **完了** |
| [03 §3-2](03-DESIGN.md) 「forward runner は分ける」 | **そのとおりにした。**`RealForwardRunner` は無変更 |
| [03 §3-3](03-DESIGN.md) `RecurrentStateManager` | **入った** (§1-1) |
| [03 §4-1](03-DESIGN.md) 「そのまま使える」表 | **3 行に但し書きが要る** (§2)。資産は使えるが、実物のテンソルの幅がその手前で効く |
| [03 §2-9](03-DESIGN.md) 「要らないもの」 | logit softcap と `sqrt(hidden)` は本当に要らなかった。この経路はどちらも持っていない |
| [05 §1](05-RISKS.md) の運用点 | **まだ何も測っていない。**既定のまま |

---

## 8. 残した宿題

- **64 トークンの参照。**[14 §6](14-REFERENCE.md) の生成は外から止めたので
  41 本しかない。同じプロンプトで `--max-new 64` を回し直し、
  `--qwen-decode-fixture` に渡せば Phase 3 の出口条件の文言どおりになる
- **prefill 経路が無い。**プロンプトも 1 トークンずつ流している。Phase 4
- **`prefill.metal` の MoE も gelu を焼いている。**decode 側は関数定数で分けたが、
  prefill 側は手つかず (Phase 4 で同じことをする)
- **speculative / server は触っていない。**[03 §3-4](03-DESIGN.md) の
  「巻き戻せない」の対策 (スナップショット) は入っていない。
  `maximumSafeRewind` が 0 を返すので**黙って壊れることは無い**
- **トークナイザは依然弾かれる** ([10 §3](10-MLX4BIT-AUDIT.md))。この検査が
  トークン ID を直に受け取るのはそのためでもある。Phase 5
