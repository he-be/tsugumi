# 14. float32 参照器 (実測(手元)、2026-08-21)

[04-PHASES.md](04-PHASES.md) Phase 0 の 1 番目 (`dump_reference.py`) と、
「次の一手」7〜9 の**前提**にあたる道具。**GPU は使っていない** (numpy だけ)。

計画では「CPU / float32 の `transformers` で 8 トークン流す」としていたが、
そのままでは**動かない**。理由は 2 つ:

1. **上流 bf16 が手元に無い。**あるのは 518 本 3.50 GB の抜き出し
   ([02 §2](02-CHECKPOINTS.md)) で、全 70 GB は落としていない
2. **RAM が 18 GB しかない。**4-bit 版 (19.5〜21.9 GB) ですら、
   `mlx-lm` のように「全部載せてから回す」実装では swap 前提の数字しか出ない

そこで **1 回の forward に要るものだけを開いて捨てる参照器**を書いた。

---

## 1. 何を作ったか

| ファイル | 役割 |
| --- | --- |
| `Scripts/qwen35/mlx_quant.py` | MLX affine の逆量子化 (bits 4/8、group 32/64、テンソルごとの上書き) |
| `Scripts/qwen35/reference_forward.py` | float32 の参照実装。層ストリーミング、fixtures 書き出し、greedy 生成 |
| `Scripts/qwen35/test_reference_forward.py` | 参照器を**上流実装そのもの**と突き合わせる検証 |

**なぜ層ストリーミングで足りるか** — MoE はここが有利で、1 回の forward が触る量は
全体よりずっと小さい:

| 区画 | 全体 | 1 回の forward で触る量 |
| --- | ---: | --- |
| routed experts | 18.1 GB (453 MB × 40 層) | **top-8 だけ。**11 トークンなら 1 層あたり最大 88 個 = 149 MB |
| core (attention / norm / router) | 1.26 GB | 層ごとに開いて捨てる |
| embed / lm_head | 1.08 GB | 行だけ / 語彙を 16,384 ずつ区切って 1 回 |

## 2. 逆量子化の検証 — 上流 bf16 に対して

`mlx_quant.py` の詰め物の解き方 (**低いビットが先**) を、上流 bf16 の抜き出し
([10 §4](10-MLX4BIT-AUDIT.md) の C / B / A 群) に対して確かめた。**実測(手元)**:

| チェックポイント | テンソル | bits | 相対 L2 | affine の理論値 |
| --- | --- | ---: | ---: | --- |
| oQ4e-g64 | `layers.0.mlp.shared_expert.gate_proj` | 8 | 0.00803 | 約 0.007 |
| oQ4e-g64 | `layers.0.linear_attn.in_proj_a` | 8 | 0.00698 | 〃 |
| oQ4e-g64 | `layers.3.self_attn.q_proj` | 8 | 0.00743 | 〃 |
| 公式 MLX-4bit | `layers.0.mlp.shared_expert.gate_proj` | 4 | 0.10263 | 約 0.10 |
| 公式 MLX-4bit | `layers.3.self_attn.q_proj` | 4 | 0.09559 | 〃 |
| 公式 MLX-4bit | `embed_tokens` (先頭 4096 行) | 4 | 0.09779 | 〃 |

理論値は `range/(2^bits-1)/sqrt(12)` を `range ≈ 6σ` で見たもの。
**詰め方を取り違えていれば相対 L2 は 1.0 付近に出る**ので、これで両ビット幅とも決着した。

**この表自体が [02](02-CHECKPOINTS.md) の 2 候補比較の材料でもある** —
同じテンソルで oQ4e-g64 (8-bit) が 0.007、公式版 (4-bit RTN) が 0.096。
ただし**これは重みの再現誤差であって出力の品質ではない。**品質差は Phase 6。

## 3. 算式の検証 — 上流実装に対して

`test_reference_forward.py` は、小さい乱数モデル (hidden 64 / 4 層 / うち 1 層 full /
experts 8 / top-2 / head_dim 32 / rotary 8) を `transformers` の
`Qwen3_5MoeForCausalLM` (CPU / float32 / eager) と参照器の両方に流す。**実測(手元)**:

| 見るもの | 最大 \|差\| | 相対 |
| --- | ---: | ---: |
| logits (12 トークン一括) | 9.98e-07 | **6.44e-07** |
| top-1 一致率 | — | **100%** |
| 状態の持ち越し (prefill 8 + 1 トークン × 4 対 一括) | 9.69e-07 | **6.25e-07** |

float32 の丸めの範囲。これで次の 7 点が**同時に**確定した:

1. Gated DeltaNet の再帰 (減衰 → `kv_mem` → `delta` → 出力の順)
2. `conv1d` の軸順 `[8192, 4, 1]` と因果パディング
3. `l2norm(q)/√128` と `l2norm(k)` の非対称な倍率
4. attention の**出力ゲート**を**ヘッドごとに**割ること
5. partial RoPE の組 `(i, 32+i)` と分母 `rotary_dim=64`
6. router の「全体 softmax → top-8 → 再正規化」
7. **`Qwen3_5MoeRMSNorm` は `1+w`、`RMSNormGated` は素**という norm 規約の使い分け

3 本目の持ち越し検査が Phase 3 に効く: **conv 状態・再帰状態・KV の 3 つを
繰り越しても一括と一致する**ことが、実装側で保証された。

### 3-1. 3 つのソースが一致していること

算式の出典は 3 つあり、**全部一致した**。docs は `transformers` から書き起こしたもので、
mlx-lm (Apple の独立移植) は 3 番目の証人にあたる:

| 論点 | docs ([01 §3](01-MODEL.md)) | `transformers` 5.6.2 | mlx-lm `qwen3_5.py` |
| --- | --- | --- | --- |
| GDN の q/k 正規化 | `k=l2norm(k)`, `q=l2norm(q)/√128` | `l2norm` 後に `q *= Dk**-0.5` | `k=Dk^-0.5·rms_norm(k)`, `q=Dk^-1·rms_norm(q)` |
| 減衰 | `exp(-exp(A_log)·softplus(a+dt_bias))` | 同じ | `compute_g` が同式 |
| 出力ゲート | `chunk(q_all, 2, -1)` をヘッドごと | 同じ | 同じ |
| partial RoPE | 64 次元、`(i,32+i)`、分母 64 | `rotate_half` が rotary 内で半割り | `initialize_rope(64, traditional=False)` |
| router | 全体 softmax → top-8 → 再正規化 | 同じ | 同じ |

mlx-lm の `rms_norm` は `√D·l2norm` なので、倍率を約すと transformers と同じ式になる。
`eps` の入り方だけが違う (`sum(x²)+eps` 対 `sum(x²)+D·eps`) が、`eps=1e-6` では効かない。

**上流実装は取得しただけで、リポジトリには持ち込んでいない** (Apache-2.0 / 参照のみ)。

## 4. 実物での 1 回の forward — **oQ4e-g64 が初めて動いた**

```
scratch/vision-venv/bin/python Scripts/qwen35/reference_forward.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64 --text "日本の首都は" \
    --dump scratch/qwen35/fixtures-oq4e-g64.npz
```

**実測(手元)** (M3 Pro / 18 GB、CPU のみ、11 トークンの ChatML 1 ターン):

| 見るもの | 値 |
| --- | ---: |
| forward | **102.7 s** |
| float32 に展開したバイト | 26.42 GB (ファイルは 21.86 GB) |
| 常駐した状態 (線形 30 層 + KV 10 層) | **66.3 MB** |
| ピーク RSS | 約 3 GB (`--cache-core` なし) |

**`<|im_start|>assistant\n` の直後の top-5:**

| 順位 | トークン | logit |
| ---: | ---: | ---: |
| 1 | **248068 = `<think>`** | **18.716** |
| 2 | 13215 | 9.133 |
| 3 | 1628 | 8.767 |
| 4 | 1715 | 8.728 |
| 5 | 11373 | 8.583 |

`<think>` が 2 位に **9.6 の差**をつけている。思考ブロックを開くのは
このモデルのテンプレートどおりの振る舞いで、**チェックポイントが壊れていない**
最初の証拠になる ([11](11-OQ4E-G64-REBUILD.md) の打ち直し 248 本を含めて)。

状態 66.3 MB の内訳は、線形 30 層が `32 × 128 × 128 × 4 B × 30 = 62.9 MB` で
**文脈長に依らず固定** ([01 §3-4](01-MODEL.md) の導出と一致)。残りが KV 10 層 × 11 トークン。

## 5. fixtures

`--dump` は 194 本 / 81 MB の `.npz` を出す:

| 名前 | 形 | 何本 |
| --- | --- | ---: |
| `embed` / `final_hidden` | `[T, 2048]` | 2 |
| `layerN.input` | `[T, 2048]` | 40 |
| `layerN.gdn_state` | `[32, 128, 128]` | 30 |
| `layerN.gdn_out` | `[T, 32, 128]` | 30 |
| `layerN.attn_out` | `[T, 16, 256]` | 10 |
| `layerN.router_idx` / `router_w` | `[T, 8]` | 80 |
| `logits` / `tokens` | `[T, 248320]` / `[T]` | 2 |

**Phase 0 の出口はまだ閉じない。**残り 2 つ:

1. **81 MB はリポジトリに入らない。**Phase 2 が要る層だけに絞る (どの層を選ぶかは
   カーネルを書く側の都合なので、[04](04-PHASES.md) Phase 2 の着手時に決める)
2. **2048 トークン流した後の状態**が要る ([04](04-PHASES.md) Phase 2 の出口)。
   1 トークンでは `qwen_delta_rule` の蓄積誤差を見られない

## 6. 生成スモーク — **oQ4e-g64 は生きている**

[04](04-PHASES.md)「次の一手」7 の答え。**実測(手元)**、greedy (`temp 0` 相当)、
`--cache-core` (展開した core を持ち続ける、約 5 GB):

```
--text "日本の首都はどこですか。一文で答えてください。" --max-new 48 --cache-core
```

| 見るもの | 値 |
| --- | ---: |
| prefill 19 トークン (core の展開を含む) | 119.4 s |
| decode | **47.5 s/トークン** (46.9〜48.4、41 回) |
| 生成できたトークン | 41 (**外から止めたため 48 には届いていない**) |

出力 (`skip_special_tokens=False`):

```
<think>
The user is asking in Japanese: "Where is the capital of Japan? Please answer in one sentence."

The capital of Japan is Tokyo (東京).

I should answer in one sentence in
```

**日本語の問いを解釈し、正答に達し、指示 (一文で) を保持している。**
[11](11-OQ4E-G64-REBUILD.md) で **248 本を 8-bit g64 に打ち直した**チェックポイントが、
壊れていないことがこれで確定した。

**47.5 s/トークンはこの参照器の速度であって、モデルの速度ではない。**
float32 の numpy で、1 トークンごとに 40 層 × top-8 のエキスパートを
展開し直している。運用の数字は Phase 6 で実機のカーネルから取る。

### 6-1. テンプレートについて 1 つ

上流の `chat_template.jinja` は `add_generation_prompt` のとき
`<|im_start|>assistant\n` の**後ろに `<think>\n` を差し込む** (`enable_thinking`
が false のときは `<think>\n\n</think>\n\n`)。参照器は差し込まずに
`<|im_start|>assistant\n` で止めたが、**モデルは自力で `<think>` を出した**
(§4 の top-5、2 位に 9.6 差)。テンプレートの意図と重みの振る舞いが一致している。

参照器の CLI は system ブロックを省いた最小の ChatML を組む。**本番の
テンプレート適用は Phase 5** で、上流の jinja をそのまま同梱する
([04](04-PHASES.md) Phase 5)。

## 7. この参照器で測れないこと

| 測れないもの | 理由 |
| --- | --- |
| **4-bit が bf16 からどれだけ傷んだか** | 展開しているのは **4-bit を float32 に戻したもの**で、bf16 そのものではない。上流全体 (約 70 GB) の取得が要る。手元の `bf16-partial` は 518 本 3.50 GB |
| **速度・ヒット率・TTFT** | CPU の numpy。エキスパートキャッシュもページングも本ランタイムとは別物 |
| **実機カーネルの誤差床** | GPU が要る ([04](04-PHASES.md) Phase 2、`fp16_error_floor.py` と同じ手続き) |

**候補どうしの比較は測れる。**`--nll` が同じ文章に対する平均 NLL を出す。
bf16 という共通の物差しが無くても、**低い方が良い**とは言える。
ただし 1 本の文章では n=1 なので、**解釈を書く前に文章を 3 本以上**取る
(本計画の運用ルール)。

## 8. この文書が動かした結論

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) Phase 0 の 1 番目 | **完了。**ただし計画の `transformers` をランタイムとして使う形ではなく、層ストリーミングの自前実装 (§1) |
| [04](04-PHASES.md) Phase 0 の出口 | **まだ閉じない。**fixtures の絞り込みと 2048 トークン後の状態が残る (§5) |
| [04](04-PHASES.md)「次の一手」7 | **完了。**生成スモークが通った (§6) |
| [04](04-PHASES.md)「次の一手」8 | 実活性が手に入ったので**測れる**ようになった。まだ測っていない |
| [04](04-PHASES.md)「次の一手」9 | 物差し (`--nll`) は入った。**まだ測っていない** |
| [01 §3](01-MODEL.md) の算式 | **全部が上流実装に対して検証済み** (§3)。書き換えは無し |
| [02](02-CHECKPOINTS.md) の 2 候補 | 重みの再現誤差が付いた (§2)。**出力の品質差はまだ** |
