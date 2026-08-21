# 17. Phase 2 の残りカーネル — `qwen.metal` 7 本

**実測(手元)、GPU。**2026-08-21 夜。[15](15-PHASE2-GDN.md) で `qwen_delta_rule` が
通った続き。[04](04-PHASES.md) 次の一手 #13。

線形注意 30 層の入口と出口、full attention 10 層の q/k 後処理、MoE の 2 か所を
埋めた。**新規は全部 `Sources/TurboFieldfare/Metal/Qwen/qwen.metal`** に置き、
既存 `.metal` は 1 行も触っていない (Gemma 4 の実測値を凍結したまま second
architecture を足すため — [README](README.md) 運用ルール)。

| | |
| --- | --- |
| カーネル | 7 本 (`qwen.metal`)、Swift 側は `QwenKernels` |
| 検査 | `TurboFieldfareKernelCheck --qwen` で **29 本すべて緑** |
| 時間 | prefill 2048 で線形注意 30 層の**周辺**が 33.6 ms、full attention 10 層が 10.0 ms |
| 途中で分かったこと | ① fast math が減衰ゲートを床の 19 倍に落とす ② conv をトークン方向に切ると 50.0 → 21.9 ms ③ **本線の `lm_head` は 4-bit ではない**ので [03 §2-8](03-DESIGN.md) は書き直しが要る |

---

## 1. 書いたもの

| カーネル | 役割 | 置き換わる算式 |
| --- | --- | --- |
| `qwen_delta_qkv_prepare` | 因果 depthwise `conv1d` + SiLU + q/k の l2norm + q の `1/sqrt(Dk)` | [01 §3-6](01-MODEL.md) の `conv1d` 〜 `l2norm` |
| `qwen_delta_gates` | `g = exp(-exp(A_log)·softplus(a+dt_bias))` と `beta = sigmoid(b)` | 同 §3-6 の 2 行 |
| `qwen_delta_norm_gate` | `RMSNormGated` (**`+1` しない**) × `silu(z)` | 同 §3-6 の最終行 |
| `qwen_qkv_epilogue` | `q_norm` / `k_norm` + **Qwen 規約の partial RoPE** | [01 §3-2](01-MODEL.md) |
| `qwen_attn_output_gate` | `o *= sigmoid(gate)` | 同 §3-2 の `attn_output_gate` |
| `qwen_moe_shared_gate` | `shared *= sigmoid(w·x)` | [01 §3-4](01-MODEL.md) |
| `qwen_silu_mul` | `y = silu(gate) * up` | `hidden_act: silu` |

計画 ([03 §2](03-DESIGN.md)) との差は 2 つ:

1. **§2-5 の conv と `l2norm` を 1 本にまとめた** (`qwen_delta_qkv_prepare`)。
   l2norm の縮約はヘッド 128 チャネルの中で閉じ、conv の出力はその 128 チャネルが
   同じ threadgroup にいる。分けると同じ 33.5 MB をもう 1 往復することになる
2. **`qwen_delta_gates` は計画に無かった。**`g` を「掛ける値そのもの」にして
   `qwen_delta_rule` に渡すため、指数を取るところまでをここに置いた

`qwen_delta_qkv_prepare` の出力レイアウトは `qwen_delta_rule` の入力そのもの
(`q`,`k` が `[T, Hk, Dk]`、`v` が `[T, Hv, Dv]`)。`in_proj_qkv` の出力 `[T, 8192]` を
そのまま渡すと行の刻みが合わないので、ここで詰め直している。

## 2. 検査 (29 本すべて緑)

```
swift run -c release TurboFieldfareKernelCheck --qwen             # 既定 512 トークン
swift run -c release TurboFieldfareKernelCheck --qwen --qwen-tokens 2048
```

立て方は [15 §2](15-PHASE2-GDN.md) と同じ: **チェックポイントも fixtures も開かず**、
実物と同じ形の合成入力に対して GPU / CPU float32 (床) / CPU double (真値) の
3 通りを走らせる。出力が FP16 のものは許容 4e-3 (FP16 の刻み 2^-11 の 8 倍)、
FP32 のものは**床の 20 倍**。

**負例を 6 本入れた。**正例が通るだけでは物差しが働いている証明にならないので、
「このモデルで静かに壊れる道」を参照側に作って、同じ GPU 出力がそこから桁違いに
離れることを確かめる:

| 負例 | 何を間違える | GPU 出力の相対誤差 |
| --- | --- | ---: |
| conv タップの向き | `[C, K]` の K 軸を逆順に読む (MLX と上流 bf16 で軸順が違う — [10 §4](10-MLX4BIT-AUDIT.md)) | 0.83 / 1.03 |
| l2norm の eps | 和ではなく**平均**に足す (すぐ隣の RMSNorm と同じ形なので紛れる) | 0.91 |
| `dt_bias` の脱落 | 減衰ゲートの bias を忘れる | 0.84 |
| `1+w` を足す | `RMSNormGated` は 30 層の norm で**ここだけ足さない** ([01 §3-1](01-MODEL.md)) | 0.70 |
| Gemma の組 | partial RoPE を `(i, HD/2+i)` / 分母 `HD` で回す ([03 §2-2](03-DESIGN.md)) | 1.29 / 1.78 |
| (`--gdn` 側) 減衰の位置 | [15](15-PHASE2-GDN.md) | 0.084 |

正例の許容 4e-3 に対して、いちばん小さい負例でも 0.70 — **175 倍離れている。**

ビット一致を要求する検査が 3 本ある:

- **conv 状態の持ち越し** — T を 1 回で流したものと、32 トークンずつ切って状態を
  渡したものが**ビット一致**。窓が 4 なので、状態の書き戻しがずれていれば境界の
  トークンで必ず出る
- **`qkv_epilogue` の gate 側が無傷** — `q_proj` は 2 倍幅で後半が
  `attn_output_gate` 用。ここに norm も RoPE も掛かってはいけない
- (`--gdn` 側) 再帰状態の持ち越し

## 3. fast math が減衰ゲートを床の 19 倍に落とす

`qwen_delta_gates` の `g` だけが最初から許容ぎりぎりだった (1.16e-06、float32 の
床の 19 倍)。原因は `precise::` を書いても **Metal の既定 (fast math) では
超越関数が高速版のまま**であること。`MTLCompileOptions.mathMode = .safe` にすると
床に落ちる:

| | `g` の相対誤差 | float32 の床 |
| --- | ---: | ---: |
| fast math (既定) | 1.16e-06 | 6.2e-08 |
| **safe math (採用)** | **8.9e-08** | 6.2e-08 |

`g` は再帰状態に**毎トークン掛かる**ので、ここの誤差は 1 回の丸めでは終わらない。
代償は `qwen_delta_qkv_prepare` の +1.9% (48.9 → 49.9 ms / 30 層、当時の実装で
3 回とも同じ向き)。**この差なら精度を取る。**

`moduleLibrary(device:module:safeMath:)` を足して**`qwen` モジュールだけ**に効かせた。
`gdn` は既定のままなので、[15](15-PHASE2-GDN.md) の 125.7 ms は動いていない。

なお `softplus` は `log1p` を自前で持つ (MSL に無い)。素朴な `log(1+u)` は
u が小さいとき 1+u の丸めで有効数字を落とす — 減衰ゲートが 1 に貼り付く側が
まさにその領域なので、Kahan の補正を掛けてある。

## 4. 時間

```
swift run -c release TurboFieldfareKernelCheck --qwen --qwen-bench --qwen-tokens 2048
```

20 回の中央値。カーネル 1 本のマイクロベンチなので `bench.sh` の作法 (temp /
クールダウン) の対象ではない。数字は「このカーネルが 1 層に何 ms 使うか」であって
モデルの速度ではない。実物の活性での再測は結線 (Phase 3) の後。

| カーネル | 層数 | T=1 (ms) | T=2048 1 層 (ms) | 層数ぶん (ms) |
| --- | ---: | ---: | ---: | ---: |
| `qwen_delta_qkv_prepare` (R=32) | 30 | 0.010〜0.018 | 0.731 | **21.9** |
| `qwen_delta_norm_gate` | 30 | 0.006〜0.012 | 0.383〜0.407 | 11.5 |
| `qwen_delta_gates` | 30 | 0.006〜0.012 | 0.009 | 0.3 |
| `qwen_qkv_epilogue` | 10 | 0.006〜0.011 | 1.000 | 10.0 |

### 4-1. conv をトークン方向に切る (50.0 → 21.9 ms)

最初は「1 threadgroup = 1 ヘッド、時間は threadgroup の中の逐次ループ」で書いた。
`qwen_delta_rule` の幾何をそのまま真似たのだが、**depthwise conv の因果依存は
窓 4 ぶんしかない**ので時間は本当は並列に出せる。切らないと threadgroup が
64 個しか立たず、prefill 2048 で 40 GB/s (M3 Pro の帯域の約 4 分の 1) だった。

R (1 threadgroup が持つトークン数) を振った結果 (30 層ぶん、ms):

| R | 16 | **32** | 64 | 128 |
| --- | ---: | ---: | ---: | ---: |
| 30 層 | (先頭行は暖機で 25〜33) | **21.9** | 21.9〜22.0 | 22.7〜22.8 |

**R=32 を既定にした** (3 回とも 21.9)。読み直しは窓のぶん (R+3)/R = +9% 増えるが、
threadgroup が 64 → 4,096 になるほうが効く。約 96 GB/s。

`qwen_delta_norm_gate` は 0.383 ms で 50 MB を往復していて約 132 GB/s、
**ここはもう帯域で頭打ち**。`qwen_qkv_epilogue` は 70 GB/s で余地があるが、
10 層で 10.0 ms なので後回し。

### 4-2. ★ 中止線 (150 ms) は線形注意の**周辺まで数えると**超える

[05 §2](05-RISKS.md) #2 の中止線は「線形注意の 30 層合計 150 ms」。
チャンク 2048 の実測を足すと:

| | 30 層 (ms) |
| --- | ---: |
| `qwen_delta_qkv_prepare` | 21.9 |
| `qwen_delta_rule` ([15](15-PHASE2-GDN.md)) | 125.7 |
| `qwen_delta_norm_gate` | 11.5 |
| `qwen_delta_gates` | 0.3 |
| **合計** | **159.4** |

**`qwen_delta_rule` 単体は 125.7 ms で線の内側、周辺まで数えると 159.4 ms で外側。**
中止線が「再帰カーネルの逐次形が遅いか」を見るためのものである以上 ([05 §2](05-RISKS.md) #2 の
「検知」欄)、**再設計の引き金は引かない**。ただし [04](04-PHASES.md) Phase 4 の出口条件は
「線形注意の 30 層合計が 150 ms 以内」と書いてあり、この読み方だと外れる。
**どちらの定義で締めるかはユーザーの判断**なので、ここでは数字だけ置く。
なお 4-1 の 1 手で 28 ms 縮んでおり、`qwen_delta_rule` 側の TB / Dv の振り直しは
まだ 1 度も実物の活性でやっていない。

## 5. ★ 本線の `lm_head` は 4-bit ではない

[03 §2-8](03-DESIGN.md) は「`LMHeadChainInt4` に `D=2048` / `vocab=248,320` の
specialization を足す」と書いてあるが、**本線の `oQ4e-g64` の `embed_tokens` /
`lm_head` は 8-bit g64** ([02 §1](02-CHECKPOINTS.md))。バイトでも合う:
[12 §3](12-OQ4E-G64-AUDIT.md) の区画表で embed + lm_head が 1,080,688,640 B、
パラメータは 2 × 248,320 × 2,048 = 1,017M なので約 8.5 bit/weight
(4-bit なら 572 MB — [01 §5](01-MODEL.md))。
本リポジトリに INT8 の LM head 経路は無い (`grep lm_head Sources/.../logit.metal` は
`*_int4_*` だけ、実測 = ソース)。

したがって #13 の 4 番目は **「int4 の specialization を足す」ではなく
「INT8 の LM head chain を書く」**に変わる。これは [13 §4-2](13-PHASE1-REPACK.md) の
混在ビット幅 (#11) と同じ根から出ているので、**#11 の決着を待ってから着手する**のが
筋が通る (索引からビット幅を導くなら、LM head もその仕組みに乗る)。
**#11 はその通りに決着した** ([18](18-MIXED-BITS.md)) — 幅は
`Model.residentWeightBits(_:)` がテンソル名から返す。
語彙の末尾 243 行が未学習である件 ([10 §3](10-MLX4BIT-AUDIT.md)) は、
**`vocab` に 248,077 を渡して行を採点しない**だけで済む — マスクのコードは要らない。

## 6. 残り

| | やること | 状態 |
| --- | --- | --- |
| a | INT8 の LM head chain | **着手できる。**#11 は片づいた ([18](18-MIXED-BITS.md))。幅は `Model.residentWeightBits` で引く |
| b | `qwen_qkv_epilogue` の 70 GB/s | 余地はあるが 10 層で 10.0 ms。後回し |
| c | 実物の活性での再測 | Phase 3 の結線後 |
| d | `qwen_silu_mul` / `qwen_moe_shared_gate` の結線先 | prefill / decode の MoE 経路 (Phase 3) |

**Phase 2 で残っているカーネルは LM head だけ**になった。
