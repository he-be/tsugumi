# 37. SWA の attention も帯の GEMM → softmax → GEMM に — 台帳 2.626 → 2.482 s (実測、2026-09-25)

[36 §3](36-FULL-ATTN-UNFUSED.md) の続き。全注意で使った 3 カーネルを headDim 256 のテンプレートにも広げ、SWA の 25 層 (16/8 ヘッド、窓 1024) に使った (a9d4ce6)。
S は 64 行ブロックごとに、そのブロックが見える鍵の帯だけを持つ。帯は `floor64(ブロックの最初の行の窓の始まり)` から `roundup64(窓 + 127)` 列 (窓 1024 で 1152 列)。全注意は、帯の始まり 0・幅が全鍵という特別な場合になった。
KV の ring (長い文脈で SWA の KV が折り返す) は、K と V の行を `鍵 % ring` で読む。
戻すのは `TF_PREFILL_ATTN_SWA=mpp` (26 の融合 `matmul2d` カーネル)。

証拠 (M6、この順に取った。どれも `bench/m6/results/2026-09-25-Mac18-5-<ラベル>/`):
`swa-unfused-kernel-check`、`attn_bench-swa-unfused`、`s1-{default,unfused}-{a,b}`、`s1-host-unfused`、`s1-greedy-unfused`、`swa-unfused-default-kernel-check`、`ledger-swa-unfused`、`s1-greedy-default`。

```
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 994a0ef / minos 15.0
(台帳の run は 994a0ef、ソースは a9d4ce6 と同じ。候補の run は 391f4c5〜983fb02 で、ソースは 391f4c5 と同じ)
```

コマンドは台帳の固定形 (`prefill_bench.sh`)。greedy は末尾に `--max-new 64 --temperature 0`。カーネル単体は `swift bench/m6/attn_bench.swift Sources/Tsugumi/Metal/Prefill/prefill.metal <カーネル>...` (SWA の形、25 層 × 2 チャンク)。
全 run exit 0、Swapouts 0。MBP の `Scripts/test.sh` は a9d4ce6 のソースで 1629 本通過。

**台帳** ([06](06-LEDGER.md)): 既定を変えた。`ledger-swa-unfused` **2.482 s (2.471〜2.984)**。前の行 (084de6b) の 2.626 s から −0.144 s (導出)。

## 0. 結論

1. **SWA 25 層のカーネル単体は 0.277 → 0.112 s** (sgmm GQA 0.848 s、MLX の融合版 0.285 s は [31](31-MLX-KERNELS-PER-STAGE.md))。
2. **壁時計 (c) は、同じバイナリの交互で 2.748 → 2.550 s** (各 n = 6)。GPU busy は 2.20 → 2.04 s。
3. **greedy 64 トークンの本文が変わった。**`c0707c48e5` → `b93b5f98b7` (新しい既定で 2 本とも同じ)。5 トークン目で「Prefill」が「 prefill」になり、以降は同じ文意で表記だけが違う (「KVキャッシュ」→「KV キャッシュ」)。本番の形 (2478 行、窓 1024、2 チャンク) の全行で、unfused と sgmm の出力の最大差は 1.2e-4 で、mpp と sgmm の差と同じ。帯の扱いの誤りではなく、拮抗した 2 候補の入れ替わりと読む (ロジットを出す口が無いので未確認)。[04 原則 5](04-GATES.md) に従い、記録して止めない。

## 1. 数字

### 1.1 カーネル単体 (`attn_bench-swa-unfused`、GPU s、25 層 × 2 チャンク)

| カーネル | 秒 | sgmm GQA との最大差 |
| --- | ---: | ---: |
| `attention_prefill_causal_sgmm_gqa_d256` | 0.848 | — |
| `attention_prefill_swa_mpp_d256` (26 の既定) | 0.277 | 1.2e-4 |
| unfused (37 の既定) | 0.112 | 1.2e-4 |

### 1.2 壁時計 (c)、既定と交互 (run 順、秒)

| 組 | 既定 (SWA は mpp) | 候補 (SWA も unfused) | 中央値 n = 6 (導出) |
| --- | --- | --- | --- |
| `s1` a / b | 2.853、2.733、2.637 / 2.842、2.762、2.632 | 2.626、2.554、2.545 / 2.624、2.476、2.416 | 2.748 → 2.550 |

- 台帳 `ledger-swa-unfused` (994a0ef): 2.984、2.504、2.471、2.482、2.481、2.473 s。

### 1.3 host の内訳 (`s1-host-unfused`、2・3 本目、秒)

| GPU busy | GPU idle | idle のうち `fetch` | `wait.front` | `fetch` | `wait.tile` |
| --- | --- | --- | --- | --- | --- |
| 2.043、2.041 | 0.403、0.303 | 0.326、0.249 | 0.857、0.856 | 0.807、0.693 | 0.652、0.703 |

36 の unfused (全注意だけ) では GPU busy 2.196〜2.210 s、`wait.front` 1.014〜1.027 s。

## 2. 分かったこと / 外れたこと

- 31 では `attn.swa` は MLX (融合版 0.285 s) と同じ程度で差が無い段と読んでいた。MLX の融合版がこの形の上限ではなかった。GEMM の形にすると、S の帯を書いて読み直す分を払っても短い。
- 正しさ: KernelCheck の SWA の検査に unfused を足した (窓 100、2 チャンク目、192 スロットで折り返した ring、頭 4 と 16 のまとめ方、窓を半分にした参照で落ちること)。group 64 の 143 件・group 32 の 173 件がみな PASS。unfused の rel は mpp と同じ 3.6〜4.1e-4。
- スクラッチは 36 と共用 (64 MiB)。SWA は 1 頭あたり 2048 × 1152 × 6 バイトで、1 チャンク目は 4 頭ずつ。peak は 36 と同じ 1.30〜1.31 GB。

## 3. 次に渡すもの

- 台帳は 2.482 s、到達目標 (mlx_lm 1.823 s) との差は 0.66 s (導出)。GPU busy は 2.04 s、GPU の空きは 0.30〜0.40 s で、その大半は `fetch` の間 (SSD からのページ)。
- 段の秒はこの既定の上で取り直していない。31 の残りの差は `rope` → `moe.down` → `moe.gateup`。
