# 36. 全注意 (headDim 512) を MLX と同じ GEMM → softmax → GEMM に — 台帳 2.897 → 2.740 → 2.626 s (実測、2026-09-24〜25)

[31 §3](31-MLX-KERNELS-PER-STAGE.md) の差の順で、`post.router`・`shared` の次の `attn.full` に手を入れた。
全注意の 5 層 (16/2 ヘッド、headDim 512) は、`matmul2d` を使わない query-blocked のカーネルで動いていた。
2 段で既定を変えた。先に SWA の `matmul2d` カーネルを d512 に広げ (102fe93)、次に MLX と同じ形の 3 カーネルにした (f4075ac)。
戻すのは `TF_PREFILL_ATTN_FULL=mpp64` (1 段目の融合カーネル) か `=qblock` (変更前)。前置きが長くてスクラッチ (64 MiB) に 1 頭分の S も入らないときは、既定でも 1 段目に落ちる。

証拠 (M6、この順に取った。どれも `bench/m6/results/<日付>-Mac18-5-<ラベル>/`):
`full-attn-kernel-check`、`attn_bench-full-mpp`、`a1-{default,mpp64}-{a,b}`、`a1-greedy-{default,mpp64}`、`full-attn-default-kernel-check`、`ledger-full-attn-mpp` (以上 09-24)、
`unfused-kernel-check` (M6 でコンパイル失敗)・`unfused-kernel-check-2`、`attn_bench-unfused`、`a2-{default,unfused}-{a,b}`、`a2-host-{default,unfused}`、`a3-host-unfused` (09-24)、
`a3-{default,unfused}-{a,b}`、`a3-greedy-unfused`、`unfused-default-kernel-check`、`ledger-full-attn-unfused`、`ref-mlx-sdpa512` (09-25)。

```
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 084de6b / minos 15.0
(1 段目の台帳は c08e2de、ソースは 102fe93 と同じ。候補の run は 123f3a1〜3a28a91)
```

コマンドは台帳の固定形 (`prefill_bench.sh`、`--verification trusted-install`)。greedy は末尾に `--max-new 64 --temperature 0`。
カーネル単体は `TF_ATTN_BENCH_SHAPE=full swift bench/m6/attn_bench.swift Sources/Tsugumi/Metal/Prefill/prefill.metal <カーネル>...` (5 層 × 2 チャンク、GPU 秒、5 回の中央値)。
全 run exit 0 (`unfused-kernel-check` を除く)、Swapouts 0。MBP の `Scripts/test.sh` は 102fe93 と f4075ac のソースで 1629 本通過。

**台帳** ([06](06-LEDGER.md)): 既定を 2 回変えた。`ledger-full-attn-mpp` **2.740 s (2.722〜3.110)**、`ledger-full-attn-unfused` **2.626 s (2.617〜3.030)**。前の行 (ef29110) の 2.897 s から −0.271 s (導出)。

## 0. 結論

1. **MLX は headDim 512 で融合カーネルを持たない。**`mx.fast.scaled_dot_product_attention` は 0.0727 s、GEMM → softmax → GEMM を手で書いた形は 0.0746 s で同じ ([`ref-mlx-sdpa512`](../../bench/m6/results/2026-09-25-Mac18-5-ref-mlx-sdpa512/)、n = 3)。31 の `attn.full` 0.073 s は、GEMM の速さで QKᵀ と PV を回した秒だった。
2. **同じ形を 3 カーネルで書くと、全注意は 0.042 s** (カーネル単体、qblock 0.331 s、1 段目の融合カーネル 0.156 s)。MLX の 0.073 s より短いのは、因果マスクで全部隠れる S のタイルを飛ばしているため (MLX は矩形全体を計算する)。
3. **壁時計 (c) は、同じバイナリの交互で 2.909 → 2.791 s (1 段目) と 2.767 → 2.686 s (2 段目)** (各 n = 6)。greedy 64 トークンの本文は 3 構成とも `c0707c48e5`。

## 1. 数字

### 1.1 カーネル単体 (GPU s、5 層 × 2 チャンク)

| カーネル | `attn_bench-full-mpp` | `attn_bench-unfused` |
| --- | ---: | ---: |
| `attention_prefill_causal_qblock_d512` (変更前) | 0.374 | 0.331 |
| `attention_prefill_full_mpp_d512_r32` (32 行、4 SG) | 0.179 | — |
| `attention_prefill_full_mpp_d512_r64` (64 行、8 SG、1 段目の既定) | 0.158 | 0.156 |
| unfused (scores → softmax → PV、2 段目の既定) | — | 0.042 |

- qblock の 2 回の値の差 (0.374 と 0.331) は調べていない。出力の最大差は、どの組も qblock に対して 2.4e-4。

### 1.2 壁時計 (c)、既定と交互 (run 順、秒)

| 組 | 既定 | 候補 | 中央値 n = 6 (導出) |
| --- | --- | --- | --- |
| `a1` qblock / mpp64 | 3.135、2.906、2.875 / 2.953、2.842、2.912 | 2.913、2.809、2.715 / 2.883、2.773、2.738 | 2.909 → 2.791 |
| `a2` mpp64 / mpp64 (§2) | 3.021、2.769、2.765 / 2.863、2.734、2.735 | 2.878、2.794、2.848 / 2.864、2.729、2.744 | 2.767 / 2.821 |
| `a3` mpp64 / unfused | 2.920、2.778、2.734 / 2.852、2.724、2.756 | 2.765、2.702、2.629 / 2.778、2.669、2.607 | 2.767 → 2.686 |

### 1.3 host の内訳 (2・3 本目、秒)

| 構成 | GPU busy | GPU idle | idle のうち `fetch` | `wait.front` | peak GB |
| --- | --- | --- | --- | --- | --- |
| mpp64 (`a2-host-default`) | 2.303、2.306 | 0.367、0.340 | 0.312、0.282 | 1.117、1.126 | 1.23〜1.24 |
| unfused (`a3-host-unfused`) | 2.210、2.196 | 0.471、0.342 | 0.400、0.289 | 1.027、1.014 | 1.30〜1.31 |

## 2. 分かったこと / 外れたこと

- **`a2-unfused` は unfused を通っていなかった。**全注意の層は窓に 0 ではなく前置き全体 (`startPosition + t`) を渡しており、候補の条件 (窓 0) で外れていた。GPU busy が既定と同じ 2.305 s だったので気づいた (3a28a91 で直した)。結果として `a2` は同じ構成どうしで、中央値が 0.054 s 離れた。**この測定環境で、同じ構成の n = 6 の中央値は 0.05 s 程度揺れる。**
- 1 段目は GPU 0.2 s の縮みのうち約 0.12 s、2 段目は GPU busy 0.10 s の縮みのうち約 0.08 s が壁時計に届いた。2 段目の残りは、`fetch` の間の GPU の空き (SSD からのページ) に移った (1.3、run 2)。
- `unfused-kernel-check` は M6 の MSL 4.0 で `op.template` が要るコンパイルエラーで落ちた。MBP は tensor の経路をコンパイルしないので、この種の誤りは M6 でしか出ない。
- 正しさ: KernelCheck の全注意の検査 (CPU の softmax と比べる。qblock、mpp32・mpp64、unfused の頭 2 と 16 のまとめ方、1 キーずらしたマスクで落ちること) が group 64 の 129 件・group 32 の 159 件でみな PASS。unfused の rel は 3.7〜4.4e-4。
- スクラッチ: S は f32、P は f16、行の和は f32。1 頭あたり `行数 × roundup64(kvValidCount) × 6` バイト。台帳の形では 1 チャンク目が 2 頭ずつ、2 チャンク目が 8 頭ずつ。peak は +0.07 GB。

## 3. 次に渡すもの

- 台帳は 2.626 s、到達目標 (mlx_lm 1.823 s) との差は 0.80 s (導出)。GPU busy は 2.20 s。
- 同じ形 (S を帯で持つ GEMM → softmax → GEMM) は、SWA の 25 層 (`attn.swa`、27 の段で 0.27 s、MLX の融合版 0.285 s) にも使える。窓 1024 の帯だけを持てば 1 頭あたり 2048 × 1088 × 6 バイト。ring の読み方は行ごとの剰余で足りる。
- 31 の残りの差は `rope` → `moe.down` → `moe.gateup`。
