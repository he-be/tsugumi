# 調査: prefill (pp) はどこまで行けるか — SSD か実装か

作成: 2026-08-16
更新: 2026-08-16 (attn バケットの分離計測と full attention の置き換え、§7-4 - §7-6)
目標: **300 pp tok/s**。表記は PLAN.md に合わせる: **実測** / **導出** / **未確認**

**現在地: 1973 トークンで 78.5 pp (出荷既定)。調査開始時は 37.2。**
**2478 トークンでは 70.2 → 91.1 pp (§7-5、作業ツリーに未コミット、数値検証が残っている)。**
何をしたかは §7。以下 §1-§6 は調査時点の記録で、数字はそのとき (チャンク上限 128、
カーネル変更前) のもの。

---

## 0. 結論を 3 行で

1. **SSD は律速ではない。**チャンクを 128 → 2048 にするだけで、disk0 の実転送量が
   **110.5 GB → 11.8 GB (9.4 分の 1)**、expert I/O 時間が **23.9 s → 1.85 s** になる (**実測**)。
2. **エキスパートを載せない設計は正しい。**chunk 2048 なら
   **16 スロット (peak 3.1 GB) と 80 スロット (peak 10.3 GB) が同速** (どちらも 70.2 tok/s、**実測**)。
   キャッシュは無意味になり、SSD ストリーミングだけで足りる。
3. **残りは全部 GPU カーネル。**最良構成で壁時計の **97% が GPU busy**。
   **300 tok/s は到達可能**だが、必要な実効演算性能は **2.0 TFLOP/s** (現在 0.47、**導出**)。
   最大の的は routed MoE ではなく **SWA attention (GPU 時間の約 40%)**。

現状の出荷既定 39.2 tok/s に対し、**チャンク上限を上げるだけで 70.4 tok/s (1.8 倍) が
メモリ半分で出る** (**実測**、§3)。そこから 300 までは 4.3 倍のカーネル書き直し。

---

## 1. 計測環境

- コミット `efd5d02`、Apple M3 Pro (GPU 18 コア) / 18 GB / **DRAM 150 GB/s**、
  macOS 15.7.5、Swift 6.2.4
- `TurboFieldfareCLI --model scratch/gemma4-qat.gturbo --max-new 1 --verification trusted-install`
- プロンプト 1973 トークン (`--max-context 4096`)。一部の内訳は 797 / 405 トークン
- 30 層 (SWA 25 / full 5) / expert 128 / top-8 / `hiddenSize` 2816 /
  `moeIntermediateSize` 704 / `ffnIntermediate` 2112 / `slidingWindow` 1024 /
  `headDim` 256 (SWA) / 512 (full) / `expertStride` 3,719,168 B / **quant group 32**
- packed_experts = 13 GB (layer_NN.bin × 30、各 476 MB)

### SSD の実力 (**実測**、`F_NOCACHE` でページキャッシュを迂回)

| パターン | スループット |
| --- | ---: |
| 連続読み (1 層ファイル 476 MB) | **4.74 GB/s** |
| ランダム 3.72 MB pread (1 層内) | 6.87 GB/s (0.54 ms/expert) |
| ランダム 3.72 MB pread (30 層またぎ) | **4.19 GB/s** (0.89 ms/expert) |
| オフセット昇順 3.72 MB pread | 6.77 GB/s |

**シングルスレッドで 4.2 GB/s、expert 1 個 0.9 ms。**
`expertStride` = 3.72 MB は NVMe にとって十分大きく、ランダムでも連続と同等に出る。
以下ではこの **4.2 GB/s** を SSD 帯域として使う。

---

## 2. 支配方程式 — なぜチャンク幅がすべてを決めるのか

1 層あたりの routed expert は 128 個 = **476 MB**。全 30 層で **14.3 GB**。
これは**チャンクの大きさに関係なく一定**である。
一方、1 チャンクで処理するトークン数 C は自由に選べる。したがって:

```
トークンあたりの expert バイト = U(C) × 30 層 × 3.72 MB / C     (U(C) ≤ 128)
```

`U(C)` = 1 層 1 チャンクで触るユニーク expert 数。chunk 32 の実トレースを
層ごとに union して**厳密に再構成**した (**実測**、`--dump-expert-trace`):

| C | U(C) | expert バイト/トークン | SSD 4.2 GB/s での pp 上限 (キャッシュ 0%) |
| ---: | ---: | ---: | ---: |
| 32 | 53.3 | 186 MB | 23 tok/s |
| 128 (**現状の上限**) | 71.3 | **62.2 MB** | **68 tok/s** |
| 256 | 76.7 | 34.3 MB | 122 tok/s |
| 512 | 89.2 | 19.4 MB | 216 tok/s |
| 1024 | 97.9 | 10.7 MB | 393 tok/s |
| 2048 | 約 110 (**導出**) | 6.0 MB | 700 tok/s |

**chunk 128 では、キャッシュを一切使わない場合の SSD 上限が 68 tok/s しかない。**
出荷既定の実測 39-46 tok/s はこの天井のすぐ下にいる。
つまり「SSD にエキスパートを置く以上避け難い」ように見えていたのは、
**SSD の性質ではなく `maxChunkTokens = 128` という定数の帰結**だった。

### 検証 — disk0 の実転送量 (**実測**、`iostat -d -w 1 disk0`)

1973 トークン / 16 スロット:

| chunk | disk0 実転送 | prefill | expert io |
| ---: | ---: | ---: | ---: |
| 128 | **110.5 GB** | 48.96 s | 21.96 s |
| 2048 | **11.8 GB** | 31.44 s | 2.10 s |

理論値 (§2 の表 × 1973 トークン) は chunk 128 で 122.7 GB、chunk 2048 で 12.3 GB。
**実測と一致する。**ページキャッシュが数字を作っているのではなく、
本当にディスクが 110 GB 動いていた。

---

## 3. 「エキスパートを載せない」は成立する (**実測**)

1973 トークン、共有 MLP バッチ化プロトタイプ込み:

| 構成 | prefill | **pp** | expert io | **peak RAM** |
| --- | ---: | ---: | ---: | ---: |
| **出荷既定** (48 スロット / chunk 128) | 50.27 s | **39.2** | 23.90 s | 6.39 GB |
| 16 スロット / chunk 2048 | 28.03 s | **70.4** | 2.03 s | **3.11 GB** |
| 48 スロット / chunk 2048 | 28.10 s | 70.2 | 2.09 s | 6.73 GB |
| 80 スロット / chunk 2048 | 28.12 s | 70.2 | 2.29 s | 10.27 GB |

**16 / 48 / 80 スロットが完全に同速。**expert キャッシュのヒット率は 0.0% で、
それでも 70 tok/s 出る。**キャッシュを 5 倍積んでも 1 秒も速くならない。**

出荷既定に対して **1.8 倍速く、メモリは半分以下**。
プロジェクトの前提 (「全部載せない」) は正しく、
**むしろスロットを積むほうが間違い**だったことになる。
`RESULTS.md` §3-3 の「スロットを増やすと prefill 自体は遅くなる」はこの現象の断片。

greedy 出力は chunk 128 と 2048 で一致を確認 (**実測**、temp 0.0 / seed 1)。

---

## 4. 残っているのは GPU だけ (**実測**)

1973 トークン / 16 スロット / chunk 2048 / 共有 MLP バッチ化:

```
attn    gpu = 17.06s  (63%)   ← attention 本体 + q/k/v/o 射影 + rope + KV copy + router
moe     gpu =  6.84s  (25%)   ← routed expert
shared  gpu =  3.32s  (12%)   ← 共有 MLP (バッチ化後)
tail    gpu =  0.04s
      GPU busy 合計 27.26s  /  壁時計 28.09s  = 97%
expert io      1.85s  (完全にオーバーラップ)
peak           3.13 GB
```

**壁時計の 97% が GPU busy。**I/O も CPU 同期も、もう見えない。

### 内訳の推定 (**導出**、797 トークン chunk 512 での分割計測から外挿)

| | GPU 時間 | 演算量 | 実効 |
| --- | ---: | ---: | ---: |
| **attention 本体** | **約 11 s** | 0.93 TFLOP | **0.085 TFLOP/s** |
| q/k/v/o 射影 | 約 6 s | 4.4 TFLOP | 0.73 TFLOP/s |
| routed MoE | 6.8 s | 5.6 TFLOP | 0.82 TFLOP/s |
| 共有 MLP | 3.3 s | 2.1 TFLOP | 0.64 TFLOP/s |
| **合計** | **27.3 s** | **13.1 TFLOP** | **0.47 TFLOP/s** |

**最大の的は routed MoE ではなく attention。**理由は帯域:
`PrefillAttention.encodeCausal` の tensorops 経路は `headDim == 512` 条件で
**full attention 5 層にしか当たらず**、SWA 25 層は `attention_prefill_causal_tiled` に落ちる。
そのディスパッチは `MTLSize(width: queryCount, height: numQHeads)` =
**(クエリ, ヘッド) ごとに threadgroup 1 個**で、クエリ間で K/V タイルを共有しない。

読み出し量の**導出**: SWA 層で 1 クエリあたり K+V = 1024 × 256 × 2 × 2 B = 1 MB。
1973 クエリ × 16 ヘッド × 25 層 = **約 790 GB**。
150 GB/s なら L2 が全く効かない場合で 5.3 s、実測 11 s はこの桁。
**クエリブロック化 (64 トークン単位で KV タイルを threadgroup メモリに置く) だけで
理屈上 60 分の 1 になる。**

---

## 5. どこまで行けるか — 3 つのルーフライン (**導出**)

300 tok/s = 1973 トークンを **6.58 s**。

| 制約 | トークンあたり | 300 tok/s での要求 | 手持ち | 判定 |
| --- | ---: | ---: | ---: | --- |
| **SSD** (chunk 2048) | 6.0 MB | 1.8 GB/s | **4.2 GB/s** | **余裕 (43%)** |
| **DRAM** (expert + dense + KV、完全再利用時) | 約 12 MB | 3.6 GB/s | **150 GB/s** | **問題外 (2.4%)** |
| **GPU 演算** | 6.6 GFLOP | **2.0 TFLOP/s** | fp32 6.4 / fp16 12.8 TFLOP/s | **31% / 16% — ここが律速** |

- **SSD は 43% しか使わない。**chunk 4096 まで上げれば 22%。
  キャッシュを併用すればさらに下がる。**設計の前提は完全に守られる。**
- **DRAM 150 GB/s は一度も効いてこない。**ただし条件付き:
  現行の MoE カーネルは pair 単位 GEMV なので論理読み出しが 892 MB/トークン =
  300 tok/s で **267 GB/s** となり DRAM を超える。
  **expert 単位 GEMM (重みを threadgroup メモリに載せて expert 内の全トークンで再利用) が必須。**
  そうすれば 6.0 MB/トークンに落ちる。
- **GPU 演算が唯一の壁。**必要な 2.0 TFLOP/s は fp32 ピークの 31%、
  `simdgroup_matrix` fp16 ピークの 16%。int4 デクォンタイズ付き GEMM としては
  **要求が高いが標準的な水準**。現在 0.47 TFLOP/s。

### 到達点の見通し (**導出**)

| 段階 | 想定 pp (1973 tok) | 根拠 |
| --- | ---: | --- |
| 出荷既定 | 39 tok/s | **実測** |
| チャンク上限撤廃 + 共有 MLP バッチ化 | **70 tok/s** | **実測** (§3) |
| + SWA attention のクエリブロック化 | 約 150 tok/s | attention 11 s → 1 s (**導出**) |
| + MoE を expert 単位 GEMM に | 約 210 tok/s | 6.8 s → 2 s (**導出**) |
| + 射影/共有 MLP を simdgroup_matrix に | **約 330 tok/s** | 9.3 s → 2.5 s (**導出**) |
| 理論上限 (2.5 TFLOP/s 到達時) | 約 380 tok/s | |

**300 tok/s は届く。ただし 4 段すべてが要る。**
1 段目は実測済み、2 段目 (attention) が最大の一撃、3-4 段目でようやく到達する。
なお短いプロンプト (500 トークン前後) では attention の比重が下がるので、
同じ作業でより高い pp が出る。

---

## 6. 確かめるプラン

各段階に**予測値**を先に書く。外れたら仮説が間違っている、という形にする。

### Gate 0 — 現状の再現 (半日)

- `bench.sh` に **pp 専用モード**を足す (`--max-new 1`、prefill 秒とトークン数から pp)。
  今の `ptime` は壁時計しか見ておらず、GPU busy と disk0 転送量を取っていない。
- 取るもの: pp / GPU busy 合計 / expert io / **disk0 実転送量 (`iostat`)** / peak。
- **合否**: 出荷既定 (48 スロット / chunk 128) で 1973 トークン
  **39 ± 3 tok/s、disk0 110 ± 10 GB**。

### Gate 1 — チャンク上限の撤廃 (1-2 日、最優先)

変更は定数 3 か所 (`PrefillRuntimeConfig.maxChunkTokens`、
`RuntimeConfiguration.allowedPrefillChunkTokens`、`PrefillChunkScratchLayout` 内の clamp)
だが、付随する検証が要る:

- KV リング容量 `min(maxContext, slidingWindow + maxPrefillChunkTokens)` が
  chunk 2048 で 3072 行になる。**16K コンテキストでの `ExpertCacheBudget` 再計算が必須**
  (今回 `--max-context 8192` + 96 スロットは起動時に弾かれた)。
- `routePartials` スクラッチは `C × topK × D × 2 B` = chunk 2048 で 92 MB。
- 数値検証: `TurboFieldfareValidation` を chunk 256/512/1024/2048 で通す。
  greedy 出力の chunk 128 一致は 2 プロンプトで確認済みだが、**回帰テストに落とす**。
- **既定値は変えない。**まず `--prefill-chunk-tokens` で選べるようにするだけ。

**予測 (これが外れたら §2 の支配方程式が間違っている)**:

| | pp | disk0 | expert io |
| --- | ---: | ---: | ---: |
| 16 スロット / chunk 2048 | **62 ± 4 tok/s** | **12 ± 1 GB** | **2.0 ± 0.3 s** |
| 80 スロット / chunk 2048 | 同上 (差 3% 以内) | 同上 | 同上 |

**合否**: スロット数を 16 → 80 に振っても pp が 3% 以内で動かないこと。
動いてしまったら「キャッシュはもう要らない」という結論が崩れる。

### Gate 2 — SWA attention のクエリブロック化 (1-2 週、最大の一撃)

- `attention_prefill_causal_tiled` を、1 threadgroup が **Q ブロック (32 or 64 トークン) ×
  1 ヘッド**を担当し、KV タイルを threadgroup メモリに段階的に載せる形に書き換える。
- あるいは `attention_prefill_full_tensorops_2d_validity_v2` を
  `headDim 256 / numKVHeads 8` に一般化する (SWA 層に当てるにはリング対応も要る)。
- **先に測るべき単体数値**: `TurboFieldfareKernelCheck` に
  「SWA attention 単体の GPU 時間と読み出しバイト数」を出すケースを足す。
  現状の 790 GB (**導出**) を**実測に置き換えてから**着手する。
  ここが導出のままだと、効果を後で説明できない。
- **予測**: attention 本体 11 s → 1.0 ± 0.5 s。1973 トークンで **pp 140-160 tok/s**。
- **合否**: GPU busy 合計が 27.3 s → 17 s 以下。

### Gate 3 — routed MoE を expert 単位 GEMM に (2-3 週)

- グルーピング (`PrefillMoEGrouping`)、タイル化、スロット stream は**そのまま使える**。
  置き換えるのは `prefill_grouped_routed_moe_batched_phase1` / `_down` と
  `encodeStreamedBatched` のマイクロバッチループのみ。
- 1 threadgroup = 1 expert × トークンブロックにして、
  gate/up/down の重みタイルを threadgroup メモリに載せて再利用する。
- **予測**: MoE 6.8 s → 2.0 ± 0.5 s。論理読み出しが 892 MB/トークン →
  6.0 MB/トークンに落ちること (**これは `iostat` ではなく Metal のカウンタで確認する**)。
- **合否**: pp 200 tok/s 以上、かつ DRAM 帯域見積りが 300 tok/s 換算で 20 GB/s 以下。

### Gate 4 — 射影と共有 MLP を simdgroup_matrix int4 GEMM に (2-3 週)

- 現行 `MPPPrefillInt4QMM` は `affineGroupSize == 64` 専用で、
  **group-32 の QAT チェックポイントでは丸ごと死んでいる** (`MPPPrefillInt4QMM.swift:22-35`)。
  新カーネルは group 32/64 の両方を扱えること。
- 共有 MLP のバッチ化 (§3、`PrefillSharedExpert.encodeBlock` の per-token ループ撤廃 +
  スクラッチを `chunkTokens × F` に拡張) はここに含める。単独で GPU −49% 実測済み。
- ついでに: full 層で K と V を**同じ重みで 2 回射影している**
  (`RealForwardRunner.swift:633`、`attentionKEqV` のとき)。射影の 1.3%、無料。
- **予測**: 射影 6 s + 共有 3.3 s = 9.3 s → 2.5 ± 0.7 s。**pp 300-340 tok/s**。
- **合否**: 実効 2.0 TFLOP/s 以上。

### 全体の合否

**1973 トークンで pp ≥ 300 tok/s、peak RAM ≤ 4 GB、disk0 転送 ≤ 15 GB、
greedy 出力が chunk 128 と一致。**

### やらないこと

- **expert キャッシュのスロット数チューニング。**§3 で無意味と実測された。
  むしろ既定 48 を**下げる**検討のほうが筋がよい (chunk を上げた後に再評価)。
- **モデルの再量子化。**group-32 が遅いのは事実だが (`RESULTS_QAT.md` §2-2)、
  それは decode の話で、pp のボトルネックではない。

---


## 7. 実装した結果 (2026-08-16)

Gate 0 / Gate 1 は通過。Gate 2 は「クエリブロック化」を入れたが**予測を外した** (§7-3)。
Gate 4 の一部だった共有 MLP のバッチ化はここに前倒しした。

### 7-1. 1973 トークンでの推移 (**実測**、2 回インターリーブ、`--max-context 4096`)

| 段階 | prefill | **pp** | expert io | disk0 | peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 調査開始時の出荷既定 (48 スロット / chunk 128) | 51.6 s | **37.2** | 23.0 s | 105.5 GB | 6.0 GB |
| + カーネル 2 本 (同じ 48/128 設定) | 46.6 s | 42.3 | 24.2 s | — | 6.1 GB |
| + chunk 2048 (16 スロット) | 24.8 s | **79.6** | 2.06 s | 12.6 GB | **3.1 GB** |
| **新しい出荷既定** (48 スロット / chunk 2048) | 25.1 s | **78.5** | 2.08 s | — | 6.7 GB |

**2.1 倍。**予測 (§6 Gate 1 = 62 ± 4 tok/s) はカーネル変更前の実測 62.1 tok/s で的中し、
そこにカーネル 2 本が乗って 78.5 になった。

チャンク幅 2048 では **16 / 48 / 80 スロットが 1% 以内で同速**のまま (§3 の再現)。
スロットは prefill には効かない。48 を既定に残したのは decode のため。

### 7-2. 入れたもの

1. **チャンク上限の撤廃** — `allowedPrefillChunkTokens` に 256/512/1024/2048 を追加、
   `maxChunkTokens` を 2048 に。**既定も 2048 に変更した。**
   - KV リングと prefill スクラッチは**設定値**で計算する (以前は全体の上限定数を使っていた)。
     `ExpertCacheBudget` はスクラッチも数えるようになり、エラーに内訳が出る。
   - 既定 48 スロットなら 4K-64K の全コンテキストで通る (**実測**)。
     96 スロット以上は 8K 以上で弾かれるが、これは主にスロット側の 10.7 GB が理由
     (chunk 2048 の追加ぶんは KV 0.4 GB + スクラッチ 0.26 GB)。
2. **SWA attention のクエリブロック化** — `attention_prefill_causal_qblock_d256`。
   1 simdgroup が 4 クエリ × 1 ヘッドを持ち、K/V 行をレジスタに 1 回読んで使い回す。
   threadgroup 同期は 1 回もない。headDim 256 = SWA 25 層にだけ当たる。
3. **共有 MLP のバッチ化** — チャンク全行を 1 QMM で通す (`PrefillSharedExpert`)。
   調査時のプロトタイプをそのまま恒久化。GPU 6.49 s → 3.31 s (**−49%**、予測どおり)。
4. **GPU 時間の計装** — `TF_PREFILL_GPU_PROFILE=1` で stderr に 1 行出る。
   `bench.sh pp` から使える。§4 の内訳が導出でなく実測になった。

数値の同一性: greedy (temp 0.0 / seed 1) で
**tiled 対 qblock / per-token 対 batched / chunk 128 対 256/512/1024/2048** の
すべてが完全一致 (**実測**、40 トークンおよび 24 トークン)。

### 7-3. Gate 2 の予測は外れた

予測は「attention 本体 11 s → 1.0 ± 0.5 s、pp 140-160」。実際は:

```
attn   13.9s (58%)   ← 射影 + rope + KV + router 込み。変更前は 17.1s
shared  3.3s (14%)
moe     6.8s (28%)
       GPU 合計 24.1s / 壁時計 25.1s = 96%
```

クエリブロック化で attention 系は **17.1 s → 13.9 s (−3.2 s)** しか下がっていない。
帯域だけの問題ではなかった、ということ。新カーネルはキー 1 本ごとに
`simd_sum` を 4 回 (クエリ 4 本ぶん) 回すので、有効な FMA 8 個に対して
シャッフル縮約が十数命令付く。**縮約がクリティカルパスに残っている。**

次に効くのは、この縮約自体を消すこと =
`simdgroup_matrix` (8×8 タイル) で QK^T と PV を回す構成。
full 層用の `attention_prefill_full_tensorops_2d_validity_v2` が既にその形なので、
それを headDim 256 + リング + スライディング窓に一般化するのが素直。

> **この段落は §7-4 の分離計測で否定された。**13.9 s の内訳を取ると
> attention 本体は 2.0 s しかなく、その大半は SWA ではなく **full 5 層**だった。
> tensorops 経路は qblock 経路より演算あたり 8 倍遅い。一般化する方向は逆で、
> **qblock を headDim 512 に広げるのが正しかった** (§7-5)。
> SWA 側の縮約も律速ではなく、帯域だった (§7-6)。

### 7-4. attn バケットの分離計測 (2026-08-16、**実測**)

§7-3 が「射影と attention 本体の比率が分からない」で止まっていたので、
`TF_PREFILL_GPU_PROFILE=2` を足した。層ごとに 1 本だったコマンドバッファを
グループごとに切り、それぞれの `gpuStartTime`/`gpuEndTime` を足す
(`PrefillGPUProfile.Detail`、`RealForwardRunner.cutProfiled`)。

**計装のコストは +1.3%** (attn 19.90 s → 20.15 s、**実測**)。
コマンドバッファが 1 チャンクあたり 30 本から 240 本に増えるが、
その程度で済むので分離計測はほぼ無害に取れる。

以下このセクションの数字は**プロンプト 2478 トークン**のもの
(`bench/l.json` が §7-1 当時の 1973 から変わっている)。
同じ設定の基準値は **pp 70.2** (prefill 35.29 s、48 スロット / chunk 2048 /
`--max-context 4096` / QAT)。§7-1 の 78.5 は 1973 トークンでの値で、
長いほど attention の二次項が効くので直接は比べられない。

| グループ | GPU | attn 内の比 | 演算量 (**導出**) | 実効 |
| --- | ---: | ---: | ---: | ---: |
| q/k/v 射影 | **6.36 s** | 32% | — | — |
| o 射影 | **3.06 s** | 15% | — | — |
| (射影 合計) | 9.42 s | 47% | 5.57 TFLOP | **0.59 TFLOP/s** |
| **attention 本体 / full 5 層** | **8.31 s** | **41%** | 0.50 TFLOP | **0.06 TFLOP/s** |
| attention 本体 / SWA 25 層 | 1.72 s | 9% | 0.82 TFLOP | 0.48 TFLOP/s |
| rope + per-head norm | 0.27 s | 1% | | |
| post-attention + router | 0.42 s | 2% | | |
| embed / input norm / KV copy | 0.02 s | 0% | | |
| **合計** | **20.15 s** | | | |

**§7-3 の見立ては逆だった。**
「SWA 25 層が遅く、full 5 層は tensorops で速い」と思っていたが、
実際は **full 5 層が 8.31 s で、SWA 25 層は 1.72 s**。
層あたりにすると full は 1.66 s、SWA は 0.069 s で **24 倍**の開きがある。
演算量あたりでも full の tensorops 経路は SWA の qblock 経路の **8 分の 1**。
§7-3 が次の的に挙げた「`attention_prefill_full_tensorops_2d_validity_v2` を
headDim 256 に一般化する」は、**いちばん遅いカーネルを広げる作業**だった。

`attention_prefill_full_tensorops_2d_validity_v2` が遅い理由 (**導出**、コード読み):

- 1 threadgroup = **1 クエリ** × 8 ヘッド。クエリ間で K/V を共有しない
  (§4 で `attention_prefill_causal_tiled` を批判したのと同じ形)。
- キータイル 64 本ごとの softmax を `if (lid < 8)` で **128 スレッド中 8 スレッド**が
  直列に回す (最大取り + exp の 2 周)。tensor op 本体より
  こちらがクリティカルパスに乗っている。

### 7-5. full attention を qblock に置き換えた (**実測**)

`attention_prefill_causal_qblock_impl` はテンプレート
`<kElemsPerLane, kQBlock>` なので、headDim 512 は
`<16u, 2u>` を足すだけ (`attention_prefill_causal_qblock_d512`)。
full 層は窓なし・リングなしだが、`slidingWindow` に
`startPosition + t` が入るので既存のコードがそのまま全可視になる。
`kQBlock` を 2 にしたのはレジスタで、`q_reg` と `acc` が
`kQBlock × kElemsPerLane` 本ずつ要るため 4 だと 128 本を超えて溢れる。

| | 変更前 | 変更後 |
| --- | ---: | ---: |
| attention 本体 / full 5 層 | 8.31 s | **0.30 s** (**28 分の 1**) |
| attn バケット合計 | 20.15 s | **12.03 s** |
| GPU 合計 | 32.94 s | **24.82 s** |
| prefill (壁時計) | 35.41 s | **27.20 s** |
| **pp (2478 トークン)** | **70.2** | **91.1** |

**+30%。**カーネル 1 本 (テンプレート実体化 20 行) とディスパッチ表の差し替えだけ。

置き換え後の内訳 (**実測**):

```
qkv=6.31s(52%) oproj=3.06s(25%) attn.swa=1.72s(14%) attn.full=0.30s(3%)
post=0.42s(3%) rope=0.20s(2%) norm+kvcopy+embed=0.02s
                                          attn 合計 12.03s
shared=4.15s  moe=8.58s  tail=0.05s       GPU 合計 24.82s / 壁時計 27.20s = 91%
```

**検証の状態**: `RuntimePrefillAttentionPath` に `.causalQBlock` を足して既定にした。
`.fullTensorOps2DValidityV2` は残してあるので A/B は取れる。
`PrefillAttentionTests` の headDim 512 のケース (`full-origin` /
`full-short-gqa` / `full-production-gqa` /
`prefillAttentionProductionDimsBoundedVisibility`) は既定 `.causalTiled` 経由で
**新カーネルに当たるようになった**ので CPU 参照との突き合わせはそこで効く。
ブロック境界 (kQBlock 2 / threadgroup 16) を跨ぐケースを
`qBlockFullAttentionMatchesReferenceAcrossBlockBoundaries` として追加した。
**ただしこの環境ではテストを実行できていない** —
Xcode が入っておらず CommandLineTools だけなので `Testing` モジュールが解決できず、
`Scripts/test.sh` が (触っていない Repack / Format ターゲットも含めて) 落ちる。
**greedy 出力の一致もまだ取っていない。**再開時の最初の作業はこの 2 つ。

### 7-6. 次の的 (GPU 24.8 s の内訳から)

| 的 | 現状 | 実効 | 手段 |
| --- | ---: | ---: | --- |
| **q/k/v/o 射影** | **9.37 s** | 0.59 TFLOP/s | §6 Gate 4。`MPPPrefillInt4QMM` が group-32 で死んでいる |
| routed MoE | 8.58 s | | §6 Gate 3 (expert 単位 GEMM、手つかず) |
| 共有 MLP | 4.15 s | | 同じく Gate 4 |
| attention 本体 (SWA) | 1.72 s | 0.48 TFLOP/s | **帯域律速。優先度は低い** |
| attention 本体 (full) | 0.30 s | 1.7 TFLOP/s | 済み |

**射影と MoE と共有 MLP で GPU の 89%。**つまり残りは全部
「int4 GEMM を `simdgroup_matrix` で書く」という一つの作業に集約された。
attention は 2.0 s / 24.8 s = 8% まで落ちたので、もう主戦場ではない。

SWA の 1.72 s は帯域で説明がつく (**導出**): qblock は 4 クエリごとに
窓を読み直すので、K+V の実読み出しは 2478 トークンで **約 210 GB**、
1.72 s なら **122 GB/s** = DRAM 150 GB/s の 81%。
これ以上は「1 回読んだ K/V をもっと多くのクエリで使う」しかなく、
`kQBlock` を上げるか threadgroup メモリで共有するかだが、
取れるのは最大 1.7 s なので後回しでよい。

300 tok/s は 2478 トークンを 8.26 s。GPU 24.8 s に対してまだ 3.0 倍。

---

## 8. 未確認

- **`attention_prefill_causal_qblock_d512` の数値検証** (§7-5)。
  テストが実行できておらず、greedy 一致も未取得。**再開時の最初の作業**
- full 層の attention 本体が 0.30 s = 1.7 TFLOP/s と、SWA (0.48) より
  演算あたり速い理由。1 層の K+V が 2478 トークンで 10 MB しかなく
  丸ごとキャッシュに載っている、という仮説を立てただけ
- `kQBlock` を 2 より上げたときの full 層の挙動 (レジスタが溢れる見込み、未測定)
- `U(2048) ≈ 110` は chunk 1024 (97.9) からの外挿
- chunk 512 で 1 回だけ観測した 30.6 s の外れ値 (peak 11.4 GB、スワップの疑い)
- 現行 `prefill_dequant_int4_qmm_f16_block` が per-token GEMV とほぼ同速だった理由
- クエリブロック幅 4 が最適か (8 にするとレジスタが溢れる見込み、未測定)
