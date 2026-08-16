# 調査: prefill (pp) はどこまで行けるか — SSD か実装か

作成: 2026-08-16
スコープ: 調査。計測用の計装とプロトタイプは**すべて元に戻してある** (`git status` クリーン)。
目標: **300 pp tok/s**。表記は PLAN.md に合わせる: **実測** / **導出** / **未確認**

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

## 7. 中断メモ (2026-08-16) — 再開する人へ

### 7-1. いまのツリーの状態

**ソースは 1 行も変わっていない。**`git status` の差分は
このファイルを含む未追跡ドキュメントだけ。

本調査で入れた計装とプロトタイプは**全部 `git checkout -- Sources/` で捨てた**。
測定値だけがこのドキュメントに残っている形なので、
**再開時はまず §7-2 のパッチを当て直すところから始める。**
当て直せば §3 / §4 の数字はそのまま再現する (何度か往復して確認済み)。

### 7-2. 捨てたプロトタイプ (再現手順)

#### (a) チャンク上限の撤廃 — 定数 3 か所

| ファイル | 変更前 | 変更後 |
| --- | --- | --- |
| `Runtime/Configuration/RuntimeConfiguration.swift:29` | `allowedPrefillChunkTokens = [32, 64, 128]` | `[32, 64, 128, 256, 512, 1024, 2048]` |
| `Runtime/Prefill/PrefillRuntimeConfig.swift:171` | `maxChunkTokens = 128` | `2048` |
| `Runtime/Prefill/PrefillChunkScratch.swift:16` | `min(chunkTokens, 128)` | `min(chunkTokens, 2048)` |

これだけで動く。KV リング容量とスクラッチは自動で追随した。
**ただし `--max-context 8192` 以上 + 96 スロットは `ExpertCacheBudget` に弾かれる**
(§6 Gate 1 の宿題)。今回の測定は `--max-context 4096` で取った。

#### (b) 共有エキスパート MLP のバッチ化

1. `Runtime/Prefill/PrefillChunkScratch.swift:46`
   `var sharedExpertScratchElements: Int { sharedIntermediate }`
   → `{ chunkTokens * sharedIntermediate }`
2. `Kernels/Prefill/MoE/PrefillSharedExpert.swift` に
   `PrefillInt4QMM` と `context.pipeline("gelu_mul_fp16")` を持たせ、
   `encodeBlock` の `for row in 0..<queryCount` ループの**手前**に
   バッチ経路を足して `return` する (weightBits == 4 のときだけ):

   ```
   qmm(gate) -> scratchGate[t,F]
   qmm(up)   -> scratchUp[t,F]
   gelu_mul_fp16(count = t * F)   // 既存カーネルがそのまま使える。要素単位なので [t,F] で成立
   qmm(down) -> y[t,D]
   ```

   `PrefillInt4QMM.encode(t:n:k:)` に渡すのは
   gate/up が `(t, intermediate, d)`、down が `(t, d, intermediate)`。
   `xStrideElements == d && yStrideElements == d` のときだけ有効にすること
   (それ以外は既存の per-token ループに落とす)。

**効果**: 共有 MLP の GPU 時間 −49% (405 トークンで 1.362 s → 0.690 s)。
greedy 出力は `--temperature 0.0 --seed 1` で完全一致を確認済み。

#### (c) GPU 時間の計装

`RealForwardRunner` に `waitProfiled(_ key:_ cb:)` を足し、
`waitForCompletion` の代わりに使って `cb.gpuEndTime - cb.gpuStartTime` を種別に積算する。
差し替えた 4 か所は `executePrefillChunk` 内の

- ルータまでのコマンドバッファ (`cb.commit()` 直後) → `"attn"`
- `sharedCB` → `"shared"`
- `drainOldestPendingTile` の中 → `"moe"`
- `tailCB` → `"tail"`

で、`prefillChunked` の span ループ末尾で stderr にダンプする。
**§4 の表はこれで取った。恒久化するなら `bench.sh` から使える形にすること**
(§6 Gate 0)。

### 7-3. 計測フィクスチャの作り方

プロンプトはスクラッチ領域に置いたので**消えている**。同じものは次で作れる:

```python
import json
base = ("チャンク化した prefill が time to first token を短くしつつ、"
        "メモリ使用量を上限内に抑えられる理由を、KV キャッシュの確保タイミングと"
        "ルーティングされたエキスパートのフェッチ回数の観点から説明してください。")
# 反復回数 -> プロンプトトークン数:  8 -> 405,  16 -> 797,  40 -> 1973
json.dump([{"role": "user", "content": base * 40}], open("p4k.json", "w"), ensure_ascii=False)
```

主な測定コマンド:

```bash
# pp と expert io と peak
.build/release/TurboFieldfareCLI --model scratch/gemma4-qat.gturbo \
  --messages-file p4k.json --max-new 1 --max-context 4096 \
  --verification trusted-install --expert-cache-slots 16 --prefill-chunk-tokens 2048

# disk0 の実転送量 (別ターミナルで並走させて MB/s 列を積分する)
iostat -d -w 1 disk0

# ユニーク expert 数の再構成: chunk 32 でトレースを取り、連続ステップを層ごとに union
.build/release/TurboFieldfareCLI ... --prefill-chunk-tokens 32 --dump-expert-trace trace32.tsv
```

SSD 帯域 (§1) は `fcntl(fd, F_NOCACHE=48, 1)` を立てた `os.pread` で測った。
`packed_experts/layer_NN.bin` に対する 3,719,168 B のランダム読みを 300 回。

### 7-4. 次にやること

**Gate 1 (§6) から。**チャンク上限の撤廃は定数 3 か所で、効果は実測済み
(39 → 70 tok/s、メモリ 6.4 → 3.1 GB)。作業の本体は定数ではなく:

1. `ExpertCacheBudget` の再計算 — chunk 2048 で KV リングが
   `slidingWindow + chunkTokens` = 3072 行になる。**16K コンテキストで通るか未確認**
2. chunk 256/512/1024/2048 での数値検証を `TurboFieldfareValidation` に追加
3. greedy 出力一致を回帰テストに落とす (今は手動で 2 プロンプト確認しただけ)
4. 既定値は**まだ変えない**。`--prefill-chunk-tokens` で選べるようにするだけ

そのうえで §6 Gate 1 の合否 (**16 → 80 スロットで pp が 3% 以内**) を取り直す。
ここが崩れなければ、Gate 2 (SWA attention) が次の一撃になる。

### 7-5. 引き継ぐべき判断

- **expert キャッシュのスロット数チューニングはもうやらない。**§3 で無意味と実測された。
  既定 48 は chunk を上げた後に**下げる**方向で再評価する
- **300 tok/s のボトルネックは attention であって MoE ではない。**
  §4 の内訳が直感に反するので、着手前に必ずここを読み直すこと
- `RESULTS.md` §3-3 の「スロットを増やすと prefill 自体は遅くなる」と
  `PLAN_VISION.md` §4-5 の「チャンク上限 128 が画像スパンを分割する」は、
  **どちらも本調査と同じ根 (チャンク幅)**。Gate 1 が通れば両方片付く

---

## 8. 未確認

- `U(2048) ≈ 110` は chunk 1024 (97.9) からの外挿。トレースは 797 トークンぶんしかなく、
  2048 トークンの単一チャンクを直接 union していない
- §4 の内訳 (attention 本体 11 s、射影 6 s) は 797 トークン chunk 512 の
  分割計測からの外挿。1973 トークンで直接分割していない
- SWA attention の読み出し 790 GB は導出。実測していない (Gate 2 の前提条件)
- chunk 2048 + `--max-context 16384` が `ExpertCacheBudget` を通るか
- chunk 512 で 1 回だけ観測した 30.6 s の外れ値 (peak 11.4 GB、スワップの疑い)
- 現行 `prefill_dequant_int4_qmm_f16_block` が per-token GEMV とほぼ同速だった理由
  (バッチ Q は GPU 時間 −3% だった。帯域律速か 8×8 threadgroup の占有率か)
