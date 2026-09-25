# 03. 事実の台帳 — QFN と 2 機

値 / 出所。`bench/qfn/results/` の証拠が無い行は出所の文書名を書き、札を付ける。**証拠が無い行は無いと書いてある。**

## 1. モデル (DS4-IQ2、`~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/`)

| 項目 | 値 | 出所 |
| --- | --- | --- |
| 構造 | `qwen4exp`。48 層 = full attention 12 + Gated DeltaNet 36 (3 層に 1 つが attention)。hidden 2560、hyper-connection 4 本 (残差は 4 × 2560 fp32/トークン)。MoE 512 expert × top-10、expert ff 640、shared expert あり。MTP 1 層 (blk.48) | [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md)、[investigations/QWEN38_FLASH_NEXT_VERIFY_PLAN §2-2](../investigations/QWEN38_FLASH_NEXT_VERIFY_PLAN.md) |
| PLE (n-gram 埋め込み) | **層 1 (0 始まり) に 1 か所だけ。**1 トークンに 16 行 × 160 (2-gram 8 + 3-gram 8)。行番号は直前 3 トークンのハッシュだけで決まる (`Qwen38Runner.pleRows`) | [qwen38/31](../qwen38/31-PLE-BF16-VS-Q4_1.md)、コード |
| ファイル | 本体 GGUF 44,806,612,192 B (41.73 GiB)。`down/` サイドカー 13,337,903,104 B (12.42 GiB、既定で読む)。`bf16/` 146,964,480 B。PLE BF16 102,400,495,520 B (95.37 GiB、**既定**)。PLE Q4_1 32,000,157,440 B (29.80 GiB) | `ls -la` (2026-09-25)、[qwen38/16](../qwen38/16-WEIGHT-DIET.md)、[31 §7](../qwen38/31-PLE-BF16-VS-Q4_1.md) |
| routed expert の大きさ | 1 個 1,489,920 B (gate/up 各 422,400 + down 645,120)。down サイドカー後は 1,387,520 B。**1 層 762,839,040 B、48 層 34.10 GiB (サイドカー後 約 32 GiB)** | VERIFY §1、[qwen38/16 §1・§6](../qwen38/16-WEIGHT-DIET.md) |
| dense の読み | 約 5 GB/パス (hc の F16 1.3 GB を含む)。residency set は既定で切っている (`Q38_DENSE_RESIDENT=1` で戻る) | [qwen38/15 §1](../qwen38/15-DECODE-PLAN.md)、[09](../qwen38/09-12K-MEMORY-VERIFY.md) |
| チャンクが触る expert | T = 128 で 185 本、1024 で 322 本 (層あたり)。数百トークンでも 400 本以上 | [qwen38/06 §2-2](../qwen38/06-ROUTED-GEMM.md)、[08 §2](../qwen38/08-ROUTED-MIXED.md) |
| チェックポイント | GDN 36 層の状態 + conv 履歴 + PLE 履歴 = 117.6 MiB/本、文脈長に依らない。既定は SSD (`F_NOCACHE`、unlink 済み fd)。取得 52〜115 ms、復元 34〜37 ms | [qwen38/18 §0](../qwen38/18-PROMPT-CACHE-CHECKPOINTS.md) |
| 運用点 | 32K、KV Q8_0 (インデクサ鍵 F16)、thinking off、サンプリングは公式値固定、MTP n_max 1 (server) | [qwen38/17](../qwen38/17-SERVER-APP-WIRING.md)、[23](../qwen38/23-KV-Q8-32K.md)、[37 §2](../qwen38/37-RESTART-ON-PC-Q4KXL.md) |

## 2. 今のランナーの読み方 (コード、2026-09-25 時点)

| 場面 | 読み方 | 出所 |
| --- | --- | --- |
| prefill (T ≥ `Q38_PREAD_MIN_T` = 32) | 層の top-10 が決まってから、選ばれた expert の範囲を `GGUFFile.preadRanges` で 4 スレッド・16 MB ブロックで読み (ページキャッシュを温める)、そのあと routed の GPU を投入。**読みと GPU は同じ層の中で直列** | `Qwen38Runner.moeRouted`、[qwen38/05](../qwen38/05-EXPERT-READ.md) |
| decode (T < 32) | `F_RDADVISE` (隣接をまとめる)。層またぎの予測 (`Q38_PREVIEW_N`) は既定 0 | [qwen38/12](../qwen38/12-PIPELINE-PREVIEW.md)、[13](../qwen38/13-KERNEL-TO-GPU-WAIT.md) |
| パイプライン | `cb1(L)` (attention/GDN、hc、router) → wait → host top-10 → shared commit → 読み → `cb2(L)` (routed) を commit して待たない → `cb1(L+1)`。`Q38_PIPELINE=1`、`Q38_SHARED_LATE=1` | [qwen38/12 §1](../qwen38/12-PIPELINE-PREVIEW.md) |
| PLE の行 | `forwardBody` の頭で、チャンク全トークンの 16 行を **host の 1 スレッドで mmap から順に読む**。先行発行なし | `Qwen38Runner.pleRows` |
| routed の GPU | T ≥ 1024 は expert ごとに float32 に展開 + MPS sgemm (グループ 4)。組が 24 未満の expert は per-pair カーネル。**fp32 の GEMM** | [qwen38/06](../qwen38/06-ROUTED-GEMM.md)、[08](../qwen38/08-ROUTED-MIXED.md) |
| top-10 | host の Swift ループ (Double の softmax、512 本から 10 本) | `Qwen38Runner.moeRouted` |

## 3. MBP 単体の prefill (旧文書の実測、`bench/qfn/results/` の証拠なし)

| 項目 | 値 | 出所 |
| --- | --- | --- |
| ベンチ | 8K 87.4 s (チャンク 2048)、12K 133.6 s、**32K 336.5 s** (n = 1) | [qwen38/08 §0](../qwen38/08-ROUTED-MIXED.md)、[09 §0](../qwen38/09-12K-MEMORY-VERIFY.md)、[23 §0](../qwen38/23-KV-Q8-32K.md) |
| チャンク 2048 の 1 本 (8K の 2 本目以降) | 20〜22 s: pre-router 壁 6.4〜7.2 (GPU 5.2〜5.9) / route 6.3〜6.5 (pread 約 5、top-10 約 1.3) / routed 壁 6.8〜6.9 (GPU 6.2〜6.3) / PLE 0.1〜2.2 | [qwen38/08 §4](../qwen38/08-ROUTED-MIXED.md) |
| 層あたり (**導出**、上の行 ÷ 48) | GPU 0.24〜0.25 s、pread 0.10 s、top-10 0.027 s | 上の行 |
| アプリの実ラウンド (R2、新規平均 2,650) | prefill 37.3 s = 前半 24 層 (pre-router 10.41 + route 5.11) + 後半 24 層 (10.17 + 4.96) + PLE 3.15 + MTP KV 2.65 + head 0.55。GPU は pre-router 6.2 + routed 10.1。実読み 70.7 GB (pread 51.1 GB) | [qwen38/43 §2-1](../qwen38/43-KVA-SPEED-ON-APP-ROUND.md) |
| アプリの回帰 | prefill 秒 ≈ 6.20 + 新規 ÷ 89.6 (83 ラウンド、R² 0.968)。新規 30〜255 のラウンドは 5〜8 s・14〜30 GB。8 トークンのチャンクが 2.5〜2.7 s | [qwen38/42 §1](../qwen38/42-KVA-SPEED-ON-APP-ROUND-PLAN.md)、[43 §2-3](../qwen38/43-KVA-SPEED-ON-APP-ROUND.md) |
| 1 要求の SSD 読み | 43〜46 GB (1,577 トークン + 64 生成、毎回冷やす) | [qwen38/48 §3](../qwen38/48-ONLINE-EXTRACT-BY-E4B.md) |
| 1 ターン | ラウンド 3〜7、新規の中央値 1,278 トークン/ラウンド、1 ターンで 8〜14k トークン | [qwen38/34](../qwen38/34-ROUND-BREAKDOWN.md)、ユーザーのメモ |
| PLE が冷えているとき | 1 チャンク 2.5〜5.4 s | [qwen38/03 §3](../qwen38/03-PREFILL.md)、[05 §2](../qwen38/05-EXPERT-READ.md) |
| pread の速さ (MBP 内蔵) | 6.2〜6.6 GB/s。mmap のフォルトは 0.73〜1.48、advise 後 0.97 GB/s | [qwen38/05](../qwen38/05-EXPERT-READ.md) |
| decode | 32K で MTP off 5.83、on 7.51 tok/s。ページインは 156〜189 MB/ステップ | [qwen38/23](../qwen38/23-KV-Q8-32K.md)、[11 §1](../qwen38/11-VERIFY-PASS-SPLIT.md)、[13 §3](../qwen38/13-KERNEL-TO-GPU-WAIT.md) |
| M6 で QFN を動かした実測 | 段の秒は [10 §1.2](10-Q1-SPLIT-DECISION.md) | `bench/qfn/results/2026-09-25-Mac18-5-e1-stages/` |

## 4. 2 機 (m6-prefill/03 と investigations の実測)

| 項目 | MBP (Mac15,6) | M6 (Mac18,5) | 出所 |
| --- | --- | --- | --- |
| RAM / `recommendedMaxWorkingSetSize` | 18 GB / 15,360 MiB | 16 GB / 12,124 MiB (wired limit 14336 のとき 14336 MiB、[21](21-Q5-M6-OVERLAP.md)) | [m6-prefill/03 §1](../m6-prefill/03-FACTS.md) |
| 内蔵 SSD (`F_NOCACHE`) | AP1024Z 6.68 GB/s | AP0256Z **3.34 GB/s** (QD を上げても伸びない) | [investigations/M6_SSD_BANDWIDTH §0](../investigations/M6_SSD_BANDWIDTH.md) |
| 外付け `SSD256` (TB4、型番は未記載) | — | ランダム 1.57 GB/s (p50 8.5 ms @QD4)、連続 2.10。**内蔵と同時に読んで合計 4.83〜4.92 GB/s** | [m6-prefill/44 §1.1](../m6-prefill/44-EXPERT-MIRROR-EXTERNAL-SSD.md) |
| GPU | fp16 GEMM 5.94 TFLOP/s | fp16 18.61、**fp32 4.85** TFLOP/s、帯域 144.4 GB/s | [m6-prefill/03 §1](../m6-prefill/03-FACTS.md) |
| QFN の routed GEMM (1 層、T = 2048) | 腕ごとの秒は [10 §1.4](10-Q1-SPLIT-DECISION.md)。f32 sgemm の腕は両機でほぼ同じ | | |
| OS / Swift | macOS 15.7.5 / Swift 6.3.3 / Xcode 26.6 | macOS 27.0 / Swift 6.4 / CLT | TWO_MACHINE_DEV |
| wired limit | — | `iogpu.wired_limit_mb` は再起動で 0 | [m6-prefill/29](../m6-prefill/29-REFERENCE-RUNTIMES-ON-M6.md) |

## 5. TB4 直結と重みの置き場 (2026-09-25、手測り。証拠ディレクトリなし。生 TCP の実測は [10 §1.1](10-Q1-SPLIT-DECISION.md))

| 項目 | 値 |
| --- | --- |
| link | IP over Thunderbolt、`bridge0`、両端 link-local。MBP 169.254.218.152 → M6 169.254.13.135 (この日)。生 TCP 37.8 Gb/s、ssh aes128-gcm 2.15 GB/s は TWO_MACHINE_DEV §3 (証拠なし) |
| この日の ssh | 2 GiB を 1.43 s (起動込み)。本体 GGUF 44.8 GB を M6 内蔵へ 39 s。**PLE BF16 102.4 GB を M6 外付けへ 約 12.5 分 (約 140 MB/s、外付けの書き込みが遅い)** |
| M6 内蔵 (`/`) | 228 GiB、転送後の空き **46 GiB**。Gemma `gemma4-qat-sym.gturbo` 15 GB が同居 |
| M6 外付け (`/Volumes/SSD256`) | 238 GiB、転送後の空き **111 GiB**。Gemma の写し 12 GB とプローブ用 `layer_*.bin` 20 GB が同居 (Q4 の写し 約 23 GiB は入る) |
| M6 に置いたもの | `~/dev/tsugumi/scratch/Qwen3.8-Flash-Next-DS4-IQ2/` (本体・down・bf16・manifest、`tokenizer` → `../Qwen3.8-Flash-Next-tokenizer`)、`ple-bf16` → `/Volumes/SSD256/qfn/ple-bf16/` (symlink)。**照合済み**: 本体は `SHA256SUMS` と一致、PLE BF16 は両機の `shasum -a 256` が一致 (d9fa5c72…8dff3)。PLE Q4_1 は置いていない |
| MBP 内蔵 | 926 GiB、空き 135 GiB。QFN 一式 181 GB は `~/LLM/` |
