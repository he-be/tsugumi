# 03. M6 の実測台帳

**証拠のパスが無い行は「証拠なし」と書く。**書き足すときは [02 §2](02-EVIDENCE.md) の形の証拠ディレクトリを先に作る。
値は両機を並べる (MBP は対照であって記録値ではない)。

## 1. 機械が返すもの (2026-09-22、**実測**)

| 項目 | M6 mini | MBP | 証拠 |
| --- | --- | --- | --- |
| 機種 / チップ / RAM | Mac18,5 / Apple M6 / 16 GB | Mac15,6 / Apple M3 Pro / 18 GB | 各 `hostinfo.txt` |
| OS / ツールチェーン | macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT | macOS 15.7.5 (24G624) / Swift 6.3.3 / SDK 26.5 / Xcode 26.6 | 同上 |
| `MTLGPUFamily` | apple10 **true**、apple11 **true**、apple12 false、metal4 **true** | apple9 まで、apple10 false、metal4 false | [M6](../../bench/m6/results/2026-09-22-Mac18-5-gpu_family_probe/output.txt) / [MBP](../../bench/m6/results/2026-09-22-Mac15-6-gpu_family_probe/output.txt) |
| `#available(macOS 26.0)` → MSL | true → 4.0 | false → 3.2 | 同上 |
| `tensorops.metal` を MSL 4.0 で | **コンパイル OK、`mpp_prefill_affine_threadgroup_f16` の PSO OK** (1024 threads / simd 32) | **コンパイラが `-std=metal4.0` を拒否** | 同上 |
| 同 MSL 3.2 で | 関数 0 本 (`__HAVE_TENSOR__` 未定義) | 同じ | 同上 |
| `recommendedMaxWorkingSetSize` | 12,124 MiB | 15,360 MiB | 同上 |
| MPS fp16 GEMM 4096³ | **18.61 TFLOP/s** (5 本 18.58〜18.62) | 5.94 (5.93〜5.96) | [M6](../../bench/m6/results/2026-09-22-Mac18-5-gpu_peak_probe/output.txt) / [MBP](../../bench/m6/results/2026-09-22-Mac15-6-gpu_peak_probe/output.txt) |
| MPS fp32 GEMM 4096³ | 4.85 (4.84〜4.85) | 5.13 (5.10〜5.16) | 同上 |
| GPU の帯域 (1 GiB、GPU 時計、各 7 本の中央値) | 読み **144.4 GB/s**、書き 143.0、コピー (読み + 書き) 136.2 | (記録値なし) | [M6](../../bench/m6/results/2026-09-24-Mac18-5-gpu_bw_probe/output.txt) |

**導出 (上の実測から)**: fp16 / fp32 の比は M6 が **3.84**、M3 Pro が 1.16。
M6 の Apple 製カーネルには fp16 専用の行列経路があると読める。**自前カーネルからそこに届くかは別問題で、G2 が測る。**
fp32 は M6 のほうが 5% 遅い。fp32 累積を残す設計は M6 で得をしない可能性がある (**未確認**、G2 で fp32 累積の腕も取る)。

## 2. 旧文書にあって証拠ディレクトリが無いもの (取り直す)

| 項目 | 旧文書の値 | 置き場 | 状態 |
| --- | --- | --- | --- |
| tensor 無しの prefill、`bench/l.json` 2478 tok、32 スロット、chunk 2048 | M6 pp 288 / GPU 側 7.63 s、MBP pp 235 / 9.26 s (各 6 本) | [旧 §7-4a](../investigations/M6_MAC_MINI_TARGETING.md) の表のみ | **取り直し済** → [10](10-G0-BASELINE.md)。M6 の pp 288 は再現せず |
| 同じ本文を外部ランタイムで prompt 処理 (暖まったプロセス、ハッシュ無し) | 無し (旧 §5-3 は「16 GB では成立しない」) | [ref-llamacpp-iq3xxs](../../bench/m6/results/2026-09-24-Mac18-5-ref-llamacpp-iq3xxs/summary.tsv) / [ref-mlx-3bit](../../bench/m6/results/2026-09-24-Mac18-5-ref-mlx-3bit/summary.tsv) | **取った** → [29](29-REFERENCE-RUNTIMES-ON-M6.md)。llama.cpp 4c5957c IQ3_XXS 2.393 s (n = 6、tensor API なし 4.494 s n = 3)、mlx_lm 0.31.3 3-bit 1.823 s (n = 6、peak 13.3 GB、Swapouts あり)。`iogpu.wired_limit_mb=14336` |
| M6 の io が GPU と重ならない (io が 0.85 → 3.33 s でも GPU 側不変) | 同上 | 同上 | **方向のみ再現** (io の振れ 0.35 s)。[10 §2](10-G0-BASELINE.md) |
| `MPPPrefillInt4QMM` が group 32 で nil | 単体プローブで確認とある | 旧 §7-1a、プローブ未保存 | **原因どおりと確認し、直した** → [16](16-G1-MPP-GROUP32.md) |
| SSD `F_NOCACHE` 3.34 GB/s | 表は詳細 | [M6_SSD_BANDWIDTH](../investigations/M6_SSD_BANDWIDTH.md) | 表のみ、生ログ無し。**値は信用してよい** (プローブは `bench/rec_probe.c` / `io_depth_probe.c` で再現可) |
| `.macOS(.v26)` でビルドが通る (minos 26.0) | エラー 0、74.9 s | 旧 §7-5a | 証拠なし。**やらないと決めた項目なので取り直さない** |

## 3. 公称のまま (手元で取っていない)

| 項目 | 値 | 出典 |
| --- | --- | --- |
| 16 GB 構成の DRAM 帯域 | 153 GB/s (24/32 GB は 170)。実測は §1 の GPU の帯域 | [Apple 技術仕様](https://www.apple.com/mac-mini/specs/) |
| GPU コア数 / Neural Accelerator | 12 コア、各コアに搭載 | Apple newsroom |
| `matmul2d` (MPP) が Neural Accelerator に到達する | — | BaseRT (arXiv 2607.19438)、WWDC26 330。**G2 で測る** |
