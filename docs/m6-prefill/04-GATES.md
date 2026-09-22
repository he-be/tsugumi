# 04. Gate — 問い・測り方・合否

各 Gate は **証拠ディレクトリと結果文書が揃って初めて閉じる**。合否の閾値はここに先に置き、計測後に動かさない。
順序: G0 → G1 → G2 → G3 → G4。G5 と G6 は G0 の後なら G1〜G4 と独立に進められる。
**G2 が不合格なら G3 / G4 はやらない。**tensor ops の線を閉じ、G5 と G6 に資源を回す。

対象はまず Gemma 経路 (`RealForwardRunner` / `Metal/Prefill/prefill.metal`)。tensor 経路の既存物 2 本
(`Kernels/TensorCore/MPPPrefillInt4QMM.swift`、`Kernels/Attention/PrefillAttention.swift` の `.fullTensorOps2DValidityV2`) が
Gemma からしか呼ばれていないため。Qwen 経路は G6 で判定してから。

## G0. 基準線 — M6 は何を返し、素の pp はいくつか

| # | 測る | 方法 | 状態 |
| --- | --- | --- | --- |
| G0-1 | `MTLGPUFamily`、MSL 4.0 のコンパイル、MPP カーネルの PSO | `bench/m6/run_probe.sh gpu_family_probe` | **済** (両機、[03 §1](03-FACTS.md)) |
| G0-2 | Apple 製カーネルの GEMM 天井 (fp16 / fp32) | `bench/m6/run_probe.sh gpu_peak_probe` | **済** (両機、[03 §1](03-FACTS.md)) |
| G0-3 | tensor を使わない現行コードの prefill (pp、io、GPU 側) | `bench/m6/prefill_bench.sh g0-baseline 6` (両機) | **未** |
| G0-4 | G0-3 の GPU 内訳 (routed MoE / attention / 射影 / 共有 MLP) | `TF_PREFILL_GPU_PROFILE=1 … g0-profile 3` (M6) | **未** |

**合格**: 両機 6 本ずつの `summary.tsv`。M6 の中央値が `10-G0-BASELINE.md` に載り、
**以降の「N 倍」の分母は G0-4 の routed MoE の秒数と G0-3 の prefill 壁時計**と決まる。
**同時に確かめる**: run ごとの io と prefill の差から「io は GPU と重なっていない」(旧 §7-4a) が再現するか。再現すれば G5 の的が確定する。

## G1. tensor 経路を開ける — group 32 のモデルで MPP が本当に走るか

| 何を | どこ |
| --- | --- |
| family ゲート → **PSO 生成の成否**に置換 | `PrefillAttention.swift:74` (`supportsFamily(.apple10)`) |
| `MPPPrefillInt4QMM` の group 64 専用を **32/64 両対応**に (scale/bias をグローバル K 位置から索く。simdgroup 版に同じ修正が既にある) | `MPPPrefillInt4QMM.swift:21` |
| 経路の**ログ**: どの QMM 経路で走ったかを stderr に 1 行 (`TF_PREFILL_GPU_PROFILE` 下でよい) | 新規 |
| 環境変数で経路を固定できる口 (`TF_PREFILL_QMM=mpp` / `simdgroup`) — 同じバイナリで A/B を取るため | `TF_PREFILL_QMM` は既にある。値を足す |

**合格**: (a) M6 で `TsugumiKernelCheck` の QMM 検査が MPP 経路で CPU 参照と一致、(b) greedy 出力が G0-3 と同一、
(c) `bench/m6/prefill_bench.sh g1-mpp 6` の `env.txt` に腕が記録され、ログに MPP のディスパッチが残る。
**速くなることは合格条件ではない** (q/k/v/o 射影は GPU 時間の 1 割強しかない)。数字は記録する。
**MBP では**: `swift build` と既存テストが通ること (経路が死んだまま挙動不変)。

## G2. `matmul2d` の素の実効値 — 自前カーネルから何 TFLOP/s 出るか

**この Gate が tensor ops の線の生死を決める。**「Neural Accelerator は `matmul2d` からしか叩けない」は公称であり、
自前カーネルがどれだけ取れるかは誰も測っていない。

作るもの: `bench/m6/matmul2d_probe.swift` + `.metal` (未作成)。同じ形状・同じ入力で 3 腕:

| 腕 | 中身 |
| --- | --- |
| A | 既存の `simdgroup_float8x8` タイル (現行 `prefill_moe_gemm_int4` と同じ 64×64×32、fp16 in / fp32 acc) |
| B | `mpp::tensor_ops::matmul2d` (fp16 in / fp32 acc)、同じタイル境界 |
| B' | 同 (fp16 acc) — fp32 が M6 で遅い件 ([03 §1](03-FACTS.md) 導出) の確認 |
| C | MPS (天井、G0-2 と同じ) |

形状は G0-4 の内訳で最大の的 (routed MoE、expert 1 個あたり平均 128 行 × 実寸の N / K) と、射影の (2048 × N × K)。
int4 デクォンタイズは**入れない** (純粋な GEMM の差を先に見る)。次に int4 → fp16 の展開を挟んだ腕を足す。

**合格**: 実運用の形状で **B ≥ 1.5 × A** かつ **B ≥ 0.5 × C**。両方満たせば G3 へ。
**不合格**: 片方でも満たさなければ「tensor ops は自前経路では取れない」と結論し、G3 / G4 を**閉じる**。
書く先: `11-G2-MATMUL2D-PROBE.md`。

## G3. routed MoE の expert GEMM を `matmul2d` 化 (旧 M6-2)

対象: `Metal/Prefill/prefill.metal:831` `prefill_moe_gemm_int4` の内側。int4 → fp16 のタイル展開はそのまま、累積を `matmul2d` に。
腕は環境変数で切り替え (`TF_PREFILL_MOE` に値を足す)、同じバイナリ。

**測り方は本番ループのみ** (`prefill_bench.sh g3-moe-mpp 6` + `-profile 3`)。単体ベンチの数字で判断しない ([05 §2](05-RISKS.md))。
**合格**: greedy 出力同一、G0-4 比で routed MoE の GPU 秒が下がり、**壁時計も下がる**。
GPU が下がって壁時計が動かないなら「io が露出した」と記録して G5 を先にやる。
書く先: 12-G3-MOE-MATMUL2D.md。

## G4. attention と射影の tensor 化 (旧 M6-3 / M6-4)

`.fullTensorOps2DValidityV2` 対 `.causalQBlock` の A/B には **attention 経路を切り替える口が CLI / 環境変数に無い**。先にそれを足す。
射影は G1 で開いた MPP 経路の数字をそのまま使う。合否は G3 と同じ形。書く先: 13-G4-ATTN-PROJ.md。

## G5. チャンク境界の io 直列の解消 (機種非依存)

G0-3 で「io の増分がそのまま壁時計に乗る」が再現したら着手。チャンク N の GPU 実行中にチャンク N+1 の expert を読む形。
`executeExpertCachePlan` が `concurrentPerform` で読む間コマンドバッファを積む人がいない ([05 §3](05-RISKS.md)) のが的。
実装と正しさは MBP、数字は M6。**M6 は SSD が半分なので効き代は M6 のほうが大きい** (導出: G0-3 の io 列を見る)。
**合格**: greedy 同一、M6 で prefill 壁時計 − GPU 側 の差が縮む。書く先: 14-G5-IO-OVERLAP.md。

## G6. Qwen 経路への波及 — io と gpu のどちらが重いか

Ornith (`~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa-baked`、21.9 GB) を M6 内蔵に置き、
`TF_QWEN_STAGE_PROFILE` で prefill の gpu / io ms/tok を取る (MBP の旧値 gpu 3.05 / io 4.48 は [qwen35moe/27 §2](../qwen35moe/27-PHASE6-THROUGHPUT.md))。
**判定**: M6 で io > gpu のままなら Qwen に tensor ops を持ち込まない (G5 系の io 仕事が先)。io < gpu なら G3 の写しを計画する。
書く先: 15-G6-QWEN-PREFILL-M6.md。測り終えたらモデルを M6 から消す。
