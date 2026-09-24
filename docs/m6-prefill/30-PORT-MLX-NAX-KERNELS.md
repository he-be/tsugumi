# 30. MLX の Neural Accelerator 用カーネルを写す案 — 写すのは GEMM の中身だけ、順序は B2 のまま (提案、2026-09-24)

**新しい実測は無い。**[29](29-REFERENCE-RUNTIMES-ON-M6.md) で同じ M6・同じ本文の mlx_lm が 1.823 s (Tsugumi の GPU busy は 2.97〜3.00 s) だったのを受けて、
MLX のカーネルを Tsugumi に移す案の範囲と順序を書く。着手はユーザーの判断。台帳 §1 は動かない。

## 0. 結論

1. **写す対象は 3 つのヘッダにある GEMM の中身**で、Tsugumi の重み形式 (affine int4、group 32) に対応する変種がコンパイル済みで入っている (§1)。ライセンスは MIT。
2. **写さないのはメモリの持ち方。**MLX は重みを全部常駐させて peak 13.3 GB になり、16 GB ではスワップした ([29 §1.3](29-REFERENCE-RUNTIMES-ON-M6.md))。Tsugumi は expert を流して peak 1.22 GB ([26](26-SWA-ATTN-MPP-DEFAULT.md))。MoE のカーネルは番地付けを Tsugumi の tile stream に合わせて書き直す (§2)。
3. **順序は [04 B2](04-GATES.md) の下限表の順のまま**: `moe.gateup` → attention → `shared`・射影 (§3)。段ごとの閾値は着手前に 04 に置く。
4. 写しても壁時計は 1.8 s には届かない可能性がある。MLX の値には expert の読み込み待ちが入っていない (§4)。

## 1. MLX 側にあるもの (M6 の `~/LLM/ref/mlx-venv/lib/python3.12/site-packages/mlx/`、mlx 0.32.2)

| ファイル (`include/mlx/backend/metal/kernels/`) | 行 | 中身 |
| --- | ---: | --- |
| `steel/gemm/nax.h` | 898 | `mpp::tensor_ops::matmul2d` を包む `NAXTile` と `tile_matmad_nax`。tensor ops の使い方の本体 |
| `steel/gemm/gemm_nax.h` | 141 | 密行列 GEMM の threadgroup ループ |
| `quantized_nax.h` | 1700 | `affine_qmm_n_nax` (密、重みは int4 のまま読んで展開)、`affine_gather_qmm_rhs_nax` (MoE、行を expert ごとに並べてから) |
| `steel/attn/nax.h` | 898 | `attention_nax`、`attention_nax_dsplit` |
| `fp_quantized_nax.h` | 1024 | fp4 / fp8 用。今回は使わない |

`lib/mlx.metallib` のシンボルに `_qmm_n_nax_..._gs_32_b_4_bm64_bn64_bk64_wm2_wn2` と `_gather_qmm_rhs_nax_..._gs_32_b_4_bm_{32,64}_bn_64_bk_64_wm_2_wn_2` がある。
**group 32・4-bit・affine は Tsugumi の QAT int4 と同じ格子**なので、重みの展開部分はそのまま使える。活性の型は MLX が bfloat16 と float、Tsugumi は FP16 (matmul2d はどれも取れる)。

タイルは M 64 (MoE は 32 も) × N 64 × K 64、simdgroup 2 × 2。Tsugumi の mpp 経路は [28 §2](28-MOE-GATEUP-PREFETCH.md) で 64 × 64 と 32 × 64 を試している。
MLX が同じ形で単体 11 TFLOP/s を超えているかは、写してから `--moe-gemm-bench` で測る (今は分からない)。

## 2. 写すもの / 写さないもの

| | 写す | 写さない |
| --- | --- | --- |
| GEMM の中身 | `NAXTile` の読み込み、int4 → 活性型の展開、`matmul2d` の積み方、epilogue | |
| MoE の番地付け | | MLX は常駐した全 expert への index (`gather`)。Tsugumi は chunk ごとに集めた tile stream のバッファを読む。番地付けは Tsugumi のまま |
| メモリ | | 重みの常駐、`mx.array` の割り当て |
| attention | S = QKᵀ と P·V の `matmul2d` の使い方、softmax の分割 | KV cache の持ち方 (Tsugumi は FP16 の ring) |
| 活性の型 | | bf16 には替えない。FP16 のまま |

写した結果は mpp とビット一致しない (積の順と丸めの順が変わる)。検査は KernelCheck の CPU 参照との一致と greedy の記録で、差は記録して止めない ([04 原則 5](04-GATES.md))。

## 3. 順序と閾値の置き場

| 段 | 下限との差 s ([27 §1.1](27-B1-FLOOR-TABLE-ON-SWA-MPP.md)) | 写す元 | 閾値 |
| --- | ---: | --- | --- |
| 1. `moe.gateup` (と `moe.down`) | 0.487 (+0.220) | `affine_gather_qmm_rhs_nax` | 着手前に 04 B2 に置く。28 の閾値 (level 2 中央値 0.50 s 以下) は候補が届かなかったので、置き直す |
| 2. `attn.full`・`attn.swa` | 0.240 + 0.214 | `attention_nax` | 同上 |
| 3. `shared`・`qkv`・`oproj` | 0.427 + 0.165 + 0.103 | `affine_qmm_n_nax` | 同上 |

1 段ずつ。段の秒 (level 2)、KernelCheck、greedy、壁時計 (c) を既定と交互に各 n = 6。縮んだら既定に入れて `ledger-*` を取る ([04 B3](04-GATES.md))。
28 の変種 (先読み・M32・融合) は、MLX が同じ問題をどう解いているかを読んでから残すか決める。

## 4. 期待の上限 (導出、目標値ではない)

- GPU 段の実測の合計と下限の合計の差は約 2 s ([27 §1.1](27-B1-FLOOR-TABLE-ON-SWA-MPP.md))。mlx_lm が実際に詰めているのはそのうち約 1.2 s 分 ([29 §2](29-REFERENCE-RUNTIMES-ON-M6.md))。
- Tsugumi の壁時計には `wait.tile` 0.85〜0.87 s ([27 §1.2](27-B1-FLOOR-TABLE-ON-SWA-MPP.md)) が残る。MLX には無い成分なので、カーネルを写しても差は残る。これは [B0](04-GATES.md) の (b) の口ができれば分けて測れる。
- この文書は目標の秒を置かない。置くと分母になる ([18](18-RESET-ABSOLUTE-FLOOR.md))。

## 5. 判断を仰ぐこと

- 着手するか。するなら 1 段目 (`moe.gateup`) から。
- MLX のヘッダを repo に写す形 (必要な関数だけ Metal に書き写し、出所とライセンスを先頭に書く) でよいか。
