# 32. MLX の `affine_qmm_t_nax` を写して射影と shared MLP の既定に — 台帳 3.496 → 3.134 s (実測、2026-09-24)

[31](31-MLX-KERNELS-PER-STAGE.md) で決めた 1 段目。MLX 0.32.2 (MIT) の `affine_qmm_t_nax` を
`mpp_prefill_affine_nax_f16` (`Sources/Tsugumi/Metal/TensorCore/tensorops.metal`) に写し、候補 `TF_PREFILL_QMM=nax` として測った。
検査を通り、交互の壁時計が縮んだので既定に入れた ([04 原則 2](04-GATES.md))。戻すのは `TF_PREFILL_QMM=mpp64` (tile64)。

証拠 (M6、この順に取った):
[`nax-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-nax-kernel-check/)、
[`b2q-profile-nax`](../../bench/m6/results/2026-09-24-Mac18-5-b2q-profile-nax/) (展開前)、
[`dense-qmm-bench-nax0`](../../bench/m6/results/2026-09-24-Mac18-5-dense-qmm-bench-nax0/) (展開前)、
[`dense-qmm-bench-nax1`](../../bench/m6/results/2026-09-24-Mac18-5-dense-qmm-bench-nax1/)、
[`nax1-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-nax1-kernel-check/)、
[`b2q-profile-nax1`](../../bench/m6/results/2026-09-24-Mac18-5-b2q-profile-nax1/)、
`b2q-default-a` → `b2q-nax-a` → `b2q-default-b` → `b2q-nax-b` (各 n = 3)、
[`b2q-greedy-default`](../../bench/m6/results/2026-09-24-Mac18-5-b2q-greedy-default/)・[`b2q-greedy-nax`](../../bench/m6/results/2026-09-24-Mac18-5-b2q-greedy-nax/)、
[`ledger-qmm-nax-default`](../../bench/m6/results/2026-09-24-Mac18-5-ledger-qmm-nax-default/) (n = 6)、
[`qmm-nax-default-kernel-check`](../../bench/m6/results/2026-09-24-Mac18-5-qmm-nax-default-kernel-check/)。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 883df28 / minos 15.0
(交互の 4 本と greedy は 21c7b0e・da5081c・cf769bd。どれも証拠だけの commit で、ソースは 845da52 と同じ)
```

コマンドは台帳の固定形 (`prefill_bench.sh`)。単体ベンチは `TsugumiKernelCheck --group-size 32 --dense-qmm-bench` (`bench/m6/dense_qmm_bench.sh`)。
全 run exit 0、Swapouts 0。MBP の `Scripts/test.sh` は 883df28 で 1627 本通過。

**台帳** ([06](06-LEDGER.md)): 既定を変えた。`ledger-qmm-nax-default` は **3.496 → 3.134 s (3.116〜4.397)、差 −0.362 s**。

## 0. 結論

1. **台帳 §1 は 3.496 → 3.134 s。**同じバイナリで既定と交互に取った壁時計 (c) は 3.403 → 3.143 s (各 n = 6、差 −0.260 s)。台帳の行は候補の範囲 (3.119〜4.436 s) に入っている (B3)。
2. **写したカーネル単体は、M 2048 の射影と shared の形で 11.2〜19.1 TFLOP/s。**tile64 は同じ形で 8.5〜9.6 TFLOP/s (§1.1)。MLX の同じ形は 16.4〜17.5 ([31 §1](31-MLX-KERNELS-PER-STAGE.md))。
3. **B2 の閾値 (level 1 の `shared` 0.25 s 以下) には届かず、不合格。**`shared` は 0.570 → 0.417 s。`shared` の段には GEMM 以外の分が約 0.27 s 乗っている (§2)。
4. 正しさ: KernelCheck の nax の検査 42 件を含め、224 件がすべて PASS。greedy 64 トークンの本文は既定・候補とも `c0707c48e5` で、これまでの既定と同じ。

## 1. 数字

### 1.1 単体 (`--dense-qmm-bench`、M 2048、g32 sym、20 回の中央値、GPU µs)

| 形 (N × K) | tile64 | nax 展開前 | nax |
| --- | ---: | ---: | ---: |
| shared.gate 2112 × 2816 | 2793.7 | 9430.6 | 1425.7 |
| shared.down 2816 × 2112 | 2548.5 | 9216.1 | 1274.7 |
| q.swa 4096 × 2816 | 5339.8 | 18014.1 | 2731.4 |
| k.swa 2048 × 2816 | 2693.3 | 9067.5 | 1365.1 |
| o.swa 2816 × 4096 | 5394.0 | 17835.3 | 3615.8 |
| q.full 8192 × 2816 | 10637.6 | 35097.8 | 5470.4 |
| o.full 2816 × 8192 | 11153.0 | 38419.0 | 8468.5 |

tile64 の列は `-nax1` の値 (`-nax0` とは 1% 以内)。**展開前**の版は、フラグメントの配列を実行時の添字で引いていた。MLX と同じくループを `unroll(full)` にし、K の 32 刻みだけ展開しないようにした版が **nax** の列 (845da52)。変えたのはこれだけ。

### 1.2 段 (level 1、run 順)

| | 既定 ([27](27-B1-FLOOR-TABLE-ON-SWA-MPP.md)、a122dd3) | nax 展開前 (72e8108) | nax (845da52) |
| --- | --- | --- | --- |
| `attn` (qkv・oproj を含む) | 1.656、1.662、1.655 | 3.576、3.580、3.575 | 1.364、1.365、1.360 |
| `shared` | 0.570 × 3 | 1.309、1.308、1.311 | 0.417、0.419、0.417 |
| `moe` | 1.174、1.098、1.098 | 1.088、1.077、1.077 | 1.112、1.154、1.112 |

### 1.3 壁時計 (c)、既定と交互 (run 順)

| 組 | 既定 | 候補 `TF_PREFILL_QMM=nax` |
| --- | --- | --- |
| a | 4.455、3.425、3.381 | 4.436、3.147、3.119 |
| b | 4.473、3.371、3.368 | 3.264、3.138、3.126 |
| 中央値 (n = 6) | 3.403 | 3.143 |

- 候補 b の 1 本目は 3.264 s で、他の組の 1 本目 (4.4 s 台、layerVerify 9.4 s 台) より短い (layerVerify 6.109 s)。[22](22-GPU-CANDIDATES-GREEDY-AND-FIRST-RUN.md) の性質で、直前の run からの間が短いと起きる (未確認)。
- 台帳 `ledger-qmm-nax-default` (883df28、既定を変えた後): 4.397、3.170、3.125、3.131、3.116、3.137 s。

## 2. 分かったこと / 外れたこと

- **写したカーネルの速さは、ループの展開で決まっていた。**展開前は 2.5〜2.7 TFLOP/s で、tile64 より遅かった。
- `shared` の段 0.417 s のうち、GEMM 3 本は単体の値から 30 層で約 0.15 s (導出: (1.426 × 2 + 1.275) ms × 2478 / 2048 × 30)。残りの約 0.27 s は tile64 の既定でも同じ量が乗っている。中身は `gelu_mul`、post-norm、別の command buffer (`sharedCB`) を attention 側と並行に投入していることの重なりで、分けて測っていない。
- よって [31](31-MLX-KERNELS-PER-STAGE.md) で並べた MLX の `shared` 0.159 s (GEMM だけ) と Tsugumi の `shared` 段は中身が揃っていなかった。GEMM どうしでは、写した後の差はほぼ無い (導出)。
- `o` の射影 (K 4096・8192) だけ 11〜13 TFLOP/s で、他の形 (17〜19) より遅い。

## 3. 次に渡すもの

- 到達目標 (mlx_lm 1.823 s) に対して、台帳は 3.134 s。
- [31](31-MLX-KERNELS-PER-STAGE.md) の残りの差: `post.router` 0.342 → `attn.full` 0.194 → `rope` 0.157 → `moe.down` 0.103 → `moe.gateup` 0.063 s。段の秒は、この既定の上で取り直す。
- `shared` 段の GEMM 以外の 0.27 s は、front の command buffer との重なりだった ([33 §1.3](33-ROUTER-GEMM-SHARED-LATE.md))。
