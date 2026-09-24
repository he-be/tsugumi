# 40. routed MoE の GEMM を MLX 写しの 16×16 断片に — 台帳 2.267 → 1.914 s (実測、2026-09-25)

[38 §3](38-ROUTER-NAX.md) の段の表では、GPU の約 6 割が `moe.gateup` (0.746 s、下限 0.253) と `moe.down` (0.357 s、下限 0.127)。
今の MoE の GEMM (`prefill_moe_gemm_int4_mpp`) は threadgroup テンソル上の `matmul2d` で、単体で 8.2 TFLOP/s ([28 §2](28-MOE-GATEUP-PREFETCH.md))。
密行列の射影は、同じ形から MLX の `affine_qmm_t_nax` の写し (16×16 断片、17 TFLOP/s、[32](32-QMM-NAX-DEFAULT.md)) に替えて速くなっていた。
同じ形を MoE の集めた行に写した (`prefill_moe_gemm_int4_nax`、7ea63bd): 展開した重みのタイルは threadgroup に行幅 BK + 8 で置き、集めた活性の行はデバイスから断片へ直接読み、各 simdgroup が C の 32×32 をレジスタに持つ。
バインド・grid・出力は mpp と同じで、重みの half への丸めと K の順も同じ。**出力は mpp とビット一致する。**
既定にした (0564855)。戻すのは `TF_PREFILL_MOE_NAX=0`。Qwen の既定 (simdgroup) は変わらない。

証拠 (M6、この順に取った。どれも `bench/m6/results/2026-09-25-Mac18-5-<ラベル>/`):
`moe-nax-kernel-check`、`moe-gemm-bench-nax`、`m1-{default,nax}-{a,b}`、`m1-host-nax`、`m1-greedy-nax`、`moe-nax-default-kernel-check`、`ledger-moe-nax`。

```
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 275d0f5 / minos 15.0
(台帳の run は 275d0f5、ソースは 0564855 と同じ。候補の run は 7ea63bd〜dd8a007 で、ソースは 7ea63bd と同じ)
```

コマンドは台帳の固定形 (`prefill_bench.sh`)。greedy は末尾に `--max-new 64 --temperature 0`。単体は `bench/m6/moe_gemm_bench.sh` (`TsugumiKernelCheck --group-size 32 --moe-gemm-bench`、gate/up の 1 dispatch、16 expert)。
全 run exit 0。Swapouts は `m1-host-nax` の 3 本目だけ 252 ページ、他は 0。MBP の `Scripts/test.sh` は 0564855 のソースで 1630 本通過。

**台帳** ([06](06-LEDGER.md)): 既定を変えた。`ledger-moe-nax` **1.914 s (1.887〜2.431)**。前の行 (122e144) の 2.267 s から −0.353 s (導出)。到達目標 (mlx_lm 1.823 s) との差は 0.091 s (導出)。

## 0. 結論

1. **単体 (gate/up、155 行/expert) は 2765 → 1609 µs** (実効 7.1 → 12.2 TFLOP/s)。行数によらず 12〜13 TFLOP/s (64 行以上)。
2. **壁時計 (c) は、同じバイナリの交互で 2.339 → 1.948 s** (各 n = 6)。GPU busy は 1.87 → 1.39 s。
3. 正しさ: KernelCheck の `nax vs mpp` は 811008 要素すべて一致、全ケース PASS (group 64 の 153 件、group 32 の 183 件)。greedy 64 トークンの本文は 2 本とも `c0707c48e5`。

## 1. 数字

### 1.1 単体 (`moe-gemm-bench-nax`、µs、batched 列)

| 行 / expert | mpp | mpp-pf | nax | nax の実効 TFLOP/s (導出) |
| ---: | ---: | ---: | ---: | ---: |
| 27 | 788.5 | 729.7 | 514.4 | 6.66 |
| 64 | 1042.1 | 781.2 | 669.7 | 12.13 |
| 128 | 2021.9 | 1475.6 | 1271.7 | 12.77 |
| 155 | 2774.4 | 2126.6 | 1618.3 | 12.15 |
| 256 | 3965.2 | 2892.8 | 2484.6 | 13.07 |

### 1.2 壁時計 (c)、既定と交互 (run 順、秒)

| 組 | 既定 (mpp) | 候補 (nax) | 中央値 n = 6 (導出) |
| --- | --- | --- | --- |
| `m1` a / b | 2.529、2.318、2.360 / 2.402、2.299、2.255 | 2.132、1.960、1.915 / 2.037、1.935、1.911 | 2.339 → 1.948 |

- 台帳 `ledger-moe-nax` (275d0f5): 2.431、1.942、1.918、1.887、1.911、1.905 s。

### 1.3 host の内訳 (`m1-host-nax`、2・3 本目、秒)

| GPU busy | GPU idle | idle のうち `fetch` | `wait.front` | `fetch` | `wait.tile` |
| --- | --- | --- | --- | --- | --- |
| 1.389、1.389 | 0.455、0.424 | 0.400、0.372 | 0.690、0.685 | 0.740、0.698 | 0.320、0.332 |

38 の既定 (`r1-host-nax`) では GPU idle 0.311〜0.335 s、そのうち `fetch` 0.254〜0.282 s、`wait.tile` は 0.6〜0.7 s。

## 2. 分かったこと / 外れたこと

- 28 は「gate/up 融合・threadgroup メモリ 2 面・128 のタイル・half の展開は遅いか 3% 以内」で、先読みで 11.2 TFLOP/s までだった。律速はタイルの積の書き方 (threadgroup テンソルの `matmul2d`) で、密行列の 32 と同じだった。
- GPU の空きが 0.31〜0.34 → 0.42〜0.46 s に増えた。MoE が短くなった分、層ごとのエキスパートのページ (SSD) を待つ区間が GPU の前に出た。

## 3. 次に渡すもの

- 台帳は 1.914 s。GPU busy 1.39 s、GPU の空き 0.42〜0.46 s のうち `fetch` の間が 0.37〜0.40 s。
- 空きの中身は、エキスパートのページが SSD から来るのを待つ区間 ([35](35-PREFILL-WITHOUT-VERIFY.md) の residency の同期と同じもの)。MLX の 1.823 s は重みが常駐した暖まったプロセスの値 ([29](29-REFERENCE-RUNTIMES-ON-M6.md)) で、この待ちを持たない。
