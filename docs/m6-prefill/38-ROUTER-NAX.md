# 38. prefill の router を tensor の 16×16 断片に — 台帳 2.482 → 2.267 s (実測、2026-09-25)

[37 §3](37-SWA-ATTN-UNFUSED.md) の続き。37 の既定の上で段の秒を取り直すと (`u1-profile`・`u1-profile-detail`、`floor_table.py`)、
差の大きい順は `moe.gateup` 0.746 s (下限 0.253) → `moe.down` 0.357 (0.127) → `post.router` 0.169 → `shared` 0.215 → `rope` 0.101 だった。
MLX との差 ([31](31-MLX-KERNELS-PER-STAGE.md)) が大きい `post.router` (MLX 0.014 s) に手を入れた。

router の GEMM ([33](33-ROUTER-GEMM-SHARED-LATE.md)、simdgroup の float 8×8) を、36 で写した MLX の 16×16 断片と 16×32×16 の `matmul2d` に置き換えた (5122f36)。
BF16 の重みは half に正確に入る。入力 x · effective_scale は f32 の積を half の hi と lo (= 積 − hi) の 2 本にして、1 歩で 2 回積む。logit は f32 版とほぼ同じ精度で、和の順序だけが違う。top-k の選び方は 33 と同じ関数。
戻すのは `TF_PREFILL_ROUTER=gemm` (33 の GEMM) か `=row`。

証拠 (M6、この順に取った。どれも `bench/m6/results/2026-09-25-Mac18-5-<ラベル>/`):
`u1-profile`・`u1-profile-detail` (37 の既定)、`router-nax-kernel-check`、`router-bench-nax`、`r1-{default,nax}-{a,b}`、`r1-host-nax`、`r1-greedy-nax`、`router-nax-default-kernel-check`、`ledger-router-nax`。

```
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 122e144 / minos 15.0
(台帳の run は 122e144、ソースは 5122f36 と同じ。候補の run は 754d8a7〜63438f0 で、ソースは 754d8a7 と同じ)
```

コマンドは台帳の固定形 (`prefill_bench.sh`)。greedy は末尾に `--max-new 64 --temperature 0`。単体は `TsugumiKernelCheck --group-size 32 --router-bench` (`bench/m6/router_bench.sh`)。
全 run exit 0、Swapouts 0。MBP の `Scripts/test.sh` は 5122f36 のソースで 1629 本通過。

**台帳** ([06](06-LEDGER.md)): 既定を変えた。`ledger-router-nax` **2.267 s (2.252〜2.744)**。前の行 (994a0ef) の 2.482 s から −0.215 s (導出)。

## 0. 結論

1. **router 単体 (M 2048、D 2816、128 エキスパート、top-8) は 4646.9 → 345.6 µs** (row 8121.7 µs)。30 層 × 2478 トークンで約 0.17 → 0.013 s (導出)。
2. **壁時計 (c) は、同じバイナリの交互で 2.579 → 2.304 s** (各 n = 6)。GPU busy は 2.04 → 1.87 s、`wait.front` は 0.86 → 0.69 s。
3. **greedy 64 トークンの本文は `c0707c48e5`** (2 本とも)。37 で `b93b5f98b7` に分かれた本文が、36 までと同じ本文に戻った。37 の分岐は、拮抗した候補の入れ替わりという読みと合う。

## 1. 数字

### 1.1 router 単体 (`router-bench-nax`、GPU µs、20 回の中央値)

| カーネル | µs | TFLOP/s (導出) |
| --- | ---: | ---: |
| row (1 トークンに 1 threadgroup) | 8121.7 | 0.18 |
| gemm (33、simdgroup の float 8×8) | 4646.9 | 0.32 |
| nax (38) | 345.6 | 4.27 (hi と lo の 2 回分は数えていない) |

### 1.2 壁時計 (c)、既定と交互 (run 順、秒)

| 組 | 既定 (gemm) | 候補 (nax) | 中央値 n = 6 (導出) |
| --- | --- | --- | --- |
| `r1` a / b | 2.676、2.541、2.446 / 2.631、2.617、2.489 | 2.381、2.315、2.258 / 2.396、2.292、2.268 | 2.579 → 2.304 |

- 台帳 `ledger-router-nax` (122e144): 2.744、2.435、2.276、2.252、2.254、2.258 s。
- peak は 1.30〜1.31 → 1.26〜1.27 GB。理由は調べていない。

### 1.3 host の内訳 (`r1-host-nax`、2・3 本目、秒)

| GPU busy | GPU idle | idle のうち `fetch` | `wait.front` | `fetch` |
| --- | --- | --- | --- | --- |
| 1.871、1.870 | 0.335、0.311 | 0.282、0.254 | 0.686、0.692 | 0.772、0.717 |

37 の既定 (`s1-host-unfused`) では GPU busy 2.041〜2.043 s、`wait.front` 0.856〜0.857 s。

## 2. 分かったこと / 外れたこと

- 33 では router の GEMM 化 (8.13 → 4.65 ms) が壁時計に届かなかった。今回は 4.3 ms 縮めて、GPU busy 0.17 s と壁時計 0.28 s が縮んだ。33 のときは shared expert の command buffer が並行に走っていて、router の区間の終わりを決めていなかった。今は shared を front の後ろに回している (33 の既定) ので、router は front の鎖の上にある。
- 壁時計の縮み (0.28 s) が GPU busy の縮み (0.17 s) より大きい理由は調べていない。
- 正しさ: KernelCheck の router の検査 (131・32・7 行、行の途中で切れるタイルと詰め物のある stride) で、nax は top-8 の選択が CPU 参照と全行で一致し、重みの rel は gemm と同じ程度 (3.7〜4.9e-4)。group 64 の 149 件・group 32 の 179 件がみな PASS。

## 3. 次に渡すもの

- 台帳は 2.267 s、到達目標 (mlx_lm 1.823 s) との差は 0.44 s (導出)。GPU busy は 1.87 s、GPU の空きは 0.31〜0.34 s (大半は `fetch` の間、SSD からのページ)。
- 37 の既定の上の段の差の残り: `moe.gateup` 0.49 s・`moe.down` 0.23 s (下限との差)、`rope` 0.10 s (MLX 0.023 s)、`shared` 0.215 s。
