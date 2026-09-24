# 31. MLX のカーネルを Tsugumi と同じ形で段ごとに測った — 差は `shared`・`post.router`・`attn.full`・`rope`・`qkv` にあり、`moe.gateup` にはほぼ無い (実測、2026-09-24)

**校正の測定。台帳 §1 は動かない。**Tsugumi のコードは変えていない。
[29](29-REFERENCE-RUNTIMES-ON-M6.md) の mlx_lm 1.823 s が段ごとにどこで Tsugumi より短いかを、MLX のカーネルを Tsugumi の重み形式・形で 1 段ずつ回して取った。
[30](30-PORT-MLX-NAX-KERNELS.md) の順序 (`moe.gateup` から) はこの結果で置き換える (§3)。

**到達目標 (ユーザー指定、2026-09-24)**: 同じ M6・同じ本文で mlx_lm の 1.823 s (約 1360 tok/s)。
MLX の値は暖まったプロセスの prompt 処理なので、並べる相手は [04 B0](04-GATES.md) の (b)。(b) の口ができるまでは GPU busy (2.97〜3.00 s、[27 §1.2](27-B1-FLOOR-TABLE-ON-SWA-MPP.md)) と並べる。

証拠 (M6、この順に取った):
[`ref-mlx-ops-fp16`](../../bench/m6/results/2026-09-24-Mac18-5-ref-mlx-ops-fp16/) (n = 3)、
[`ref-mlx-ops-bf16`](../../bench/m6/results/2026-09-24-Mac18-5-ref-mlx-ops-bf16/) (n = 3)。スクリプトは `bench/m6/ref_mlx_ops.sh` / `ref_mlx_ops.py` (bb7ad75)。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit bb7ad75
mlx 0.32.2
```

```
/Users/mh/LLM/ref/mlx-venv/bin/python bench/m6/ref_mlx_ops.py 5 float16   (bf16 は最後の引数が bfloat16)
```

各 run は新しいプロセスで、中で 1 回暖めてから 5 回回した中央値を出す。6 run すべて exit 0、Swapouts 0、MLX の peak 4.17 GB。

**測り方**: 重みは 1 層分を乱数で作る。形式は Tsugumi と同じ affine int4・group 32 (`mx.quantize`)、活性は fp16。
T = 2478 を台帳のチャンク (2048 + 430) に分け、sliding 25 層・full 5 層分の呼び出しを 1 回ずつ eval して同期し、その秒を足した。
呼び出しは mlx_lm の `gemma4_text` と同じ形にした。gate と up は別々の `gather_qmm` (`sorted_indices=True`) で、argsort による並べ替えが前後に付く。
`qkv` だけは Tsugumi と同じく 3 本を連結した 1 本の `quantized_matmul` にした。

**Tsugumi 側との違い (逸脱)**:

1. routing は一様乱数の top-8 で、モデルの実際の偏りは入っていない。
2. MLX 側は呼び出しごとの壁時計 (起動の分を含む)、Tsugumi 側は GPU の段の秒 ([27 §1.1](27-B1-FLOOR-TABLE-ON-SWA-MPP.md)、profile) で、量が違う。
3. `moe.gelu` は MLX 側が融合していない式で、mlx_lm の `mx.compile` 版より長い。
4. `moe.sort`・`moe.unsort` は MLX にしか無い段。

## 0. 結論

1. **MLX が同じ形で Tsugumi より短い段の差の合計は約 1.51 s (導出)。**大きい順に `shared` 0.411 → `post.router` 0.342 → `attn.full` 0.194 → `rope` 0.157 → `qkv` 0.153 → `moe.down` 0.103 → `oproj` 0.088 → `moe.gateup` 0.063 s。
2. **`moe.gateup` は MLX でも 0.678 s (6.8 TFLOP/s、導出)。**Tsugumi は 0.741 s で、差は 0.063 s。30 の 1 段目 (`affine_gather_qmm_rhs_nax` を写す) で縮む余地は、この条件では 0.06 s しかない。
3. **`shared`・`qkv`・`oproj` は MLX の同じカーネル (`affine_qmm_t_nax`) で 16.4〜17.5 TFLOP/s (導出)。**3 段合わせて Tsugumi より 0.652 s 短い。最初に写すのはこのカーネルにする (§3)。
4. fp16 と bf16 で段の秒は変わらない (差は `moe.gateup` の 0.01 s 以内、`moe.gelu` 0.006 s)。

## 1. 数字 (秒、MLX は 3 run の中央値、run ごとの値は `summary.tsv`)

| 段 | Tsugumi 既定 ([27 §1.1](27-B1-FLOOR-TABLE-ON-SWA-MPP.md)) | MLX fp16 | MLX bf16 | 差 (Tsugumi − MLX fp16、**導出**) | MLX の実効 TFLOP/s (**導出**) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shared` | 0.570 | 0.159 | 0.159 | 0.411 | 16.7 |
| `post.router` | 0.356 | 0.014 | 0.014 | 0.342 | 4.0 |
| `attn.full` | 0.267 | 0.073 | 0.073 | 0.194 | 6.9 |
| `rope` | 0.180 | 0.023 | 0.023 | 0.157 | — |
| `qkv` | 0.354 | 0.201 | 0.201 | 0.153 | 17.5 |
| `moe.down` | 0.346 | 0.243 | 0.243 | 0.103 | 9.7 |
| `oproj` | 0.210 | 0.122 | 0.122 | 0.088 | 16.4 |
| `moe.gateup` | 0.741 | 0.678 | 0.685 | 0.063 | 6.8 |
| `attn.swa` | 0.258 | 0.285 | 0.285 | −0.027 | 2.9 |
| `moe.gelu` | 0.010 | 0.125 | 0.131 | (逸脱 3) | — |
| `moe.sort` + `moe.unsort` | — | 0.133 | 0.133 | (逸脱 4) | — |
| 合計 | — | 2.056 | 2.068 | — | — |

- `rope` の MLX 側は q と、このチャンクの k だけ (Tsugumi の `rope` 段の中身とは完全には揃えていない)。
- `shared` の MLX 側は GEMM 3 本だけ。Tsugumi の `shared` 段には GEMM 以外 (`gelu_mul`、post-norm、並行に投入した command buffer の重なり) が約 0.27 s 乗っていて、中身は揃っていない ([32 §2](32-QMM-NAX-DEFAULT.md))。
- `post.router` の MLX 側は fp16 の密 GEMM (`x @ Wᵀ`)。Tsugumi の router はトークンごとの bf16 GEMV ([04 B2](04-GATES.md) の B1 前の候補)。
- run 間の幅は fp16 で各段 0.003 s 以内。

## 2. 分かったこと / 外れたこと

- 30 の見立て「`moe.gateup` が最大の差で、MLX の gather 版を写せば縮む」は、この条件では外れた。MLX の gather 版も 6.8 TFLOP/s で、Tsugumi の 6.4 と同じ程度。
- 30 は密行列のカーネルを `affine_qmm_n_nax` と書いたが、重みが [out, in] の並び (`transpose=True`、Tsugumi と同じ) で MLX が使うのは `affine_qmm_t_nax`。
- mlx_lm の 1.823 s と、この段の合計 2.056 s の差は、融合 (`moe.gelu` の compile)、sort / unsort、呼び出しごとの同期の分と読める (未確認)。

## 3. 次に渡すもの

- 写す順は §1 の差の順に置き直す。1 段目は `affine_qmm_t_nax` を Tsugumi の `shared`・`qkv`・`oproj` に写す (同じカーネルで 3 段、差 0.652 s)。閾値は [04 B2](04-GATES.md) に計測の前に置く。
- `post.router` (差 0.342 s) はトークンごとの GEMV を GEMM にする変更で、MLX から写すものは無い。
- `attn.full` (差 0.194 s) は `steel/attn/nax.h`。`moe.gateup` は実際の routing の偏りで MLX と Tsugumi を並べ直してから決める。
