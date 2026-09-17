# 03. 混在 10 型の dense カーネル

実測: 2026-09-17、M3 Pro 18 GB、macOS 15。[01](01-FEASIBILITY.md) §5 の 3 段 (§3-1)。
表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ (実測 / 導出 / 未確認)。時間の数字は 1 回の実行の中央値 (各 10 回) で、解釈は書かない。

## 0. 結論

1. **27B の GGUF にある 10 型すべてに、decode 用 (直接 GEMV) と prefill 用 (逆量子化 → MPS sgemm) のカーネルができた** (`Sources/Tsugumi/Metal/Quant/ggml_iq.metal`)。
2. **実物のテンソル 52 本 (型 × 形ごとに 1 本、LM head と token_embd を含む) で、gguf-py の float64 正解との相対誤差は、直接カーネルで 9.2e-8〜7.4e-7、sgemm 経路で 7.0e-8〜7.4e-7** (§2)。
3. 全型を同じ配置 (32 重みの小ブロックをレーンに割る) にそろえ、型ごとの違いは小ブロック 1 個の逆量子化関数だけにした (§1)。

## 1. 形

| 型 (ggml id) | ブロック | 算式の要点 |
| --- | --- | --- |
| Q2_K (10) | 84 B | 16 重みごとに 4 bit の scale と min |
| Q4_K (12) | 144 B | 32 重みごとに 6 bit の scale と min (`get_scale_min_k4`) |
| Q6_K (14) | 210 B | 4 bit + 2 bit、16 重みごとに int8 の scale、−32 |
| IQ2_XXS (16) | 66 B | 8 重みごとに 256 語の表、符号は 128 語の表、4 bit scale |
| IQ2_XS (17) | 74 B | 9 bit 索引 (512 語) + 7 bit 符号索引 |
| IQ3_XXS (18) | 98 B | 4 重みごとに 256 語の表 |
| IQ3_S (21) | 110 B | 9 bit 索引 (512 語)、符号バイト、64 重みごとの 4 bit scale |
| IQ2_S (22) | 82 B | 10 bit 索引 (1,024 語)、符号バイト |
| IQ4_XS (23) | 136 B | 16 値の非線形表、6 bit scale (`scales_h` 2 bit + `scales_l` 4 bit、−32) |
| IQ1_M (29) | 56 B | 11 bit 索引 (2,048 語、int8)、±0.125 の delta、f16 scale は scales の上位 4 bit に分散 |

- 算式は手元の llama.cpp (fe8156f78、**読むだけ**) の `ggml-quants.c` の `dequantize_row_*`、構造体と表は `ggml-common.h`。gguf-py はこの算式を写しているので、検査の正解と同じ出どころになる。
  llama.cpp の Metal カーネルは型ごとにレーンの割り方が違うが、ここでは全型を一つの配置にした。
- 配置: 256 重みのブロックを 32 重みの小ブロック 8 個に切り、レーン l が行の小ブロック l, l+32, l+64, … を受け持つ。1 スレッドグループ 64 スレッド = SIMD グループ 2 × レーン 32、1 グループ 4 行。
  `ggml_q8_0_gemv` (Flash-Next) と同じ呼び方・ディスパッチ ((M+7)/8, T) なので、`GGMLDenseGEMV` の既存の経路にそのまま載る。
- 逆量子化カーネル (`ggml_*_dequant_f32`) は `ggml_q8_0_dequant_f32` と同じ形で、T ≥ `mpsMinTokens` (既定 32) の prefill で使う。
  `mpsMaxWeights` (既定 64M 重み) を超えるテンソルは直接カーネルに残る。27B の FFN (17,408 × 5,120 = 89M) はこの既定を超えるので、ランナーで値を決める (**未決**)。
- 表 9 本は ggml-common.h から生成して写した。`Scripts/qwen38_27b/check_ggml_iq_tables.py` で一致を確かめられる (9 本とも一致)。
- `GGUFFile.GGMLType` に 7 型 (Q6_K・IQ2_XS・IQ3_XXS・IQ3_S・IQ2_S・IQ4_XS・IQ1_M) のブロック長を足した。
- IQ4_XS は、この版の llama.cpp では `scales_h` / `scales_l` の形。gguf-py も同じ解釈で、02 の CPU 参照器 (NLL 0.900) はこの解釈で逆量子化している。

## 2. 正しさ (実測、`TsugumiKernelCheck --q27-dense`)

`Scripts/qwen38_27b/dense_kernel_fixture.py` が、型ごと・(行, 列) の形ごとに最初のテンソルを 1 本選び、乱数の x [3, n] と、gguf-py で逆量子化した W との積 (float64) を書く。
検査は GGUF の no-copy バッファを読み、T = 3 を 1 回のディスパッチで流す。直接カーネルと sgemm 経路 (`mpsMinTokens` = 1、上限 128M 重み) の両方で比べる。

| 型 | 形の数 | 相対誤差 (直接) | 相対誤差 (sgemm) |
| --- | ---: | --- | --- |
| IQ3_S | 7 | 9.3e-8〜7.4e-7 | 1.2e-7〜7.4e-7 |
| IQ4_XS | 7 | 1.1e-7〜1.5e-7 | 1.3e-7〜2.3e-7 |
| IQ3_XXS | 7 | 1.1e-7〜2.7e-7 | 1.0e-7〜5.3e-7 |
| Q4_K | 8 | 2.0e-7〜3.6e-7 | 9.8e-8〜1.9e-7 (LM head は直接のみ) |
| IQ2_S | 5 | 9.2e-8〜1.4e-7 | 1.3e-7〜2.6e-7 (token_embd は直接のみ) |
| Q2_K | 5 | 2.2e-7〜4.4e-7 | 1.5e-7〜2.8e-7 |
| IQ2_XS | 3 | 1.1e-7〜1.6e-7 | 1.3e-7〜2.3e-7 |
| Q6_K (MTP の 6 形) | 6 | 1.0e-7〜1.8e-7 | 7.0e-8〜2.5e-7 |
| IQ2_XXS | 3 | 1.2e-7〜1.5e-7 | 1.4e-7〜1.6e-7 |
| IQ1_M | 1 | 1.2e-7 | 1.6e-7 |

閾値は 1e-5 (Flash-Next の `--ggml-dense` と同じ)。全行の出力は `scratch/qwen38_27b/dense-check-v2.txt`。

### 2-1. 1 トークンの GPU 時間 (実測、参考)

直接カーネル、T = 1、各 10 回の中央値。

| 形 | 時間 |
| --- | ---: |
| LM head (Q4_K、248,320 × 5,120) | 5.49 ms |
| ffn_up / ffn_gate (17,408 × 5,120) | 0.29〜1.07 ms |
| ffn_down (5,120 × 17,408) | 0.30〜0.68 ms |
| attn_qkv (10,240 × 5,120) | 0.18〜0.42 ms |

逆量子化関数の形に作り直す前後で、52 形の合計は 25.03 → 25.40 ms (LM head と token_embd を除くと 15.87 → 16.06 ms)。どちらも 1 回の実行。
1 層あたりの合計や tok/s の見込みは、ランナーで実際の並びを測るまで出さない。

## 3. 再現

```bash
~/LLM/venv/bin/python Scripts/qwen38_27b/dense_kernel_fixture.py \
    --types IQ3_S,Q2_K,Q4_K,Q6_K,IQ2_XXS,IQ2_XS,IQ2_S,IQ3_XXS,IQ1_M,IQ4_XS \
    --out scratch/qwen38_27b/dense-fixture          # 数分、常駐は数百 MB
swift build -c release --product TsugumiKernelCheck
.build/release/TsugumiKernelCheck --q27-dense scratch/qwen38_27b/dense-fixture
python3 Scripts/qwen38_27b/check_ggml_iq_tables.py
```

## 4. 次

01 §5 の 4 段: ランナー (`Qwen38Runner` から hc・PLE・indexer・MoE を外し、`post_attention_norm` と dense SwiGLU を足す) を 02 の参照と全位置で突き合わせる。
GDN の出力ノルムのゲートが SiLU であること (02 §2) を、既存の Metal カーネルが sigmoid 固定かどうかから確かめる。
