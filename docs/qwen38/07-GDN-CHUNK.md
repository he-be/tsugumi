# 07. GDN step をチャンク (WY) 形に — 1 層 170 → 40 ms、8K チャンク 4096 で 86.5 → 79.9 s

実測: 2026-09-14、M3 Pro 18 GB、macOS 15。対象は [06](06-ROUTED-GEMM.md) と同じ GGUF・ランナー・プロンプト。
表記は 01 と同じ (実測 / 導出 / 未確認)。**ランナーの速度の行はどれも n=1 で、数字だけを書く。**1 層ベンチは 3〜5 回の中央値。

## 0. 結論

1. **T ≥ 16 のバッチは、GDN の状態更新を「トークンごと」でなく「32 トークンのチャンクごと」に回す** (`Qwen38GDNChunk`、`q38_wy_*`)。
   チャンク内のトークン間の結合は L×L の単位下三角行列の逆行列と sgemm で、value head 48 本をまとめて流す。
   T < 16 (decode を含む) は従来の `q38_gdn_step`。`Q38_GDN_CHUNK=0` で全部を従来経路に、`Q38_GDN_MIN_T` で切り替え点。
2. **1 層 (乱数入力、T = 4096)**: 170〜179 ms → **GPU 28.8〜29.9 ms + encode 9.8 ms** (§2)。T = 16 でも 0.9 → 0.5 ms で勝つ。
3. **正しさ**: 1 層で出力の相対差 5.9e-7・状態 8.6e-7。05 §6-2 の 11 本と 06 §8-2 の 4 本は全部 PASS、
   GDN をチャンク形に強制した France decode (T = 1) 9.24e-7・France チャンク 4 で 8.19e-7・Fuji チャンク 53 を GDN チャンク 5 (端数あり) で 2.24e-6 (§3)。
4. **速度** (8,192 トークン、§4):

   | | チャンク 2048 | チャンク 4096 |
   | --- | ---: | ---: |
   | 06 | 117.7 s (69.6 tok/s) | 86.5 s (94.7 tok/s) |
   | 07 | **107.4 s (76.3 tok/s)** | **79.9 s (102.6 tok/s)** |
   | gdn_step GPU / チャンク | (06 は未記録) → 0.60 s | 6.3 → 1.20 s |
   | peak footprint | 3.70 → 3.72 GB | 5.59 → 5.62 GB |

5. **`MPSMatrixSolveTriangular` はバッチの 1 枚目しか解かない** (`batchStart` / `batchSize` も無視、macOS 15、§1-2)。head ごとに 48 回解くと 1 層 190 ms のうち 165 ms がそこで、速くならなかった。
   逆行列を列ごとに自前のカーネルで作る形 (`q38_wy_tinv`) にして解決した。

## 1. 形

### 1-1. 式

value head ごと、状態 S (Dv×Dk = 128×128)、減衰 g_t、β_t。従来の漸化式は `S ← g S; u = β (v − S k); S += u kᵀ; o = S q`。
チャンク (L トークン、先頭の状態 S0) の中で r(t, s) = g_{s+1}⋯g_t、γ_t = r(t, −1) と置くと:

- u_t + Σ_{s<t} β_t r(t, s) (k_s·k_t) u_s = β_t (v_t − γ_t S0 k_t) → **(I + A) U = R**
- o_t = γ_t S0 q_t + Σ_{s≤t} r(t, s) (k_s·q_t) u_s → **O = γ Y + M U**
- S_L = γ_{L−1} S0 + Σ_s r(L−1, s) u_s k_sᵀ

r は対数の累積和でなく、スレッド (t, hv) の中で s を t から 0 へ下りながら掛けていく (γ が 0 に落ちても割り算が出ない)。

1 チャンクの流れ (`Qwen38GDNChunk.encode`、どれも [hv] 48 枚のバッチ):

1. `q38_wy_gather`: conv から kq = (k_0..k_{L−1}, q_0..q_{L−1}) と k を集める (value head j は key head j % 16)。
2. sgemm: G = kq kᵀ ([2L][L])、XY = kq S0ᵀ ([2L][128])。
3. 1 つのエンコーダで `q38_wy_tri` (A・M・γ・w)、`q38_wy_rhs` (R)、`q38_wy_tinv` (X = (I + A)⁻¹)。
4. sgemm: U = X R、MU = M U。
5. 1 つのエンコーダで `q38_wy_out` (o = MU + γ Y)、`q38_wy_kw` (k の行に w_s = r(L−1, s) を掛ける)、`q38_wy_decay_state` (S0 *= γ_{L−1})。
6. sgemm (左転置、β = 1): S += Uᵀ K_w。

一時領域はチャンク L ぶんだけ (L = 32 で 5 MB 未満、導出)。

### 1-2. 三角ソルブ

最初は `MPSMatrixSolveTriangular` をバッチ記述子 (`matrices: 48`) で呼んだ。**head 0 だけ 1e-8 で合い、残り 47 本はずれた。**
3 枚のバッチの小さい単体テスト (`scratch/qwen38/mpsbatch`) で、同じ記述子の sgemm (転置あり・β = 1 を含む) は 3 枚とも合い、ソルブだけ 2 枚目以降が合わない。
`batchSize = 3` を明示しても、`batchStart = k, batchSize = 1` を 3 回でも同じ。

head ごとのオフセット付き記述子で 48 回解く形は正しいが、T = 4096 で 1 層 GPU 180〜250 ms (段ごとに分けると solve が 164〜169 ms、他は計 26 ms)。
逆行列を列 c ごとのスレッド (c, hv) で作る `q38_wy_tinv` (x[i][c] = −Σ_{s=c..i−1} A[i][s] x[s][c]、1 列はスレッド内の配列に持つ) にして、U = X R を sgemm にした。
L³/6 の手間なので L は小さいほど効く (§2)。L は 512 まで。

## 2. 1 層の A/B (`--q38-wy-bench [T] [チャンク,...] [回数]`)

入力は乱数: conv を正規乱数にして `q38_gdn_qk_norm`、ゲートは A = −[1, 16]・dt bias は Qwen3-Next の初期化の幅で `q38_gdn_gates`、状態の初期値も乱数 (0 でない)。
減衰の分布は T = 4096 で p10 0.23・中央値 0.78・p90 0.97・最大 0.9999。**実モデルのゲートの分布ではない** (正しさは §3 の参照照合で見る)。

T = 4096 (5 回の中央値):

| チャンク L | GPU | encode | 出力の相対差 | 状態の相対差 |
| ---: | ---: | ---: | ---: | ---: |
| (A 逐次) | 169.9 ms | | | |
| 16 | 29.2 | 19.9 | 6.71e-7 | 6.48e-7 |
| 32 | 29.9 | 9.8 | 5.87e-7 | 8.64e-7 |
| 48 | 36.1 | 6.6 | 5.87e-7 | 5.76e-7 |
| 64 | 38.7 | 4.7 | 6.71e-7 | 8.64e-7 |
| 96 | 48.7 | 3.3 | 6.29e-7 | 7.20e-7 |
| 128 | 59.8 | 2.4 | 6.29e-7 | 6.84e-7 |

段ごと (`Q38_WY_SPLIT=1`、段ごとに commit するので合計は上より大きい、2 回):
L = 32 で gather 1.8・gram+XY 6.0・tri+rhs+tinv 8.7・U 1.1・MU 1.1・out+kw+decay 7.9・state 4.4 ms。
L = 128 では tri+rhs+tinv が 37 ms、L = 256 で 126〜129 ms。

小さい T (7 回の中央値、チャンク 32): T = 16 で A 0.9 / B 0.4 + 0.1 ms、T = 32 で 1.7 / 0.7 + 0.1、T = 64 で 1.7 / 0.7 + 0.2、T = 128 で 1.9 / 0.9 + 0.4、T = 1024 で 36.2 / 7.5 + 2.4。
**既定をチャンク 32・T ≥ 16 にした**のはこの 2 つの表から。T < 16 は測っていない。

## 3. 参照との照合

05 §6-2 の 11 本と 06 §8-2 の `Q38_GEMM_MIN_T=1` 4 本 (`scratch/qwen38/checks07.sh`、どれも PASS):

| 検査 | GDN の経路 | 07 | 期待値 (05 / 06) |
| --- | --- | ---: | ---: |
| `--qwen38-decode` France | 従来 | 8.50e-7 | 8.5e-7 |
| `--qwen38-decode` Fuji 予算 16 | 従来 | 6.95e-7 | |
| `--qwen38-prefill` France チャンク 4 | 従来 | 8.50e-7 | 8.5e-7 |
| Fuji チャンク 16 | チャンク形 | 7.91e-7 | 6.95e-7 |
| 同 `Q38_MPS_MIN_T=16 Q38_ATTN_MPS_MIN_T=16` | チャンク形 | 1.46e-6 | 1.78e-6 |
| Fuji チャンク 53 | チャンク形 | 2.36e-6 | 2.26e-6 |
| 同 `Q38_ATTN_MPS_MIN_T=0` | チャンク形 | 2.03e-6 | 2.48e-6 |
| Fuji PLE チャンク 53 | チャンク形 | PASS | PASS |
| `Q38_GEMM_MIN_T=1` France decode | 従来 | 7.42e-7 | 7.42e-7 |
| `Q38_GEMM_MIN_T=1` Fuji チャンク 16 | チャンク形 | 6.59e-7 | 6.06e-7 |
| `Q38_GEMM_MIN_T=1` Fuji チャンク 53 | チャンク形 | 2.52e-6 | 2.11e-6 |
| `Q38_GEMM_MIN_T=1` Fuji PLE | チャンク形 | PASS | PASS |

`--q38-select-check`・`--q2-expert`・`--ggml-dense` も PASS。top-1 のずれはどれも 0。

チャンク形を強制したもの:

| 検査 | 結果 |
| --- | ---: |
| `Q38_GDN_MIN_T=1` France decode (T = 1、L = 1) | 9.24e-7 |
| `Q38_GDN_MIN_T=1` France チャンク 4 | 8.19e-7 |
| `Q38_GDN_MIN_T=1 Q38_GDN_CHUNK=5` Fuji チャンク 53 (5 × 10 + 端数 3) | 2.24e-6 |
| `Q38_GDN_CHUNK=0` Fuji チャンク 53 (従来経路に戻す) | 2.26e-6 (05 と同じ) |

## 4. ランナーでの結果 (`Q38_SPLIT_PRE=1 --qwen38-prefill-bench`、8,192 トークン)

チャンク 4096 (`memlog.sh` 越し):

| | 合計 | pre wall / GPU | route (うち読み) | routed wall / GPU | PLE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 06、0..<4096 | 43.4 s | 18.1 / 14.7 s | 7.6 s (5.0) | 12.4 / 11.9 s | 3.4 s |
| 06、4096..<8192 | 43.1 s | 18.6 / 16.4 s | 8.2 s (5.5) | 12.8 / 12.2 s | 3.4 s |
| 07、0..<4096 | 40.8 s | 13.7 / 9.6 s | 7.6 s (5.0) | 12.5 / 11.9 s | 5.1 s |
| 07、4096..<8192 | 39.1 s | 14.4 / 11.3 s | 8.3 s (5.6) | 13.0 / 12.3 s | 3.4 s |

2 チャンク目の pre-router GPU の内訳 (ms): 注意 3,279・GDN 入力 2,630・**gdn_step 1,197** (06 は 6.3 s)・hc_ffn 1,236・gdn (norm_gate + ssm_out) 1,275・hc_attn 1,053・shexp+router 544。
pre の wall と GPU の差は 06 の 2.2 s から 3.1 s に広がった。1 層 encode 9.8 ms × 36 層 = 0.35 s (導出) では全部は説明できない (未確認)。

チャンク 2048: 28.4 / 25.9 / 26.6 / 26.6 s、gdn_step は 1 チャンク 595〜600 ms、合計 107.4 s。

decode (8,192 の直後 1 トークン、従来経路): 0.56 s (チャンク 4096 の走行)、0.43 s (2048)。06 は 0.51 / 0.48 s。

### 4-1. メモリ

| | peak footprint | wired の最大 | file-backed の最小 |
| --- | ---: | ---: | ---: |
| 06、チャンク 4096 | 5.59 GB | 12.49 GB | 2.00 GB |
| 07、チャンク 4096 | 5.62 GB | 12.69 GB | 1.77 GB |
| 06、チャンク 2048 | 3.70 GB | 12.38 GB | 2.48 GB |
| 07、チャンク 2048 | 3.72 GB | 12.04 GB | 2.24 GB |

Swapouts の増分はどれも 0 (`GUARD: Swapouts delta peak 0 pages`、走行の後に `pgrep -x TsugumiKernelCheck` は空)。

## 5. 次

チャンク 4096 の 2 つ目 39.1 s の順:

1. **routed の GPU 12.3 s** (展開 約 4.2 s・sgemm 約 6.8 s、06 §5 の導出)。組の少ない expert だけ従来カーネルに回す混成 (06 §5 の 2)。
2. **route 8.3 s** (読み 5.6 s・top-10 1.3 s・並べ替えと encode)。
3. **pre-router の GPU 11.3 s**: 注意 3.3 s (n とともに伸びる)、GDN 入力 2.6 s、hc 2.3 s、GDN 出力 1.3 s、step 1.2 s。
   step をさらに削るなら、state に依存しない gram / tri / tinv を全チャンク分まとめて 1 回で流す (チャンクあたりのエンコーダが減る)。
4. **PLE 3.4 s**、pre の wall と GPU の差 3.1 s。
5. **32K を 1 回** (チャンク 4096、memlog 越し)。06 §5 の 4 のとおり wired が 12 GB 台なので Swapouts を特によく見る。

## 6. 足したもの

| 種類 | パス |
| --- | --- |
| カーネル | `qwen38.metal`: `q38_wy_gather`、`q38_wy_tri`、`q38_wy_tinv`、`q38_wy_rhs`、`q38_wy_out`、`q38_wy_kw`、`q38_wy_decay_state` |
| ランタイム | `Sources/Tsugumi/Runtime/Qwen38/Qwen38GDNChunk.swift` (ベンチとランナーで共有) |
| ランナー | `linear` の step の分岐、`gdnChunk` (`Q38_GDN_CHUNK`、既定 32、0 で従来)、`gdnMinTokens` (`Q38_GDN_MIN_T`、既定 16) |
| 検査 | `--q38-wy-bench [T] [チャンク,...] [回数]` (`Q38WYBench.swift`、`Q38_WY_SPLIT=1` で段ごと) |
| スクリプト | `scratch/qwen38/checks07.sh` (git 管理外、§3 の 19 本) |

## 7. 落とし穴

- **`MPSMatrixSolveTriangular` はバッチを解かない** (§1-2)。バッチ記述子を渡してもエラーにならず、1 枚目だけ正しい。sgemm のバッチは正しい。
  新しい MPS カーネルをバッチで使うときは、**2 枚目以降が合うかを先に 1 本見る** (head 0 だけ合うのが症状)。
- 逆行列の手間は L³/6 なので、L を大きくすると tinv が支配する (L = 256 で 1 層 126 ms)。L を小さくすると encode が増える (L = 16 で 20 ms)。
- `q38_wy_tri` は A・M の上三角を毎チャンク 0 で書き直す (バッファをチャンク間で使い回すため)。
- 状態の減衰 γ_{L−1} は sgemm の β に入れられない (head ごとに違う) ので、別のカーネルで掛けてから β = 1 で足す。

## 8. 再開手順 (新しいセッションで続けるとき)

06 §8 の後継。**06 §7・05 §6-4・03 §6-4 の落とし穴はそのまま有効**なので、先に一度読む。

### 8-1. 状態

- コードはこの文書と同じコミットまで入っている。ランナーは `Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift`、GDN のチャンク形は `Qwen38GDNChunk.swift`。
  - 注意: T ≥ 32 で選択なしは `attentionSgemm`、選択ありは `attentionSelected` (04)、T < 32 は `attentionHost`。
  - expert の読み: T ≥ 32 は `GGUFFile.preadRanges` (4 スレッド)、T < 32 は `F_RDADVISE` (05)。
  - routed expert: T ≥ 1024 は `routedGemm` (06)、それ未満は per-pair カーネル。
  - GDN step: T ≥ 16 は `Qwen38GDNChunk` (チャンク 32)、それ未満は `q38_gdn_step` (07)。
- 速度の現在地 (8,192 トークン、n=1): チャンク 4096 で 79.9 s (102.6 tok/s、footprint 5.62 GB)、チャンク 2048 で 107.4 s (76.3 tok/s、3.72 GB)。
  decode は短文脈 0.09〜0.11 s/トークン、8K の直後 0.43〜0.56 s。
- **32K はまだ流していない。**
- ウェイト・トークナイザの在処は [01 §7-1](01-Q2-FIRST-LIGHT.md)。`G=~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`。
- `scratch/qwen38/` (git 管理外): 06 §8-1 のもの + `checks07.sh`、`mpsbatch/` (§1-2 の単体テスト)。
- 運用点は thinking 無効・32K・エージェント主体・英語主体 (メモリ `qwen38-operating-point`)。

### 8-2. 検査を一通り

```bash
swift build -c release --product TsugumiKernelCheck
scratch/qwen38/checks07.sh > checks.out 2>&1     # 数分、§3 の 19 本
grep -c PASS checks.out; grep -i fail checks.out  # 24 と空
.build/release/TsugumiKernelCheck --q38-wy-bench 4096 32 3   # A 約 170 ms / B GPU 約 29 ms + encode 10 ms / 相対差 1e-6 以下
```

期待値は §3 の表。`checks07.sh` が無ければ、05 §6-2 と 06 §8-2 の各行に §3 の強制 4 本を足して作り直す。

速度 (約 1.5 分と 2 分、memlog 越しに、間に 20 秒):

```bash
scratch/qwen38/memlog.sh mem.log out.txt /usr/bin/time -l env Q38_SPLIT_PRE=1 $B --qwen38-prefill-bench scratch/qwen38/prompt-code.tokens --q38-tokens 8192 --q38-chunk 4096
grep -E "^\s+\[|GPU ms|prefill|decode|footprint|GUARD" out.txt; pgrep -x TsugumiKernelCheck || echo none
```

§4 の表と比べる。

### 8-3. 次にやること

§5 の順 (routed の混成 → route → pre-router の残り → 32K)。

### 8-4. 落とし穴

§7 と、06 §7・05 §6-4・03 §6-4。
