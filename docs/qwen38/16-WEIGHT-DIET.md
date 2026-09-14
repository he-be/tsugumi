# 16. ウェイトのダイエット: `down` の pad 列を落とすサイドカー (W) と、ルーター類の F32 → BF16 (W-4)

実測: 2026-09-14、M3 Pro 18 GB、macOS 15。GGUF・プロンプトは [10](10-MTP-SHADOW-SPEC.md)〜[14](14-MTP-NMAX2.md) と同じ。手順は [15 §2 W](15-DECODE-PLAN.md)。
表記は 01 と同じ (実測 / 導出 / 未確認)。**速度の反復は 2 以下なので、どのセルも数字だけを書く。**

## 0. 結論

1. **W-1 (詰め直し)**: `ffn_down_exps` (Q2_K、640 列を 768 列に pad) の各行から、3 ブロック目の `scales[8..15]`・`qs[32..63]` を落としたサイドカーを書いた。
   1 行 252 → 212 B で、ファイルは 14.77 → 12.42 GiB。全 48 層で「詰めた行 == 元の行から該当バイトを抜き出したもの」がバイト単位で一致した。
   第 24 層の 4 expert で、gguf-py で元の行を逆量子化した列 0〜639 と、212 B の行からの逆量子化が完全に一致した。元の行の pad 列 640〜767 は、逆量子化すると全部 0。
2. **W-2 (カーネル)**: Q2_K の最後のブロックを半分の型 `block_q2_K_half` (44 B) として読む形を、per-pair の phase 2 と逆量子化の両方に入れた。
   第 24 層の実 route で T = 1 / 2 / 32 / 1024 / 4096 を流し、元の GGUF とサイドカーの出力がどちらの形でもビット一致した。
   変更前のカーネルの出力ともビット一致したので、既定の経路の出力も変わっていない。
3. **W-3 (ランナー)**: 終わった 15 組で、2 腕 (既定 / W) のトークン列がすべて一致した。steady tok/s の比 (W / 既定) は次のとおり。
   - 短文脈: MTP off 0.966〜1.106、spec 1.061〜1.125
   - 12K long: off 0.980 (1 組)、spec 1.106 / 1.115
   - 12K の spec の wired の最大は 14.71 → 14.39 GB と 15.03 → 14.63 GB
4. **W-4 (F32 → BF16)**: GGUF の F32 テンソル 472 本のうち、BF16 でビット単位に表せないのは `ssm_a` の 1,728 要素だけだった。
   `ssm_a` は変換器が −exp(A_log) を計算した値で、BF16 の A_log から float32 の exp で全要素が再現できる。
   GEMV で読む `ffn_gate_inp`・`ffn_gate_inp_shexp`・`ssm_alpha`・`ssm_beta` の 170 本 (279 MiB) を BF16 のサイドカー (140 MiB) にし、BF16 の GEMV・逆量子化カーネルを足した。
   170 本 × T = 1 / 3 / 32 / 64 の 680 件で、F32 と BF16 の出力がビット一致した。
5. **両サイドカーを既定にした** (ユーザー判断、「メモリがギリギリなので」)。ファイルがあれば使い、`Q38_DOWN_SIDECAR=0` / `Q38_BF16_SIDECAR=0` で切れる。
   12K long の spec で、無効 → 有効 → 有効 → 無効の 4 本の結果:
   - wired の最大: 無効 15.00 / 14.84 GB、有効 14.28 / 14.57 GB
   - file-backed の最小: 無効 1.07 / 1.08 GB、有効 1.65 / 1.40 GB
   - steady tok/s: 無効 7.07 / 6.98、有効 7.60 / 7.60
   - トークン列は 4 本とも同じで、Swapouts は 0

## 1. W-1: `down` の詰め直し (`Scripts/qwen38/down_sidecar.py`)

Q2_K のブロック (84 B) は `scales[16] @0 | qs[64] @16 | d @80 | dmin @82`。重み w の逆量子化に効くのは `d`・`dmin`・`scales[w / 16]`・`qs[w / 4]` だけ。
3 ブロック目 (列 512〜767) のうち、重み 128〜255 (列 640〜767) は acts が常に 0 である。そこで 3 ブロック目を `scales[0..7] | qs[0..31] | d | dmin` の 44 B に詰めた。
**`scales` 以外は位置が変わる** (`qs` は 16 → 8、`d` は 80 → 40)。

- 置き場は `~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/down/Qwen3.8-Flash-Next-Q2KDown640.gguf`。GGUF v3 の容器で、テンソル `blk.{0..47}.ffn_down_exps.weight` を型 I8・形 [212, 2560, 512] で持つ。
  GGML の Q2_K では 640 列を表せないので、バイト列として持ち、並びはメタデータ `tsugumi.down.{row_bytes, logical_input, physical_input, layout}` に書いた。alignment は 16384。
- blk.48 (MTP) の down は MXFP4 で、640 は 32 で割り切れるので pad が無い。対象外で、元の GGUF から読む。
- 書き出しは層ごとの pread → numpy で抜き出し → write (両方 F_NOCACHE)。14.77 GiB を読んで 12.42 GiB を書き、77.2 s、RSS は最大 1.14 GB、Swapouts +24 ページ。
- 検定 (`verify`) は全層を読み直し、`raw[:, KEEP] == 詰めた行` を見た (48/48 一致、詰めた部分の sha256 は `c56b281a…`)。
  第 24 層の expert 0 / 17 / 255 / 511 では、gguf-py の `dequantize` (元の 252 B) の列 0〜639 と、212 B の並びを直接書いた逆量子化を比べ、最大差 0 だった。

## 2. W-2: カーネル (`moe_ggml.metal`、`--q2-down-sidecar`)

- `block_q2_K_half { uchar scales[8]; uchar qs[32]; half d; half dmin; }` を足した。カーネルは `down_row_bytes < nb × sizeof(block_q2_K)` のとき、最後のブロックをこの型で読む。
  - `moe_q2k_phase2_down_reduce`: 最後のブロックでは、後半のレーン (`iq == 1`、acts が 0 の側) を読まない。`sc` / `qs` / `d` は半分の型から取る。行ごとの歩幅 (`rb`) はそのまま。
  - `moe_q2k_dequant_down_f32`: `row_bytes` を buffer(7) で受ける (いままでは `stride` から 252 を導いていた)。呼び出し側のランナー `routedGemm` と `Q2GemmBench` が渡す。MXFP4 の逆量子化は buffer(7) を読まない。
- 検査 `--q2-down-sidecar <gguf> <sidecar> <route .bin> [tokens] [--q2-down-save dir | --q2-down-against dir]` (`Q2DownSidecarCheck.swift`)。
  1 層の route の先頭 T トークンに、シード固定の x・残差を与え、down のビューを GGUF とサイドカーに差し替えて、per-pair 版と逆量子化 + sgemm 版の両方を流す。`y` はバイト比較。
- 第 24 層 (`scratch/qwen38/routes/c4096-l24.bin`) の結果:

| T | expert | per-pair: GGUF vs サイドカーで違う要素 | 逆量子化 + sgemm: 同 | 変更前カーネルの保存出力 vs (GGUF / サイドカー) | per-pair vs sgemm の最大相対差 |
| ---: | ---: | --- | --- | --- | ---: |
| 1 | 10 | 0 / 2,560 | 0 / 2,560 | 0 / 0 (両形) | 9.99e-08 |
| 2 | 16 | 0 / 5,120 | 0 / 5,120 | 0 / 0 | 1.71e-07 |
| 32 | 73 | 0 / 81,920 | 0 / 81,920 | 0 / 0 | 5.13e-07 |
| 1024 | 322 | 0 / 2,621,440 | 0 / 2,621,440 | 0 / 0 | 5.32e-07 |
| 4096 | 439 | 0 / 10,485,760 | 0 / 10,485,760 | 0 / 0 | 7.32e-07 |

- 陰性対照: 変更前のカーネルにサイドカーを読ませると、どの T でも全要素がずれた。この検査で読み違いを捕まえられる。
- 変更後も既存の `--q2-expert` (gguf-py の参照との相対誤差) は PASS (acts 1.2e-07、y 8.0e-08)。
- 型でのつかみ: Q2_K の行を読む箇所は `block_q2_K` へのポインタ 3 か所 (phase 2・逆量子化・構造体) と `down_row_bytes`・`bytesPerRow` の参照。
  書いた時点で 1 つ踏みかけた。最初は「sub-block 0..7 は同じ位置」と書いて `qs` を元の型から読んでいたのを、ビルド前に直した。

## 3. W-3: ランナー (`Q38_DOWN_SIDECAR`)

- `expertParts(pre)` が、gate / up / down それぞれの (ファイル, テンソル, expert あたりバイト) を返す。down はサイドカーに層があればそちらから取る。
  expert のビュー・advise (個別・層全体)・`Q38_COUNT_MISS` の mincore・先読みの advise は、どれもそのファイルに対して行う。
- prefill の pread は `GGUFFile.preadRanges([(file, offset, bytes)])` にした。2 ファイルでも 1 つのスレッドプールで読むので、無効のときの動きは今までと同じ。
- 逆量子化 + sgemm の `downStride` は、I8 のとき物理幅 768 にした (`rowWidth` は 212 になるため)。
- greedy のスモーク (code、40 トークン、spec): 既定と W で、トークン列と 20 ステップの logit (6 桁) が一致した。

### 3-1. ABBA (`scratch/qwen38/w3.sh`、集計 `scratch/qwen38/agg_w3.py`)

instruct、seed = パス、200 トークン、腕の順はパス 1 が 既定 → W (off → spec)、パス 2 が逆。10 s 間隔。
この時点では W は既定 off で、「既定」の腕はサイドカー無し。kernel→GPU・advise は検証を含む全ステップの中央値 (ms)。

**ドライバの不具合**: `SETS` を環境変数で渡したときに空白で割られ、短文脈の code / tool は prefill がチャンク 2048 (既定値) になった。explain だけが 512。
両腕とも同じ条件なので decode の比較には使えるが、§3 の規則とは違う。12K は途中で気づいて止め、直したドライバで流し直した。

| プロンプト | MTP | パス | 既定 tok/s | W tok/s | W / 既定 | kernel→GPU 既定 / W | advise 既定 / W | トークン/ステップ | トークン列 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| code | off | 1 | 7.14 | 7.13 | 0.999 | 33 / 31 | 16 / 14 | 1 | 一致 |
| code | off | 2 | 6.83 | 6.78 | 0.993 | 35 / 31 | 18 / 16 | 1 | 一致 |
| code | spec | 1 | 9.03 | 9.90 | 1.096 | 52 / 45 | 29 / 23 | 1.91 | 一致 |
| code | spec | 2 | 8.32 | 9.25 | 1.112 | 62 / 52 | 32 / 25 | 1.88 | 一致 |
| tool | off | 1 | 5.83 | 6.45 | 1.106 | 42 / 34 | 22 / 17 | 1 | 一致 |
| tool | off | 2 | 6.30 | 6.48 | 1.029 | 37 / 33 | 20 / 16 | 1 | 一致 |
| tool | spec | 1 | 7.98 | 8.65 | 1.084 | 78 / 62 | 34 / 32 | 1.95 | 一致 |
| tool | spec | 2 | 8.06 | 8.55 | 1.061 | 69 / 63 | 33 / 27 | 1.95 | 一致 |
| explain | off | 1 | 6.67 | 6.85 | 1.027 | 33 / 28 | 17 / 15 | 1 | 一致 |
| explain | off | 2 | 6.79 | 6.56 | 0.966 | 32 / 31 | 17 / 15 | 1 | 一致 |
| explain | spec | 1 | 7.95 | 8.70 | 1.094 | 53 / 45 | 30 / 24 | 1.73 | 一致 |
| explain | spec | 2 | 7.67 | 8.63 | 1.125 | 53 / 46 | 31 / 23 | 1.72 | 一致 |

12K (チャンク 2048、memlog 越し):

| プロンプト | MTP | パス | 既定 tok/s | W tok/s | W / 既定 | kernel→GPU 既定 / W | advise 既定 / W | wired 最大 既定 / W (GB) | file-backed 最小 既定 / W (GB) | Swapouts | トークン列 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| long | off | 1 | 5.46 | 5.35 | 0.980 | 41 / 39 | 21 / 18 | 13.21 / 13.29 | 2.65 / 2.60 | 0 / 0 | 一致 |
| long | spec | 1 | 6.81 | 7.53 | 1.106 | 69 / 60 | 38 / 31 | 14.71 / 14.39 | 1.34 / 1.59 | 0 / 0 | 一致 |
| long | spec | 2 | 6.54 | 7.29 | 1.115 | 69 / 58 | 36 / 28 | 15.03 / 14.63 | 1.05 / 1.33 | 0 / 0 | 一致 |

- ユーザーの判断で途中で止めた。long のパス 2 の off (既定の腕) は途中で切れ、long2 / long3 は流していない。
- 非常駐 MB の別走行 (`Q38_COUNT_MISS=1`、`scratch/qwen38/w3-miss.sh`) も流していない。

## 4. W-4: F32 → BF16

### 4-1. 検定 (`Scripts/qwen38/f32_bf16_check.py`)

BF16 は float32 と指数の幅が同じ (8 ビット) で、仮数は 7 ビット。float32 の値が損失なく表せるのは、仮数の下位 16 ビットが 0 のとき。
289 MiB を読んで 6.1 s、RSS 0.93 GB。

| 族 | 本数 | MiB | 下位 16 ビットが 0 でない要素 | max \|v\| |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_inp` | 49 | 245.0 | 0 | 3.281 |
| `ssm_alpha` / `ssm_beta` | 36 / 36 | 16.9 / 16.9 | 0 / 0 | 1.094 / 1.188 |
| `ssm_conv1d` | 36 | 5.6 | 0 | 0.9883 |
| `hc_attn_norm` / `hc_ffn_norm` | 49 / 49 | 1.9 / 1.9 | 0 / 0 | 10.94 / 13.38 |
| `ffn_gate_inp_shexp` | 49 | 0.5 | 0 | 0.09619 |
| `ple_conv1d` ほか norm 類 (13 族) | — | 約 0.3 | 0 | ≤ 14.88 |
| `ssm_a` | 36 | 0.007 | **1,728 (全部)** | 158 |
| 計 | 472 | 289.2 | 1,728 | |

- 非有限値と非正規化数は、どの族も 0。
- `ssm_a` の全要素は負。BF16 に丸めた `log(−a)` に float32 の `exp` を掛けて符号を戻すと、1,728 要素すべてが `a` とビット一致した (float64 の exp では 0.9977)。

### 4-2. サイドカーとカーネル (`Scripts/qwen38/bf16_sidecar.py`、`ggml_dense.metal`、`--bf16-gemv-check`)

- 対象は GEMV で読む 4 族 (`ffn_gate_inp`・`ffn_gate_inp_shexp`・`ssm_alpha`・`ssm_beta`、blk.48 を含む 170 本)。
  norm 類・conv1d・`ssm_a`・`ssm_dt.bias` は生の float32 バッファとして渡すので F32 のまま (合わせて 15 MB 足らず)。
- 置き場は `~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/bf16/Qwen3.8-Flash-Next-GatesBF16.gguf` (140.2 MiB、BF16 = float32 のビット列 >> 16)。
  書く前に全要素の下位 16 ビットを確かめ、1 つでも 0 でなければ止まる。`verify` で 73,198,080 要素すべて `BF16 << 16 == F32` (sha256 `5da18e7c…`)。
- カーネル: `GGML_FLOAT_GEMV` / `GGML_FLOAT_GEMV_CHUNK` に読み出しの式を引数として足し、`ggml_bf16_gemv`・`ggml_bf16_gemv_chunk`・`ggml_bf16_dequant_f32` を足した。
  F16 / F32 はマクロの展開結果が今までと同じ。
  BF16 → float32 はビット列を 16 ずらして float として読む。**MSL 3.2 には `bit_cast` が無い** (コンパイルエラー) ので、thread 空間のポインタで読み替えている。
- `GGMLDenseGEMV` は T < 32 で BF16 のレーン版 (行幅 1024 以上は chunk 版)、T ≥ 32 で BF16 を float32 に展開してから sgemm を呼ぶ。
- 検査 `--bf16-gemv-check <gguf> <sidecar> [tokens]` (`BF16GemvCheck.swift`): 170 本 × T = 1 / 3 / 32 / 64 の 680 件、シード固定の x で、F32 と BF16 の `y` の違う要素は 0。

### 4-3. ランナー (`Q38_BF16_SIDECAR`) と既定化

- `view(name)` は、BF16 のサイドカーにその名前があればそちらのビューを返す。`setF32` の precondition は残した (サイドカーに生の float32 で読むテンソルは入れていない)。
- 両サイドカーとも、ファイルがあれば既定で使う。`=0` で無効、`=1` で既定の置き場 (無ければエラー)、それ以外はパス。
- greedy のスモーク (code、40 トークン、spec): 有効・無効のどちらも、W-4 の前に取った走行 (§3 のスモーク) とトークン列・20 ステップの logit が一致した。

### 4-4. 12K で両サイドカーの有効・無効 (`scratch/qwen38/w4-long.sh`)

long、spec、instruct seed 1、200 トークン、チャンク 2048、memlog 越し。順は 無効 → 有効 → 有効 → 無効。

| 走行 | steady tok/s | 1 ステップ中央値 ms (ドラフト + 検証) | wired 最大 GB | file-backed 最小 GB | Swapouts | prefill 幹 / MTP s |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 無効 r1 | 7.07 | 249 (28 + 212) | 15.00 | 1.07 | 0 | 121.6 / 11.2 |
| 有効 r1 | 7.60 | 227 (27 + 193) | 14.28 | 1.65 | 0 | 121.0 / 11.1 |
| 有効 r2 | 7.60 | 235 (27 + 197) | 14.57 | 1.40 | 0 | 119.6 / 11.2 |
| 無効 r2 | 6.98 | 251 (27 + 213) | 14.84 | 1.08 | 0 | 119.0 / 11.1 |

4 本ともトークン列 (md5 `4107251e…`) と受理 (109 ステップ、0.826、1.83 トークン/ステップ) は同じ。

## 5. 触ったもの

- 本体:
  - `moe_ggml.metal` (`block_q2_K_half`、phase 2 と逆量子化)
  - `ggml_dense.metal` (BF16 の 3 カーネル)
  - `GGMLDenseGEMV.swift` (BF16)
  - `GGUFFile.swift` (I8、複数ファイルの `preadRanges`)
  - `Qwen38Runner.swift` (`Q38_DOWN_SIDECAR`・`Q38_BF16_SIDECAR`、`expertParts`)
- 検査:
  - `Q2DownSidecarCheck.swift` (`--q2-down-sidecar`)
  - `BF16GemvCheck.swift` (`--bf16-gemv-check`)
  - `Q2GemmBench.swift` (buffer(7))
- スクリプト: `Scripts/qwen38/{down_sidecar,bf16_sidecar,f32_bf16_check}.py`
- 触っていない: 参照器 `reference_forward.py`・`expert_kernel_fixture.py`・`mtp_reference.py` は元の GGUF を読む。`Qwen38ResidencyProbe` のプローブも元の GGUF のまま。

## 6. 残り

- W-3 の 12K の long2 / long3 と、long パス 2 の off。非常駐 MB の別走行 (`scratch/qwen38/w3-miss.sh`、code と long を 既定 → W → W → 既定)。
  いまの既定は両サイドカー有効なので、腕は `Q38_DOWN_SIDECAR=0 Q38_BF16_SIDECAR=0` を明示する。
- サイドカーはリポジトリの外 (`~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/{down,bf16}/`) にある。別の機材では、2 本のスクリプトで作ってから使う (無ければ自動で無効)。
- 15 §2 の B (expert の台帳) は、X (MB) を expert 1 個 1,387,520 B で数え直す。
