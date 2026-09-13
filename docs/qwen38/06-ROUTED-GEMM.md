# 06. routed expert を「expert ごとに float32 に展開して sgemm」に — 8K で 119 → 86.5 s (チャンク 2048 で 130.8 → 117.7 s)

実測: 2026-09-13、M3 Pro 18 GB、macOS 15。対象は [05](05-EXPERT-READ.md) と同じ GGUF・ランナー・プロンプト。
表記は 01 と同じ (実測 / 導出 / 未確認)。**ランナーの速度の行はどれも n=1 で、数字だけを書く。**1 層ベンチは 3〜5 回の中央値。

## 0. 結論

1. **T ≥ 1024 のバッチは、routed expert を組 (トークン, k) ごとの量子化カーネルでなく、
   「expert ごとに組を集める → その expert の gate/up/down を 1 回だけ float32 に展開 → MPS sgemm → 重み付きで散らす」で回す** (`Qwen38Runner.routedGemm`)。
   T < 1024 (decode を含む) は従来の `moe_iq2xxs_phase1_gate_up_act` / `moe_q2k_phase2_down_reduce` のまま。`Q38_GEMM_MIN_T=0` で全部を従来経路に。
2. **1 層 (実際の選択、T = 4096)**: 568〜572 → **240〜252 ms** (GPU)。層 0 / 10 / 24 / 47 のどれでも同じ幅。
   内訳は sgemm gate/up 95 ms・展開 83〜96 ms・sgemm down 47 ms・集め 8 ms・散らし 5 ms (§2)。05 §6-3 の見積もり「上限 1.6 倍」は外れた (sgemm はこの形で約 1.4T 積和/s、導出)。
3. **正しさ**: 1 層で従来カーネルとの相対差 1.1e-6〜2.1e-6。`Q38_GEMM_MIN_T=1` で全経路を新方式にして参照と照合し、France decode 7.42e-7、Fuji チャンク 16 で 6.06e-7、チャンク 53 で 2.11e-6、Fuji PLE も PASS (§2-3)。
4. **速度** (8,192 トークン、§3):

   | | チャンク 2048 | チャンク 4096 |
   | --- | ---: | ---: |
   | 05 (pread) | 130.8 s (62.6 tok/s) | 119.2 s (68.7 tok/s) |
   | 06 | **117.7 s (69.6 tok/s)** | **86.5 s (94.7 tok/s)** |
   | routed GPU / チャンク | 13.7 → 9.0〜9.7 s | 27.6 → 11.9〜12.2 s |
   | peak footprint | 3.37 → 3.70 GB | 5.02 → 5.59 GB |

5. **途中で Swapouts を約 4.1 GB 出した** (§4)。`guarded.sh` が `/usr/bin/time` だけを殺し、ベンチ本体が残って次の走行と重なった。`guarded.sh` は子孫ごと殺すように直した。
6. 残る 1 チャンク (4096、2 つ目) 43.1 s の内訳: pre-router の GPU 16.4 s (GDN step 6.3・GDN 入力 2.6・注意 3.3)、**routed GPU 12.2 s**、route 8.2 s (読み 5.5・top-10 1.3)、PLE 3.4 s (§5)。

## 1. 形

1 層の流れ (`moeRouted` → `routedGemm`、1 つのコマンドバッファ):

1. host: top-10 (従来どおり)、スロットごとの組の数 `count`・先頭 `start`、並べ替え `order` (位置 → 組) と逆引き `at` (組 → 位置)。expert の読み (pread) と per-expert の no-copy ビューも従来どおり。
2. `moe_gather_pair_rows`: `rows[i] = mixed[order[i] / 10]` (P × 2560、P = 10T)。
3. `gemmGroup` (既定 4) expert ずつ:
   - `moe_iq2xxs_dequant_gate_up_f32`: gate と up の 640 行ずつを `[G][1280][2560]` に。`moe_q2k_dequant_down_f32`: down の 2560 行の先頭 640 列を `[G][2560][640]` に
     (pad 列 640..767 は acts が 0 なので落とす)。どちらも従来カーネルと同じ引数バッファ (`routedArg`・`partOffsets`) で読み、`first_slot` からの G スロット。
   - expert ごとに sgemm `rows[start..] × W_guᵀ → gu [n][1280]`、`moe_silu_mul_halves` で `gu[:640] = silu(gate) · up`、
     sgemm `gu[:640] × W_downᵀ → rows[start..]` (その expert の入力行は読み終わっているので同じ場所に書き戻す)。
4. `moe_scatter_weighted`: `blk[t] = blkShared[t] + Σ_k w[t,k] · rows[at[t,k]]`。

一時領域 (maxBatch = B): `gemmRows` B×10×2560 float (B = 4096 で 419 MB)、`gemmGateUp` は**グループ内の組の最大数**ぶん (T = 4096 で数十 MB、導出)、
重み G×(1280×2560 + 2560×640) float (G = 4 で 79 MB)、`order`/`at`。

Q2_K のビット配置は gguf-py / `dequantize_row_q2_K` のとおり: サブブロック s (16 重み) は byte `32·(s/8) + 16·(s%2) + l`、shift `2·((s%8)/2)`。
「4 サブブロックで 16 バイト」ではない (ここを取り違えると静かに間違う。1 層の相対差で確認済み)。

## 2. 1 層の A/B (`--q2-gemm-bench`)

入力は 4,096 トークン 1 チャンクの実際の選択 (`Q38_DUMP_ROUTE` で 48 層ぶん書き出したもの)。`x` と residual は乱数、A と B に同じもの。expert は先に pread で温めてある。

### 2-1. 層とグループの大きさ (T = 4096、3 回の中央値)

| 層 | expert 数 | 1 expert の組 (中央値 / 最大 / 8 未満の数) | グループ | A GPU | B GPU | B encode | 相対差 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 24 | 439 | 36 / 1504 / 102 | 2 | 567.8 ms | 237.4 ms | 19.9 ms | 1.18e-6 |
| 24 | 439 | | 4 | 567.8 | 240.5 | 18.5 | 1.51e-6 |
| 24 | 439 | | 8 | 568.5 | 241.9 | 17.2 | 1.58e-6 |
| 24 | 439 | | 16 | 568.9 | 242.5 | 16.8 | 1.26e-6 |
| 24 | 439 | | 32 | 568.7 | 243.5 | 16.3 | 2.09e-6 |
| 24 | 439 | | 64 | 569.3 | 246.5 | 16.7 | 1.76e-6 |
| 0 | 485 | 63 / 950 / 25 | 16 | 571.9 | 252.3 | 18.6 | 1.49e-6 |
| 10 | 434 | 58 / 1405 / 74 | 32 | 569.3 | 239.9 | 16.2 | 1.68e-6 |
| 47 | 449 | 17 / 2420 / 151 | 16 | 569.0 | 248.8 | 16.7 | 1.10e-6 |

(グループ 2 と 4 は 5 回の中央値。この 2 行以外と §2-2 は、展開カーネルが expert テンソル丸ごとのビューを読んでいた版 (§4) で測った。計算は同じで、層 24・グループ 16 は両版で 242.5 / 242.0 ms。) B を段ごとに別バッファで流した GPU ms (層 24、グループ 16): sgemm gate/up 95.6、展開 87.8、sgemm down 46.9、集め 8.0、散らし 4.7、silu 1.8。

- sgemm は gate/up が 40,960 組 × 1280 × 2560 = 137G 積和を 95.6 ms、down が 67G を 46.9 ms で、どちらも約 1.4T/s (導出)。
- 展開は 1 expert あたり約 0.2 ms (5.2M 重み = 21 MB の float32 書き込み)。約 100 GB/s の書き込みで、帯域で頭打ちと見ている (未確認)。グループの大きさでは変わらない。

### 2-2. T を変える (層 24、先頭 T トークンの選択、グループ 16、5 回の中央値)

| T | expert 数 | 組の中央値 | A GPU | B GPU | B の内訳 (展開 / gate・up / down) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 185 | 4 | 17.6 ms | 68.3 ms | 36.5 / 21.3 / 10.2 |
| 256 | 246 | 6 | 35.3 | 92.8 | 48.3 / 29.8 / 13.8 |
| 512 | 290 | 10 | 71.0 | 112.9 | 58.0 / 37.2 / 17.5 |
| 1024 | 322 | 17 | 141.8 | 133.8 | 63.7 / 45.5 / 21.6 |
| 2048 | 395 | 24 | 284.1 | 180.6 | 78.4 / 65.2 / 31.7 |
| 4096 | 439 | 36 | 568.9 | 242.5 | 87.8 / 95.6 / 46.9 |

A は組の数に比例、B は expert 数ぶんの固定費 (展開 0.2 ms + sgemm の呼び出し) が乗る。**切り替えを T ≥ 1024 にした**のはこの表から。

### 2-3. 参照との照合 (`Q38_GEMM_MIN_T=1` で decode も含め全部を新方式に)

| 検査 | 結果 |
| --- | --- |
| `--qwen38-decode` France | top-1 0 ずれ、logits 7.42e-7 |
| `--qwen38-prefill` France チャンク 4 | 7.42e-7 |
| Fuji 予算 16、チャンク 16 (グループ 3) | 6.06e-7 |
| Fuji 予算 16、チャンク 53 (グループ 4 / 5) | 2.11e-6 |
| Fuji PLE チャンク 53 | PASS |

既定 (T < 1024 は従来経路) の 05 §6-2 の一式も全部 PASS で、数字は 05 と同じ。

## 3. ランナーでの結果 (`Q38_SPLIT_PRE=1 --qwen38-prefill-bench`、8,192 トークン)

チャンク 4096:

| | 合計 | pre wall / GPU | route (うち読み) | routed wall / GPU | PLE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 従来 (`Q38_GEMM_MIN_T=0`、同じ日に再走)、0..<4096 | 61.5 s | 18.3 / 14.8 s | 6.6 s (5.1) | 28.2 / 27.5 s | 6.6 s |
| 従来、4096..<8192 | 57.4 s | 18.9 / 16.5 s | 6.8 s (5.4) | 28.3 / 27.5 s | 3.4 s |
| 06、0..<4096 | 43.4 s | 18.1 / 14.7 s | 7.6 s (5.0) | 12.4 / 11.9 s | 3.4 s |
| 06、4096..<8192 | 43.1 s | 18.6 / 16.4 s | 8.2 s (5.5) | 12.8 / 12.2 s | 3.4 s |

従来の再走は 118.8 s (05 は 119.2 s)。06 は 86.5 s。route が 1 s ほど増えたのは組の並べ替えと sgemm の encode がここに入るため (導出)。

チャンク 2048 (06): 30.8 / 28.1 / 29.0 / 29.8 s、routed wall 9.6〜10.4 s / GPU 9.0〜9.7 s、読み 4.6〜5.5 s。合計 117.7 s。
1 層ベンチからの見込み (48 層 × 181 ms = 8.7 s、48 × 243 ms = 11.7 s、導出) とほぼ合う。

decode (8,192 の直後 1 トークン、従来経路のまま): 0.51 s (チャンク 4096 の走行)、0.48 s (2048)。05 は 0.44 s。

### 3-1. メモリ

| | peak footprint | wired の最大 | file-backed の最小 |
| --- | ---: | ---: | ---: |
| 従来、チャンク 4096 | 5.02 GB | 11.72 GB | 2.26 GB |
| 06、チャンク 4096 | 5.59 GB | 12.49 GB | 2.00 GB |
| 06、チャンク 2048 | 3.70 GB | 12.38 GB | 2.48 GB |

(wired・file-backed は `vm_stat` を 2 秒ごと、`scratch/qwen38/memlog.sh`、1 GB = 10⁹ バイト。従来のチャンク 2048 は記録していない。)
どの走行も 2 チャンク目で wired が 10 GB 超に上がり file-backed が 2〜3 GB に下がる。これは従来経路でも同じで、06 は wired の最大で約 0.8 GB 高く、file-backed の最小で約 0.3 GB 低い。Swapouts の増分はどれも 0。

**途中で直したもの 2 つ** (per-expert ビューに直した後でも、グループ 16 のままでは 8K チャンク 4096 で footprint 6.64 GB、下の一時領域を縮めた後もプール無しではチャンク 2048 で 4.94 GB だった):

- **autorelease**: expert ごとに作る `MPSMatrix` / `MPSMatrixDescriptor` (`MPSMatrix` だけで 1 チャンク約 13 万個、導出) がプールの無い CLI で forward の間ずっと残っていた。
  `footprint` で MALLOC_TINY 175 MB・MALLOC_SMALL 107 MB が走行中に増え続けていた。グループごとに `autoreleasepool` で包み、MALLOC_TINY は 21〜30 MB に。
- **一時領域**: グループの既定を 16 → 4 (重み 315 → 79 MB、速さは §2-1 で同じ)、`gemmGateUp` を P 行 (210 MB) からグループ内の最大組数ぶんに。

## 4. 事故: Swapouts 約 4.1 GB

最初の組み込み (展開カーネルが **expert テンソル丸ごと 750 MB の no-copy ビュー** × 3 本/層を読む形) で、05 §6-2 のとおり
`guarded.sh out /usr/bin/time -l env Q38_SPLIT_PRE=1 $B --qwen38-prefill-bench ... --q38-chunk 4096` を流した。

1. チャンク 4096 の走行で Swapouts +16,668 ページ (約 270 MB) になり、見張りが kill した。
2. **kill したのは `/usr/bin/time` だけで、ベンチ本体は生き残った。**20 秒後に始めたチャンク 2048 の走行と重なり、2 本目も +8,796 ページで kill (同じく本体は残った)。
3. 気づいたときは 2 本のベンチが並走していて、wired 899,370 ページ (約 14.7 GB)、Swapouts はセッション開始時から **+252,494 ページ (約 4.1 GB)**。

処置:

- `Scripts/qwen38/guarded.sh` を子孫ごと殺す形に (`pgrep -P` を再帰)。`GUARD_PAGES=-1` で見張りを必ず発火させ、`/usr/bin/time -l env ... zsh -c 'sleep 40'` の木が消えることを確認した。
  終了時に子孫が残っていれば `GUARD: descendants ... still alive` を書く。
- 展開カーネルを従来と同じ per-expert ビュー (引数バッファ) で読む形に変えた。丸ごとのビューは GPU に載せるとき選ばれていない expert のページまで常駐させるので。
  **ただし 1 の +270 MB がビューのせいだったかは確かめていない** (危ないので丸ごとビューの版は再走していない)。autorelease の増え (§3-1) も同時にあった。

## 5. 次

チャンク 4096 の 2 つ目 43.1 s の順:

1. **pre-router の GPU 16.4 s**: GDN step 6.3 s (時間方向に直列)、GDN 入力 GEMM 2.6 s、注意 3.3 s (n とともに伸びる)、hc 2.3 s。
   step のチャンク内並列形 (delta rule の WY / chunked 形) が一番大きい (05 §6-3 の 2 番)。
2. **routed の GPU 12.2 s** (展開 約 4.2 s・sgemm 約 6.8 s、1 層ベンチ × 48 の導出)。展開は帯域律速と見ているので、
   組の少ない expert (T = 4096 で 8 組未満が 25〜151 個) だけ従来カーネルに回す混成が次の候補。
3. **route 8.2 s** (読み 5.5 s・top-10 1.3 s・並べ替えと encode 約 1.4 s)、PLE 3.4 s。
4. **32K を 1 回** (チャンク 4096、`guarded.sh` 越し、注意の伸びを入れない導出で約 6 分)。wired が 2 チャンク目で 12 GB 台になるので、長文脈では Swapouts を特によく見る。
   `gemmRows` (B = 4096 で 419 MB) はグループごとに集めて expert ごとに散らせば数十 MB にできる (encode が増える、未測定)。

## 6. 足したもの

| 種類 | パス |
| --- | --- |
| カーネル | `moe_ggml.metal`: `moe_iq2xxs_dequant_gate_up_f32`、`moe_q2k_dequant_down_f32`、`moe_gather_pair_rows`、`moe_silu_mul_halves`、`moe_scatter_weighted` |
| ランナー | `routedGemm`、`gemmMinTokens` (`Q38_GEMM_MIN_T`、既定 1024、0 で従来)、`gemmGroup` (`Q38_GEMM_GROUP`、既定 4)、`routeDumpPrefix` (`Q38_DUMP_ROUTE`、T ≥ 1024 のバッチの選択を層ごとに書く) |
| 検査 | `--q2-gemm-bench <gguf> <route .bin> [group] [iterations] [tokens]` (`Q2GemmBench.swift`) |
| スクリプト | `Scripts/qwen38/guarded.sh` (子孫ごと kill、`GUARD_PAGES`)、`scratch/qwen38/memlog.sh` (git 管理外、guarded.sh + 2 秒ごとの wired / free / file-backed / Swapouts / RSS) |

再現:

```bash
mkdir -p scratch/qwen38/routes
Scripts/qwen38/guarded.sh dump.out env Q38_DUMP_ROUTE=scratch/qwen38/routes/c4096 $B --qwen38-prefill-bench scratch/qwen38/prompt-code.tokens --q38-tokens 4096 --q38-chunk 4096
Scripts/qwen38/guarded.sh ab.out $B --q2-gemm-bench $G scratch/qwen38/routes/c4096-l24.bin 4 3          # A/B と相対差
Scripts/qwen38/guarded.sh ab.out $B --q2-gemm-bench $G scratch/qwen38/routes/c4096-l24.bin 16 5 1024     # 先頭 1024 トークンで
```

## 7. 落とし穴

- **見張りは子孫ごと殺す。**`/usr/bin/time` や `env` を挟むと、直接の子を kill してもベンチは残る (§4)。走行の後に `pgrep -x TsugumiKernelCheck` が空であることを見る。
- **GPU に渡す no-copy ビューは使う範囲だけにする。**丸ごとのテンソルビューは選ばれていない expert まで常駐対象になる (§4、影響の大きさは未確認)。
  また `views` は dense の residency set に入るので、expert のビューをそこに入れない。
- **expert ごとに `MPSMatrix` を作る経路は `autoreleasepool` で包む。**CLI にはプールが無く、forward 全体ぶん溜まる (§3-1)。
- `MPSMatrixMultiplication` は結果の行数が init で固定。expert の組数ごとに作ってキャッシュ (`attnMuls`、キー `[10, n]` / `[11, n]`)。
- 左行列に `rowBytes` 付きの記述子 (`gu` の前半 640 列だけを 1280 列ストライドで読む) は MPS でそのまま通る (参照一致で確認)。
- Q2_K のサブブロックとバイトの対応 (§1)。

## 8. 再開手順 (新しいセッションで続けるとき)

05 §6 の後継。**05 §6-4 と 03 §6-4 の落とし穴はそのまま有効**なので、先に一度読む。

### 8-1. 状態

- コードはこの文書と同じコミットまで入っている。ランナーは `Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift` 1 本。
  - 注意: T ≥ 32 で選択なしは `attentionSgemm`、選択ありは `attentionSelected` (04)、T < 32 は `attentionHost`。
  - expert の読み: T ≥ 32 は `GGUFFile.preadRanges` (4 スレッド)、T < 32 は `F_RDADVISE` (05)。
  - routed expert: T ≥ 1024 は `routedGemm` (06)、それ未満は per-pair カーネル。
- 速度の現在地 (8,192 トークン、n=1): チャンク 4096 で 86.5 s (94.7 tok/s、footprint 5.59 GB)、チャンク 2048 で 117.7 s (69.6 tok/s、3.70 GB)。
  decode は短文脈 0.09〜0.11 s/トークン、8K の直後 0.48〜0.51 s。
- **32K はまだ流していない。**
- ウェイト・トークナイザの在処は [01 §7-1](01-Q2-FIRST-LIGHT.md)。`G=~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`。
- `scratch/qwen38/` (git 管理外): 05 §6-1 のもの + `routes/c4096-l{0..47}.bin` (§6 の 1 行目で作り直せる)、`memlog.sh` (§6)。
- 運用点は thinking 無効・32K・エージェント主体・英語主体 (メモリ `qwen38-operating-point`)。

### 8-2. 検査を一通り (どれも数分以内、GPU は 1 本ずつ、間に 20 秒)

05 §6-2 の 11 本 (期待値も同じ) に加えて:

```bash
Q38_GEMM_MIN_T=1 $B --qwen38-decode scratch/qwen38/ref-france-dump.log --q38-ref-logits scratch/qwen38/ref-france.logits
Q38_GEMM_MIN_T=1 $B --qwen38-prefill scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 16
Q38_GEMM_MIN_T=1 $B --qwen38-prefill scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 53
Q38_GEMM_MIN_T=1 $B --qwen38-prefill scratch/qwen38/ref-fuji-ple.log --q38-chunk 53
$B --q2-gemm-bench $G scratch/qwen38/routes/c4096-l24.bin 4 3
```

期待値: 7.42e-7、6.06e-7、2.11e-6、PASS、A 約 568 ms / B 約 240 ms / 相対差 2e-6 以下。

速度 (約 1.5 分と 2 分、memlog 越しに):

```bash
scratch/qwen38/memlog.sh mem.log out.txt /usr/bin/time -l env Q38_SPLIT_PRE=1 $B --qwen38-prefill-bench scratch/qwen38/prompt-code.tokens --q38-tokens 8192 --q38-chunk 4096
grep -E "^\s+\[|GPU ms|prefill|decode|footprint|GUARD" out.txt; pgrep -x TsugumiKernelCheck || echo none
```

§3 の表と比べる。

### 8-3. 次にやること

§5 の順 (GDN step の並列形 → routed の混成 → route / PLE → 32K)。32K の前に `gemmRows` の縮小を考える。

### 8-4. 落とし穴

§7 と、05 §6-4・03 §6-4。
