# 09. 12K を Swapouts なしで回す (dense の residency set を切る・decode でバッチ一時領域を縮める) と、T 行 forward のコスト

実測: 2026-09-14、M3 Pro 18 GB、macOS 15。対象は [08](08-ROUTED-MIXED.md) と同じ GGUF・ランナー。
表記は 01 と同じ (実測 / 導出 / 未確認)。**ランナーの速度・メモリの行はどれも n=1 の走行で、数字だけを書く。**幅ごとの行は 1 走行の中の中央値 (n は各行)。

運用点は同日 32K → **12K** に下がった (メモリ `qwen38-operating-point`)。MTP は thinking off + 公式 instruct サンプリングで受理率が高いことが PC / llm-server で分かっている (§4-1)。

## 0. 結論

1. **12,000 トークンの prefill (チャンク 2048) → decode → 検証パスの幅測定が Swapouts 0 で通るようになった** (§2)。prefill 133.6 s (89.8 tok/s)、wired の最大 13.78 GB、file-backed の最小 2.70 GB。
   変更前は 12K で 2 回、8K で 2 回、見張りが kill した (計 約 900 MB の Swapouts、§1)。
2. 直したもの (§2):
   - **dense の residency set を既定で切った** (`Q38_DENSE_RESIDENT=1` で戻る)。prefill と steady な decode は同じ速さで、prefill 中の wired が 12〜13 → 7〜8.5 GB。代償は prefill 直後の最初の 1 トークン (dense を読み戻して 1.5〜2.1 s)。
   - **バッチ比例の一時領域を、T ≤ 32 の forward で 32 行に作り直し、大きい T で maxBatch に戻す** (`allocateBatch`、`Q38_SHRINK_BATCH=0` で従来)。decode 中の自前のバッファ 1,412 → 22 MB。
   - **forward を `autoreleasepool` で包んだ**。無いと CLI では捨てたバッファがコマンドバッファ経由で残る (+1.2 GB)。
   - `attnScores` / `selection` を maxBatch でなく使う行数で確保 (チャンク 2048 で 403 → 201 MB)。
3. **検証パス (T 行の forward) のコスト** (§3): 12K の直後で T=1 171〜212 ms、T=2 317〜347 ms (1.64〜1.85 倍)。短文脈 (256〜4K) では 1.28〜1.53 倍。
   pre-router の GPU は T=2 で 1.5 倍、expert の和集合は 1.7〜1.75 倍 (PC の 1.665 と同じ)、SSD 待ちと advise もそれに比例して伸びる (プロンプトの続きで測った値。ドラフトの自己生成トークンでは未測定)。
4. **「dense をトークンごとに読み直している」は主因ではなかった** (§3-2)。重みブロックを 1 回だけ読んで T 本と内積を取るカーネルを書いたが、T=2 の pre-router GPU は 78 → 79 ms で変わらず (hc の F16 は悪化)。消した。
5. **MTP の効きはまだ何も分かっていない。**ドラフトの費用、Mac での受理率、自己生成トークンでの expert 和集合、巻き戻しの費用がどれも未測定 (§4)。
   このセッションでは §3 の数字に仮の値を掛けた損益表を作って MTP の是非を論じたが、判断の材料にならないので消した。**次は MTP を実装して実測する** (§5・§8)。

## 1. 何が起きていたか

| 走行 | 落ちた場所 | wired / file-backed (直前) | Swapouts |
| --- | --- | --- | ---: |
| 8K チャンク 4096 (08 §4-1、2 本) | 2 チャンク目 | 12.5〜12.6 / 1.2〜1.9 GB | +10,752 / +5,960 |
| 8K チャンク 2048 + 幅測定 (7,960 トークン) | 最後のチャンク | 12.68 / 1.93 GB | +15,916 |
| 12K チャンク 2048 + 幅測定 | 最後のチャンク | 14.52 / 0.91 GB | +20,420 |
| 12K、residency set なし | prefill 後の decode | 14.16 / 1.61 GB | +4,904 |

どれも `guarded.sh` が子孫ごと kill し、残ったプロセスは無い。走行前の時点で他プロセスのページが compressor に約 8 GB (48〜51 万ページ)、swap 使用 1.8 GB あった (環境側の余裕も小さい)。

計器 (`Q38_MEM_LOG=1`、`Qwen38Runner.memoryReport`) で 4K / チャンク 2048 の中身を数えた:

| 中身 | MB |
| --- | ---: |
| 固定のバッチ用バッファ (maxBatch 比例) | 1,420 |
| `attnScores` + `selection` | 420 |
| gemm | 316 |
| KV + インデクサ鍵 (capacity 8,194) | 466 |
| dense scratch・GDN 状態・注意の一時領域 | 約 300 |
| **dense ビュー (residency set)** | **5,418** |
| expert ビュー (64K 本、ファイルのページ上) | 32,816 |
| その他 (MPS など = Metal の確保量 − 上の合計) | 127〜183 |

wired の時系列 (memlog) で分かったこと:

- **wired は最初の forward の間 約 7 GB で、その forward が終わった瞬間に 11.4 GB へ跳ねる。**そこが residency set の `commit` + `requestResidency`。以後 dense 5.4 GB は使っていない間も wired のまま。
  08 §4-1 の「チャンク 4096 の 2 チャンク目で跳ねる」の正体はこれ (1 回目の forward の終わりが 2 チャンク目の頭)。
- **residency set なしでも、decode (T < 32) に入ると wired は 11.5〜14 GB になる。**prefill では 1 層を触るのが数十秒に 1 回なので GPU が離したページは wired から外れ、decode では 170 ms ごとに全層の dense と直近の expert を触るので外れない、と見ている (未確認)。
  expert のビューを層ごとに作って捨てても 11〜12 GB で変わらなかった (ビューの寿命ではない)。
- decode 中の自前のバッファ (maxBatch 2048 で約 3 GB) は、vmmap で IOAccelerator (graphics) resident 2.5 GB・swapped 1.5 GB。**`setPurgeableState(.empty)` → `.nonVolatile` ではページは戻らなかった** (NONVOL 2.5 GB のまま)。作り直し (§2) で戻る。
- 「MPS が行列積の大きさごとに一時領域を持つ」仮説は、その他 127〜183 MB で外れ。

## 2. 直したもの

### 2-1. residency set を既定で切る

4,096 トークン、チャンク 2048 (n=1):

| | residency set あり | なし |
| --- | ---: | ---: |
| prefill 合計 | 46.1 s | 46.1 s |
| wired の最大 / file-backed の最小 | 13.01 / 2.70 GB | 8.58 / 7.05 GB |
| prefill 直後の decode 1 トークン | 0.36 s | 2.10 s (pre 1,892 ms、GPU 103 ms) |
| steady な decode T=1 (4,000 の後、12 回の中央値) | 169 ms | 170 ms (短文脈 256 の後、150 回) / 175 ms (4K の後) |

03 §2-3 で入れたとき (expert の読みがフォルト / advise) は、dense が追い出されて pre-router の wall が GPU の数倍になった。05 で prefill の expert を pread に変えた後は測り直していなかった。

### 2-2. decode でバッチの一時領域を縮める

`forward` の頭で `T > batchRows` なら `allocateBatch(rows: maxBatch)`、`T ≤ 32` かつ `batchRows > 32` なら `allocateBatch(rows: 32)`。
作り直すのはバッチ比例の 43 本と `idxScores`、捨てるのは注意・gemm・dense scratch の伸びる一時領域。状態 (KV・インデクサ鍵・GDN の hist / state・`pleHist`) は触らない。
`acts` は新しいバッファなので pad 列は 0 (phase 2 が読む、08 §1)。

**forward が返す `UnsafeBufferPointer` は次の forward まで有効。**`logits` を作り直すと古い方は解放されるので、呼び出し側は配列にコピーする (ベンチの decode の logits を持ち越していて SIGSEGV になった)。

### 2-3. 検査

- 07 §3 の 19 本: 08 の出力と logits が完全に一致。
- 切り替えを通る照合: Fuji 予算 16 をチャンク 32 (32 + 21) とチャンク 40 (40 で広げ 13 で縮める) で。縮めあり・なしで 2.16e-6 / 1.84e-6 と同じ値、PASS。Fuji PLE チャンク 40 も PASS。
- 長文脈: 4,000 トークン (チャンク 2048) の prefill の最後の logits と直後の decode の logits が、変更前の走行と**ビット単位で一致** (`--q38-compare-logits`、rel err 0.00e+00)。

### 2-4. 12K の走行

`Q38_MEM_LOG=1 Q38_BENCH_WIDTHS=1,2,3 Q38_BENCH_REPS=15 --qwen38-prefill-bench scratch/qwen38/prompt-code16k.tokens --q38-tokens 12000 --q38-chunk 2048` (memlog 越し):

| チャンク | 0..<2048 | 2048..<4096 | 4096..<6144 | 6144..<8192 | 8192..<10240 | 10240..<12000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 合計 | 21.44 s | 22.86 | 21.97 | 22.30 | 23.52 | 21.49 |
| pre wall / GPU | 7.07 / 5.10 | 7.94 / 5.05 | 8.48 / 5.52 | 8.96 / 5.89 | 9.52 / 6.34 | 8.73 / 5.69 |

prefill 133.6 s (89.8 tok/s)、prefill 直後の decode 1.97 s、peak footprint 3.55 GB、wired の最大 13.78 GB、file-backed の最小 2.70 GB、Swapouts 0。
同じ条件の 2 本目 (幅 1・2 を 40 回) は 130.5 s、wired の最大 12.93 GB、Swapouts 28 ページ (見張りの閾値 4,096 未満)。

`prompt-code16k.tokens` はランナー・ベンチ・GGUF 読み・dense カーネルの Swift 5 本を `<file path=…>` で包んで上流トークナイザで符号化した 39,025 トークンの先頭 16,384 個 (§6)。

## 3. 検証パスのコスト (`Q38_BENCH_WIDTHS=1,2,3 Q38_BENCH_REPS=R`)

prefill の後、プロンプトの続きのトークンで T 行の forward (`allLogits: true`) を幅を交互にして R 回。位置は本文を進む。ドラフトの自己生成トークンではない (expert の和集合は近いと見ている、未確認)。

### 3-1. 幅ごとの内訳 (ms、中央値)

| 文脈 | T | n | wall | pre [GPU] | route | routed [GPU] | head | expert 和集合 |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| 256 | 1 | 12 | 147 | 61 [51] | 19 | 61 [9] | 5 | 480 |
| 256 | 2 | 12 | 208 | 68 [58] | 32 | 93 [18] | 8 | 816 |
| 256 | 3 | 12 | 360 | 139 [127] | 57 | 138 [50] | 14 | 1,045 |
| 256 | 5 | 12 | 438 | 136 [123] | 89 | 197 [53] | 16 | 1,580 |
| 4,000 | 1 | 12 | 166 | 74 [52] | 21 | 61 [9] | 5 | 480 |
| 4,000 | 2 | 12 | 212 | 102 [77] | 40 | 69 [26] | 8 | 753 |
| 4,000 | 3 | 12 | 346 | 145 [112] | 57 | 123 [42] | 13 | 995 |
| 12,000 | 1 | 15 | 182 | 76 [54] | 23 | 72 [11] | 5 | 480 |
| 12,000 | 2 | 15 | 334 | 143 [108] | 54 | 131 [37] | 10 | 838 |
| 12,000 | 3 | 15 | 415 | 160 [123] | 81 | 163 [48] | 11 | 1,222 |
| 12,000 | 1 | 40 | 212 | 98 [68] | 28 | 83 [15] | 6 | 480 |
| 12,000 | 2 | 40 | 347 | 143 [105] | 58 | 132 [37] | 10 | 849 |
| 12,000 | 1 | 30 | 171 | 74 [52] | 22 | 65 [10] | 5 | 480 |
| 12,000 | 2 | 30 | 317 | 132 [101] | 48 | 115 [27] | 9 | 832 |

(256 の行は §2 の変更前、4,000 の行は変更後、12,000 の 15 回・40 回の行は複数トークン版のカーネルを入れる前、30 回の行は入れた後。)
走行ごとの min / max は T=1 で 113〜407 ms と広い。

### 3-2. pre-router の区間 (短文脈 256、`Q38_SPLIT_PRE=1`、20 回の中央値、GPU ms)

| 区間 | T=1 | T=2 | T=4 | T=2 複数トークン版 | T=4 同 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 注意 | 8.7 | 13.9 | 24.6 | 12.0 | 20.3 |
| GDN 入力 (Q8_0) | 13.8 | 20.1 | 35.7 | 20.4 | 33.1 |
| GDN (norm_gate + ssm_out) | 6.5 | 8.8 | 14.5 | 7.8 | 12.0 |
| GDN step | 2.3 | 3.9 | 5.5 | 3.4 | 4.6 |
| hc_attn / hc_ffn (F16、行幅 320) | 8.0 / 8.1 | 11.6 / 10.9 | 15.8 / 16.1 | 14.4 / 14.4 | 24.7 / 24.7 |
| shexp + router | 4.9 | 7.1 | 12.1 | 7.5 | 12.2 |
| 合計 | 52 | 78 | 124 | 79 | 133 |

- 従来の `ggml_q8_0_gemv` はスレッドを (行グループ × トークン) で切り、トークンごとに重みを全部読む。それでも T=2 は 2 倍でなく 1.5 倍で、帯域だけで決まっていない (演算の分がある、導出)。
- 複数トークン版 (`ggml_{q8_0,f16,f32}_gemv_multi`、ブロックを 1 回読んで T 本と内積) は Q8_0 で変わらず、行幅 320 の F16 は 32 レーンのうち 10 本しか働かず悪化。**消した。**

## 4. MTP の見積もり

### 4-1. 受理率 (他機材の実測、`C:\LLM` on 192.168.0.199)

- PC (5080、llama.cpp unsloth MTP): 非 thinking・公式 instruct サンプリングで n_max 1 の受理 0.83〜0.89、n_max 2 で 0.75〜0.89 (`HANDOVER-pc-qfn-mtp.md`)。4k instruct の最速は n_max 4 (`RESULTS-pc-qsa-gather.md` §6)。
- llm-server (V100 + 3090): instruct temp 0.7 で位置別 0.94 / 0.88、n_max 2 で 2.83 トークン/ステップ (`HANDOVER-llm-server.md`)。
- thinking では 0.47〜0.73 に落ちる。受理判定はどれも「target のサンプル == ドラフトの argmax」。
- 連続 2 トークンの expert 和集合 1.665 倍 (`HANDOVER-qfn-engine.md`)。Mac の実測 1.57〜1.77 倍 (§3-1)。

### 4-2. Mac で未測定のもの (実装しないと出ない)

- ドラフト 1 回の費用 (blk.48 の注意 + MoE 10 expert の読み + LM head)。
- 受理率 (公式 instruct サンプリング / greedy、12K の運用点、英語のエージェント的な出力)。PC の値は量子化 (Q4_K_XL / IQ3_XXS) も実装も違う。
- 自己生成トークンでの T=2 の expert 和集合と SSD 待ち (§3 はプロンプト本文の続きで測った)。
- 巻き戻しの費用 (GDN 状態・conv 履歴・インデクサのブロック鍵・PLE 履歴・MTP 層の KV)。
- T 行カーネルと T=1 カーネルの加算順の差で、素の decode とトークン列がどれだけ変わるか (Ornith では 192 本中 34〜192 本一致、メモリ `ornith-mtp-acceptance`)。

§3 の T=1 / T=2 の費用はプロンプトの続きで測った実測値だが、これに仮のドラフト費用と他機材の受理率を掛けた損益は出さない。

## 5. 次: MTP を実装して実測する

一段ごとに実測が 1 つ出る順に並べる。見積もりで次の段に進むかを決めない。

1. **生成の土台** (`--qwen38-generate`): チャットテンプレートで包んだ英語のプロンプト (thinking off = `<think>\n\n</think>\n\n`) を prefill し、サンプラで N トークン生成する。
   トークン id・1 トークンごとの時間・区間を書き出す。サンプラは greedy と公式 instruct (§8-1)。**これが素の decode の基準線** (12K と短文脈)。
2. **MTP ヘッドの forward** (blk.48、§8-1 の参照実装どおり)。Q4_K (gate/up) と MXFP4 (down) のカーネルがまだ無い。
3. **影 (shadow) モード**: 素の decode を続けながら、毎ステップ MTP のドラフトだけ走らせて捨てる。**受理率 (ドラフトの argmax == target が引いたトークン) とドラフト費用が、巻き戻し無しで実測できる。**
   受理率が PC 並み (0.8 前後) にならなければ、ヘッドの実装を疑う (入力の hidden の取り方・hc の扱い)。
4. **投機ループ n_max 1**: T=2 の検証、受理・棄却、巻き戻し。中立性を「強制棄却の対照 == 素の decode のトークン列」で確かめる (greedy)。
5. **端から端の比較**: 同じプロンプトで素と MTP を交互に回す (腕を交互、プロンプト 3 本以上 × 反復 2 以上、200 トークン以上)。tok/s・受理率・1 ステップの内訳。12K と短文脈。

## 6. 足したもの・変えたもの

| 種類 | パス |
| --- | --- |
| ランナー | `denseResident` の既定を off (`Q38_DENSE_RESIDENT=1` で on)、`allocateBatch` / `batchRows` / `shrinkBatch` (`Q38_SHRINK_BATCH`)、`ensureAttnScores`、`forward` を `autoreleasepool` で包む (`forwardBody`)、`memoryReport` |
| dense | `GGMLDenseGEMV.scratchBytes`・`dropScratch` |
| 検査 | `--qwen38-prefill-bench` に `Q38_BENCH_WIDTHS` / `Q38_BENCH_REPS` (幅ごとの中央値・区間)、`Q38_MEM_LOG` (チャンクごと・最後のメモリ内訳)、標準出力を行バッファに、capacity は n + 2 + 幅測定の分 |
| スクリプト (git 管理外) | `scratch/qwen38/prompt-code16k.tokens` (§2-4) |

消したもの: 複数トークン版の dense カーネル (§3-2)、`setPurgeableState` による解放、層ごとに expert ビューを捨てるスイッチ (§1)。

## 7. 落とし穴

- **residency set で `requestResidency` した no-copy ビューは、ドライバが回収できない wired になる。**prefill 中に 5.4 GB 居座り、他プロセスが swap に出された。wired の時系列で「最初の forward の終わりに跳ねる」のが症状。
- **`setPurgeableState(.empty)` → `.nonVolatile` ではバッファのページは返らない** (vmmap の NONVOL がそのまま)。返したければ作り直す。
- **CLI には autorelease pool が無い。**コマンドバッファ・エンコーダ経由の参照で、捨てたバッファが残る。`memoryReport` の「その他」が GB 単位なら疑う。
- `forward` の戻り値のポインタは次の forward まで。作り直した `logits` を指したまま読むと SIGSEGV。
- `memlog.sh` のベンチ出力は `setvbuf` で行バッファにした (kill されても行が残る)。
- 見張りが kill すると次の走行の file-backed が戻るまで数秒かかる。間に 20 秒。

## 8. 再開手順 (新しいセッションで MTP を実装・実測する)

08 §8 の後継。**08 §7・07 §7・06 §7・05 §6-4・03 §6-4 と §7 の落とし穴はそのまま有効。**

### 8-1. 状態と材料

- コードはこの文書と同じコミットまで。ランナーは `Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift`。経路は 08 §8-1 のとおりで、加えて dense の residency set は既定 off、T ≤ 32 の forward でバッチ一時領域は 32 行。
- 速度・メモリ (n=1): 12K (チャンク 2048) の prefill 130.5〜133.6 s、Swapouts 0、wired の最大 12.9〜13.8 GB。steady な decode は 12K の直後で 171〜212 ms/トークン (プロンプトの続き)、prefill 直後の 1 トークン約 2 s。§3 の T 行の費用。
- **MTP は何も無い。**サンプラも生成ループも無い (検査は参照の logits と比べるだけ)。`forward(tokens:startPos:allLogits:)` が T 行を回せるので検証パスの土台はある。
- ウェイト: `G=~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`。blk.48 のテンソル:
  - `nextn.eh_proj` Q8_0 [5120 → 2560]、`nextn.enorm` [2560]、`nextn.hnorm` [10240]、`nextn.hc_head_{norm,down,up}` (F32/F16)
  - `hc_attn_*` / `hc_ffn_*` (F16)、`attn_{q,k,v,output}` Q8_0、`attn_{q,k}_norm`、`indexer.*` (Q8_0、ただしドラフトは密な注意)
  - `ffn_gate_inp` F32、`ffn_{gate,up}_exps` **Q4_K** [2560 → 640] × 512、`ffn_down_exps` **MXFP4** [640 → 2560] × 512 (計 1.39 GB)、`ffn_*_shexp` Q8_0
  - `nextn.embed_tokens` / `shared_head_head` は無い → トランクの `token_embd` (BF16、host で読む) と `output.weight` を使う。
- **参照実装** (PC `ssh masah@192.168.0.199`、`C:\LLM\llama.cpp-unsloth-mtp`、`a9e9c3c`): `src/models/qwen4exp.cpp`
  - 455〜480 行: トランクの最後の `res_hc` (hc 残差 4 × 2560、**出力の `hc_head` ミックスの前**) を `t_h_nextn` として MTP に渡す。
  - 489〜660 行 `graph_mtp`: 次トークンの埋め込み → `enorm` → hc 本数ぶん repeat、h は stream ごとに RMS → `hnorm`、**stream ごとに concat** して `eh_proj` → `hc_attn` ミックス → 注意 (「QSA は文脈を刈るだけなのでドラフトは密」とコメント) → gate → `attn_output` → combine → `hc_ffn` → FFN (MoE) → combine → 新しい `h_nextn` → `nextn.hc_head` ミックス → `output.weight`。
  - 投機のループは `common/speculative.cpp` (`--spec-type draft-mtp`)。受理判定は「target のサンプル == ドラフトの argmax」。
  - コピーは `scp masah@192.168.0.199:C:/LLM/llama.cpp-unsloth-mtp/src/models/qwen4exp.cpp .`。Windows の OpenSSH で PowerShell の入れ子引用符は崩れるので、読むときは scp する。
- **サンプリング**: 公式 instruct = temp 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5 (llm-server の `bench.py` と `C:\LLM\HANDOVER-llm-server.md` 44 行目。モデルカードでは未確認)。`~/LLM/Qwen3.8-Flash-Next-tokenizer/generation_config.json` は thinking 用 (temp 1.0 / top_p 0.95 / top_k 20)。
  thinking off はテンプレートの `enable_thinking=false` で `<think>\n\n</think>\n\n` (chat_template.jinja 165 行目)。
- 他機材の受理率はメモリ `qwen38-mtp-evidence` (instruct で n_max 1 が 0.83〜0.89、位置別 0.94 / 0.88)。
- 運用点: thinking off・12K・英語のエージェント主体 (メモリ `qwen38-operating-point`)。

### 8-2. 最初に確かめること

```bash
swift build -c release --product TsugumiKernelCheck
B=.build/release/TsugumiKernelCheck; S=scratch/qwen38
Scripts/qwen38/guarded.sh checks.out $S/checks07.sh
grep -c PASS checks.out; grep -i fail checks.out   # 24 と空
$B --qwen38-prefill $S/ref-fuji-idx16.log --q38-ref-logits $S/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 40   # PASS 1.84e-6 (バッチの伸縮を通る)
```

12K の土台 (約 2.5 分、memlog 越し、wired 13 GB 台・Swapouts 0 のはず):

```bash
scratch/qwen38/memlog.sh mem.log out.txt /usr/bin/time -l env Q38_BENCH_WIDTHS=1,2 Q38_BENCH_REPS=30 $B --qwen38-prefill-bench $S/prompt-code16k.tokens --q38-tokens 12000 --q38-chunk 2048
grep -E "prefill |decode|T=|footprint|GUARD" out.txt; pgrep -x TsugumiKernelCheck || echo none
```

### 8-3. 進め方

§5 の 1〜5。各段で次を守る:

- **数字は実測だけで判断する。**未実装の部分を仮の値で埋めた損益で、段を飛ばしたり止めたりしない。
- 速度は腕を交互に回す (ページキャッシュの温まりで逐次は偽の差が出る、メモリ `nvmai-sibling-runtime`)。
- 1 段ごとに `docs/qwen38/10-…` 以降に実測を書き、コミットする。番号は書く前に `ls docs/qwen38/`。
- MTP の expert 1.39 GB はファイルのページ。12K の decode の wired は 13 GB 台なので、memlog 越しに Swapouts を見る。

### 8-4. 落とし穴 (MTP 固有、先に知っておくこと)

- 巻き戻しが要る状態: KV (カーソルを戻す)、GDN の `linState` / `linHist` (T=2 の 1 トークン目の後の値が要る。T < 16 は `q38_gdn_step` の直列なので途中を退避できる)、インデクサのブロック鍵 (4 トークン目で確定するので、棄却されたトークンで確定したブロックは作り直し)、PLE の `plePrev` / `pleHist`、MTP 層自身の KV。
- ドラフトの注意は密 (参照実装)。12K では MTP 層の KV もプロンプト全体ぶん要る (prefill 中に MTP 層を回すかどうかは参照実装を読んで決める)。
- `forward` の戻り値のポインタは次の forward まで (§7)。
- Q4_K / MXFP4 のビット配置は gguf-py / ggml の `dequantize_row_*` で確かめる (06 の Q2_K で取り違えかけた、06 §1)。

