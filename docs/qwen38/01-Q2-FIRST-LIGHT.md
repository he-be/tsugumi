# 01. Q2 検証の初回点灯 — DS4-IQ2 GGUF を Tsugumi の Metal で直接回し、CPU 参照と全位置一致

実測: 2026-09-13、M3 Pro 18 GB (この MBP が唯一の参照機)、macOS 15。
対象: `ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2` (41.73 GiB、sha256 一致) + PLE Q4_1 サイドカー (29.80 GiB、`…-DS4-Q4` から)。
表記は他の文書と同じ (実測 / 導出 / 未確認)。**速度の数字は最適化前の正しさ優先の実装で、上限ではない。**

## 0. 結論

1. **低ビット (IQ2_XXS / Q2_K) の routed expert は Metal で遅くならない。**同じ D=2560 / F=640 で
   Tsugumi の affine int4 g64 カーネルと並べると、1 層の GPU 時間は **0.69 倍** (k=8 同士、常駐・I/O 無し)。
   読むバイトが 2.76 → 1.49 MB/expert に減る効果が、コードブック展開の計算増より大きい (§2)。
2. **qwen4exp の CPU 参照器ができた。**ds4-metal `ds4.c` の `qwen4_ref_*` (GGUF をそのまま読む parity oracle) を numpy に写し、
   "The capital of France is" → " Paris." を出す。footprint 1 GB 以下、19 秒/トークン (§3)。
3. **Metal ランナーが参照と全位置一致した。**`Qwen38Runner` (GGUF を mmap、`.moepack` への詰め替え無し) が
   2 本のプロンプト (5+5 位置、53 位置) で **top-1 が 63/63 一致**。最適化前で約 1 秒/トークン (§4)。
4. **PLE の負例は通った。**53 トークンの英文で平均 NLL は PLE あり 0.928 / 抜き 1.028、抜いた方が 52 トークン中 36 で悪い (§3-2)。
5. **logits も数値で一致した。**10 位置で相対誤差 ≤ 1.7e-6 (48 層を通した float32 の床) (§4-2)。
6. **QSA indexer (32K に必須) も一致した。**予算を両側で 16 トークンに上書きした 53 位置で top-1 53/53、logits ≤ 1.41e-6。
   上書きしないランナーは同じ参照から位置 19 で 4e-2、最大 1.7e-1 外れる (負例) (§4-4)。
7. 残り: 速度 (マシンを空けて測る)、prefill、MTP (§5)。

## 1. 前提の変更 (ユーザー指示、2026-09-13)

- 参照機はこの MBP だけ。**DS4 も llama.cpp も推論には使わない。**ds4-metal はカーネルと CPU 参照の算式を読む資料としてだけ使う。
- llm-server では新規 DL しない。ウェイトは DS4-IQ2 (Q2) を本線にし、Jundot oQ4e (106 GB) は引かない。
- [QWEN38_FLASH_NEXT_VERIFY_PLAN.md](../investigations/QWEN38_FLASH_NEXT_VERIFY_PLAN.md) の「oQ4e を引いて affine で打ち直す」本線と §2-4 の「DS4-IQ2 は取らない」はこの時点で置き換わる。

### 1-1. 運用点 (ユーザー指定、同日)

- **thinking 無効**、**文脈 32K**、**MTP は使えるなら使う**。即答してツールを呼ぶ**エージェント動作が主体**で、thinking 有効の長考コーディングは主用途ではない。
- **英語主体** (日本語力は期待しない)。品質は英語の短答 + ツール呼び出しで見る。
- 帰結: 32K は indexer の予算 (512 ブロック = 2048 トークン) の外なので QSA indexer が必須。ツール結果を読む長い prefill の速度が効く。
  MTP ブロック (blk.48) は gate/up Q4_K・down MXFP4 なので、そのカーネルも要る。

## 2. routed expert カーネル (`moe_ggml.metal`)

ds4-metal (MIT) の `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` と `kernel_glm_q2_K_addr_down_f32` の行演算を、
Tsugumi の decode の呼び方 (expert blob の引数バッファ、phase 2 で routing weight と残差を畳む) に載せ替えた。
IQ2_XXS のブロックは ds4 と gguf-py (`quants.py`) で同じ形 (66 B / 256 重み、32 重みごとに 4 コードブック番号 + 符号 + 4-bit スケール)。

### 2-1. 正しさ (実測、`TsugumiKernelCheck --q2-expert`)

`Scripts/qwen38/expert_kernel_fixture.py` が第 24 層から実物の expert 10 個 (x を実際の router に通した top-10) を切り出し、
gguf-py で逆量子化した float64 を正解にする。

| | 相対誤差 (max) |
| --- | ---: |
| acts = silu(gate x) · (up x) | 1.2e-7 |
| y = residual + Σ w · down(acts) | 8.0e-8 |
| 何もしない (y = residual) の場合 | 4.0e-1 |

### 2-2. 速度 (実測、`--q2-expert-bench 20`、half 活性の版で測った)

48 層ぶんを 1 本のコマンドバッファに積み、20 本の中央値。ABBA (int4, q2, q2, int4)。int4 側は合成ウェイト、
どちらも形の function constant 無しの汎用 PSO。

| 1 層の routed expert | affine int4 g64 | IQ2_XXS + Q2_K | 比 |
| --- | ---: | ---: | ---: |
| gate/up (k=8) | 0.112 ms | 0.075 ms | 0.67 |
| down (k=8) | 0.070 ms | 0.048 ms | 0.69 |
| 1 層 (k=8) | 0.184 ms | 0.127 ms | 0.69 |
| 1 層 (k=10) | — | 0.156 ms | — |

活性は後で float32 に変えた (§4 のランナーに合わせるため)。この表はその前の half 版の値で、float32 版では測り直していない (**未確認**)。

## 3. CPU 参照器 (`Scripts/qwen38/reference_forward.py`)

算式と重みの慣習は `ds4.c` `3030554` の 66149 行〜 (`qwen4_ref_*`) に従う:
ノルムの gamma は ssm_norm 以外 1+w 焼き込み済み、`ssm_a` = −exp(A_log)、**GDN の value head j は key head j % 16 と組む** (巡回。
HF の `repeat_interleave` はブロック対応だが、GGUF 変換で並べ替え済み)、残差は 2560 × 4 本、PLE は第 1 層の入口、最終ノルムは output_hc の mixer が兼ねる。
文脈 2048 以下では QSA indexer が全トークンを選び密な注意と一致するので、indexer は実装していない (超えたら止まる)。
上流の仕様は transformers main の `modeling_qwen4_exp.py` で照合した。

### 3-1. 生成 (実測)

"The capital of France is" (5 トークン) → greedy ` Paris.\n\n<think>\nThe`。19 秒/トークン、footprint ≤ 1.0 GB、Swapouts 増分 0。

### 3-2. PLE の負例 (実測)

53 トークンの英文 (富士山) で、PLE を足す / 足さない (`--ablate-ple`) の平均 NLL を比べる。
PLE の算式が正しければ、足さない方が悪くなるはず。

| | 平均 NLL (52 トークン) |
| --- | ---: |
| PLE あり | 0.9277 |
| PLE 抜き | 1.0280 |

トークン単位では抜いた方が 52 中 36 で悪く、平均差 +0.100 (実測、プロンプト 1 本)。行番号のハッシュが外れていれば
無関係な行を足すことになり、ここまで一貫して効くことはない、というのが負例の読み方。

## 4. Metal ランナー (`Qwen38Runner`、`TsugumiKernelCheck --qwen38-decode`)

- GGUF を `GGUFFile` で mmap し、dense (Q8_0 / F16 / F32) はテンソルごとの no-copy `MTLBuffer` で読む。
- routed expert は毎トークン、選ばれた 10 個を mmap から 10 個のスロット blob にコピーする (キャッシュ無し)。
- 層の中は GPU (`qwen38.metal`: hyper-connection、GDN 1 ステップ、24 ヘッドの gated attention。`ggml_dense.metal`: Q8_0/F16/F32 GEMV)。
  host に残るのは router の top-10 (logits を読み戻す)、expert のコピー、PLE (16 行の Q4_1 と dilated conv)。

### 4-1. top-1 一致 (実測)

| プロンプト | 位置 | top-1 一致 |
| --- | ---: | ---: |
| "The capital of France is" + greedy 5 | 10 | 10/10 |
| 富士山の英文 53 トークン | 53 | 53/53 |

dense GEMV 単体 (`--ggml-dense`、実物 10 テンソル、LM head 248,320 行を含む) は相対誤差 1e-7 台。

### 4-2. logits の数値一致 (実測)

参照器の `--dump-logits` (10 位置 × 248,320) と `--q38-ref-logits` で位置ごとの max 相対誤差。

| 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 相対誤差 (×1e-6) | 0.68 | 0.92 | 0.90 | 0.81 | 1.61 | 0.91 | 1.71 | 1.48 | 1.43 | 1.44 |

この走行は CPU 参照器が止まった後で、1 トークン 0.24〜0.50 s (初回 1.36 s) だった (n=1 ずつ、数字だけ。§4-3 は参照器と同時に走らせた値)。

### 4-3. 1 トークンの内訳 (実測、n=1 ずつ、CPU 参照器と同時に走らせた状態。数字だけ)

| 段 | ms (10 位置の範囲) |
| --- | ---: |
| PLE (host) | 20〜56 |
| pre-router (48 層の encode + GPU 待ち) | 277〜511 |
| route (host top-10 + expert 480 個 715 MB のコピー) | 448〜1,310 |
| routed (expert カーネル、48 層) | 35〜42 |
| head (mixer + LM head) | 7〜29 |
| 合計 | 0.79〜1.91 s |

### 4-4. QSA indexer (実測)

32K は予算 (2048 トークン = 512 ブロック) を超えるので必須。算式は ds4 `qwen4_ref_select`: 4 トークンのブロックの生の鍵を平均し、
k_norm と rope (ブロック先頭の位置) をかけ、score = Σ_h relu(q_h · key)、上位 512 ブロック (同点は小さい番号) と端数のトークンを選ぶ。
ランナーはブロックの鍵を 4 トークン目が入った時点で 1 回だけ計算してキャッシュし、スコアは GPU、選択は host。

照合は予算を両側で 16 トークン (4 ブロック) に上書きし、53 トークンの英文で位置 19 以降に選択を発動させて top-1 と logits を比べる
(参照器 `--indexer-top-k 16 --dump-logits`、ランナー `--q38-indexer-top-k 16`)。

| ランナー | 位置 18 | 位置 19 (選択が始まる) | 位置 30 | 位置 52 | top-1 | 最悪 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 予算 16 (参照と同じ) | 6.3e-7 | 7.5e-7 | 1.1e-6 | 1.4e-6 | 53/53 | 1.41e-6 |
| 予算 2048 (負例) | 6.3e-7 | 4.2e-2 | 8.8e-2 | 1.3e-1 | 52/53 | 1.68e-1 |

参照器の予算 16 の平均 NLL は 0.9282 (予算 2048 では 0.9277)。この文の長さでは選択を 4 ブロックに絞ってもほとんど動かない (プロンプト 1 本、数字だけ)。

途中で 1 つ直した: ランナーで `pos + 1` が 4 の倍数のとき端数トークンの範囲が空になり、閉区間 `(nBlocks*4)...pos` が trap していた。

## 5. 次 (運用点 §1-1 に合わせた順)

1. 速度: expert は Tsugumi の本線と同じ「expert ごとの no-copy バッファ + residency set」(`docs/mtp/48`・`49`) にしてコピーを消し
   (no-copy ビューまでは実装済み、正しさは同じ logits で確認。速度はまだ空いた状態で測っていない)、
   スロットのキャッシュを入れる。pre-router はマシンを空けて測り直してから手を付ける。PLE を GPU に移す。**prefill (複数トークンの一括) を測る。**
2. MTP (blk.48: Q4_K gate/up、MXFP4 down のカーネル)。
3. 品質: ランナーを server に繋ぎ、thinking 無効・英語のツール呼び出しタスクで見る。

## 6. 足したもの

| 種類 | パス |
| --- | --- |
| 参照器 | `Scripts/qwen38/reference_forward.py`、`Scripts/qwen38/expert_kernel_fixture.py` |
| GGUF | `Sources/Tsugumi/Infrastructure/ModelIO/GGUFFile.swift` |
| カーネル | `Sources/Tsugumi/Metal/MoE/moe_ggml.metal`、`Sources/Tsugumi/Metal/Quant/ggml_dense.metal`、`Sources/Tsugumi/Metal/Qwen/qwen38.metal`、`Sources/Tsugumi/Kernels/Qwen38/GGMLDenseGEMV.swift` |
| ランナー | `Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift` |
| 検査 | `TsugumiKernelCheck --q2-expert`、`--ggml-dense`、`--qwen38-decode` |
| 帰属 | `THIRD_PARTY_NOTICES.md` に ds4-metal (MIT) |

## 7. 再開手順 (新しいセッションで続けるとき)

### 7-1. 手元にあるもの

| もの | 場所 | 備考 |
| --- | --- | --- |
| 本体 GGUF (41.73 GiB) | `~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf` | sha256 照合済み。ランナーと参照器の既定パス |
| PLE サイドカー (29.80 GiB) | `~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/ple/Qwen3.8-Flash-Next-PLE-Q4_1.gguf` | 同上 |
| トークナイザ | `~/LLM/Qwen3.8-Flash-Next-tokenizer/` (`tokenizer.json`、`chat_template.jinja` ほか) | 上流 `Qwen/Qwen3.8-Flash-Next` から。重みではない |
| 参照ログと logits | `scratch/qwen38/ref-france-dump.log` + `ref-france.logits` (10 位置)、`ref-fuji-ple.log` (53 位置、logits 無し)、`ref-fuji-idx16.log` + `ref-fuji-idx16.logits` (予算 16) | `scratch/` は git 管理外。消えたら §7-3 で作り直す (1 本 3〜17 分) |
| expert カーネルの fixture | `scratch/qwen38/expert-fixture-l24/` | `Scripts/qwen38/expert_kernel_fixture.py` で再生成 |
| 仕様の資料 | ds4-metal `origin/qwen3.8-flash-next` (`~/LLM/ds4-metal`、`git show origin/qwen3.8-flash-next:ds4.c` の 66149 行〜が CPU 参照)、transformers main の `models/qwen4_exp/modeling_qwen4_exp.py` | ds4 は**読むだけ**。推論には使わない (§1) |
| 途中で止めた不要物 | `~/LLM/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/` (26 GiB、未完のコピー) | Q2 検証には不要。消すかはユーザー判断待ち |

### 7-2. 検査を一通り回す (どれも数分以内、GPU 1 本ずつ)

```bash
swift build -c release --product TsugumiKernelCheck
B=.build/release/TsugumiKernelCheck
G=~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf
$B --q2-expert scratch/qwen38/expert-fixture-l24            # expert カーネル (+ --q2-expert-bench 20 で int4 比)
$B --ggml-dense $G                                           # dense GEMV
$B --qwen38-decode scratch/qwen38/ref-france-dump.log --q38-ref-logits scratch/qwen38/ref-france.logits
$B --qwen38-decode scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16
```

期待値は §2-1、§4-1、§4-2、§4-4 の表。`--qwen38-decode` の各行の `(ple … pre … route … routed … head … ms)` が 1 トークンの内訳。

### 7-3. 参照を作り直す

```bash
~/LLM/venv/bin/python Scripts/qwen38/reference_forward.py --text "The capital of France is" --new 6 \
    --dump-logits scratch/qwen38/ref-france.logits > scratch/qwen38/ref-france-dump.log
~/LLM/venv/bin/python Scripts/qwen38/expert_kernel_fixture.py --gguf <本体 GGUF> --layer 24 --out scratch/qwen38/expert-fixture-l24
```

参照器は 19 秒/トークン、footprint ≤ 1 GB。**footprint 5 GB 超か Swapouts +64 MB で自分で止まる**。
走らせている間は CPU を食うので、**ランナーの速度はその間に測らない** (§4-3 の値はそれで汚れている)。

### 7-4. 次にやること (§5 の順)

1. **速度**: ユーザーに重いアプリを閉じてもらってから。まず今の no-copy 版で 1 トークンの内訳を取り、
   コピー版と A/B する。コピー版は**コミットしていない** (同じセッション内で no-copy に置き換えた) ので、
   `moeRouted` で選ばれた 10 個の gate/up/down を mmap から 10 本の連続 blob に `memcpy` し、`part_off` を (0, gate, 2·gate) にする形で作り直す
   (`Q2ExpertCheck.swift` の連続 blob の組み方と同じ)。空いた状態のコピー版の参考値は 0.24〜0.50 s/トークン (§4-2 の走行、n=1 ずつ)。
   その上で Tsugumi 本線と同じ **層ごとの `MTLResidencySet` + ミスへの `F_RDADVISE`** (`Sources/Tsugumi/Infrastructure/Streaming/MmapExpertMapping.swift`、`docs/mtp/48`・`49`・`52`) を入れる。
   pre-router (48 層の encode と同期) は GPU 時間と host 時間を分けて測ってから手を付ける。
2. **prefill**: 32K の運用点ではツール結果の読み込みが効く。今のランナーは 1 トークンずつしか流せない。
3. **MTP**: blk.48 は gate/up Q4_K・down MXFP4 (ds4 の `moe.metal` にカーネルがある)。
4. **品質**: server に繋ぎ、thinking 無効・英語のツール呼び出しで見る。チャットテンプレートは `~/LLM/Qwen3.8-Flash-Next-tokenizer/chat_template.jinja`。

### 7-5. 落とし穴 (このセッションで踏んだもの)

- **GDN の head の組み方**: GGUF は value head j ↔ key head **j % 16** (巡回)。Tsugumi の既存 `gdn.metal` は `hv / ratio` (ブロック) で、流用すると落ちずに静かに間違う。
- **QSA**: 文脈 2048 までは全選択で密な注意と同じなので、短い検査では indexer を通らない。**予算を上書きしないと検査にならない。**
- **ノルム**: GGUF は 1+w 焼き込み済み、`ssm_norm` だけ素の w。HF の式をそのまま写すと二重に 1 が足される。
- **大きなファイルの取得**: `hf download` (xet) は DL 量に比例して RSS が膨らみ、再開もしない。`curl -L -C -` で 1 本ずつ (メモリ `hf-xet-download-memory`)。
- **運用点**: thinking 無効・32K・MTP 可なら使う・ツール即答のエージェント主体・英語主体 (メモリ `qwen38-operating-point`)。
