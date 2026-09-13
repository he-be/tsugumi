# 10. MTP ヘッドを実装して、素の decode・影モード・投機ループ (n_max 1) を実測する

実測: 2026-09-14、M3 Pro 18 GB、macOS 15。対象は [09](09-12K-MEMORY-VERIFY.md) と同じ GGUF・ランナー。
表記は 01 と同じ (実測 / 導出 / 未確認)。**速度・受理率の行はどれも n=1 の走行で、数字だけを書く。**
§1〜§4 の走行は腕を交互にしていない逐次の走行なので、腕の間の速さの差は読まない (比較は §5)。

09 §5 の 1〜5 の順に進めた。

## 0. 結論

1. **MTP ヘッド (blk.48) を実装し、CPU 参照と一致した** (§2)。11 位置で top-1 が全部一致、logits の相対誤差は最大 1.3e-5。
   routed expert は Q4_K (gate/up) と MXFP4 (down) を GPU で float32 に展開して sgemm で掛ける。注意は密。
2. **素の decode** (§1): 短文脈で steady な 1 トークン 135〜176 ms (中央値)、12K の直後で 190 ms。
3. **影モードの受理率** (§3、ドラフトの argmax == target が引いたトークン):
   短文脈で greedy 0.789〜1.000、instruct 0.770〜0.974。12K で greedy 0.869、instruct 0.794。
   **ドラフト 1 回は steady で 14〜15 ms (短文脈)、18〜19 ms (12K)。**MTP の prefill は 11.7K トークンで 11.2〜11.4 s (幹は 132.6〜133.1 s)。
4. **投機ループ (n_max 1) は中立だった** (§4)。短文脈 3 本 (greedy) で、素の decode・全部棄却 (毎ステップ巻き戻し)・投機のトークン列が一致し、
   素と全部棄却では選んだトークンの logit が全ステップ 6 桁まで一致した。巻き戻しは 0.0〜0.2 ms。
5. 端から端の速さ (腕を交互) は §5。

## 1. 素の decode (09 §5-1)

`--qwen38-generate <prompt> --q38-new 200 --q38-mtp off`。プロンプトは `Scripts/qwen38/chat_prompts.py` (§6) で、
上流の chat_template を thinking off で描いた英語のもの。短文脈はチャンク 512、12K はチャンク 2048。
greedy は argmax、instruct は公式の非 thinking 設定 (temp 0.7・top_p 0.8・top_k 20・presence 1.5 を生成したトークンに、HF の順)。
steady は最初のステップ (prefill の直後で dense を読み戻す、09 §2-1) を除いた中央値。

| プロンプト | トークン | サンプラ | 生成 | 最初 ms | steady 中央値 (平均) ms | steady tok/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| code | 80 | greedy | 199 | 1,467 | 176 (176) | 5.68 |
| code | 80 | instruct | 199 | 1,404 | 135 (144) | 6.96 |
| tool | 468 | greedy | 84 (EOS) | 1,700 | 171 (179) | 5.59 |
| tool | 468 | instruct | 39 (EOS) | 1,685 | 157 (165) | 6.07 |
| explain | 47 | greedy | 199 | 1,338 | 158 (168) | 5.94 |
| explain | 47 | instruct | 187 (EOS) | 1,216 | 141 (153) | 6.54 |
| long (12K) | 11,683 | instruct | 199 | 2,033 | 190 (193) | 5.19 |

12K の prefill は 133.7 s、wired の最大 12.47 GB、file-backed の最小 3.52 GB、Swapouts 0。
tool は read_file と run_shell のツール呼び出しを書いて `<|im_end|>` で止まった (greedy)。

## 2. MTP ヘッド

### 2-1. 算式 (参照実装: PC の llama.cpp unsloth-mtp `a9e9c3c`、`src/models/qwen4exp.cpp` 489〜660 行)

- 入力: 位置 p のトークン x_p と、幹の最後の層を出た残差 h_{p-1} (hc 4 本 × 2560、`output_hc` のミックスの前)。p = 0 は h がゼロ (llama.cpp の最初の `pending_h`)。
  出力は x_{p+1} の予測。
- x_p の埋め込み (BF16) → `nextn.enorm` の RMS → 4 本に複製。h は stream ごとに RMS → `nextn.hnorm` [10240]。
  **stream ごとに [e, h] (5120) を連結**して `nextn.eh_proj` (Q8_0 5120 → 2560) を 4 行ぶん掛ける。
- 以降は幹の 1 層と同じ形: `hc_attn` ミックス → 注意 → combine → `hc_ffn` ミックス → shared + routed → combine。
  注意は gated で、**QSA なしの密** (参照実装のコメント「QSA は文脈を刈るだけ」)、MTP 層自身の KV を位置で持つ。
- combine 後の残差を `mtpHidden` に残し (連鎖ドラフトの入力)、`nextn.hc_head` ミックス → トランクの `output.weight`。
- 投機ドライバ (`common/speculative.cpp` の `draft-mtp`) は、target のバッチ (prefill のチャンク・検証バッチ) を処理するたびに、
  同じトークンを 1 つ右にずらした h と組んで MTP に通して KV を埋める。ドラフトは (最後に引いたトークン, 直前位置の h) の 1 行。

### 2-2. 実装 (`Qwen38Runner`)

- `exportHidden`: `forward` が最後の残差の全行を `hidden(row:)` に写す (T=2048 で 84 MB)。
- `mtpForward(tokens:startPos:hidden:allLogits:)`: 上の算式。norm は GPU、[e, h] の連結は host。
- 注意は `attention(dense: true)`: インデクサを飛ばし、`attentionHost` を予算 = capacity で呼ぶ (T < 32 は host レーン、T ≥ 32 は sgemm)。
  `nCap` (host 経路の行の幅) を呼び出しごとに設定するようにした (MTP 層は文脈全体を見るので、最大値を持ち越すと幹の一時領域も広がる)。
- routed: `moe_q4k_dequant_gate_up_f32` / `moe_mxfp4_dequant_down_f32` (`moe_ggml.metal`、既存の IQ2_XXS / Q2_K 展開と同じ呼び出し形) を足し、
  MTP 層では T によらず展開 + sgemm (`routedGemm`) に回す。ビット配置は gguf-py の `dequantize_blocks` (Q4_K の `get_scale_min`、MXFP4 の `e8m0_to_fp32_half` と kvalues) に合わせた。
- `routedGemm` の一時領域を `maxBatch` でなく `batchRows` で確保するようにした (decode の T=1 で 2048 行ぶんを作らない)。

### 2-3. 検査

`--qwen38-mtp-dump scratch/qwen38/prompts/code.tokens <out>`: 幹が 80 トークンを 1 バッチで通し、MTP が同じトークンを
48・1・1・2・3・1… のバッチで通す (sgemm と host レーンの両方、KV の継続を通る)。入力行と全行の logits を書き、
`Scripts/qwen38/mtp_reference.py` (reference_forward.py の部品で同じ算式を 1 トークンずつ) が同じ入力で計算して比べる:

| 位置 | 0 | 1 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 79 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rel err | 1.42e-6 | 1.29e-5 | 8.50e-6 | 1.38e-6 | 4.26e-6 | 8.60e-7 | 3.45e-6 | 2.74e-6 | 5.40e-6 | 4.00e-6 | 1.67e-6 |

top-1 は 11 位置とも一致、PASS。参照は同じ読み (llama.cpp のグラフ) を写したものなので、読み違いは両方に入る。
それを外から確かめるのが §3 の受理率 (PC / llm-server の実測 0.83〜0.94 と同じ水準か)。

## 3. 影モード (09 §5-3)

`--q38-mtp shadow`: prefill のチャンクごとに MTP も通し (256 トークンずつ、`Q38_MTP_CHUNK`)、decode の各ステップで
幹の forward の前に (これから入れるトークン, 直前位置の h) でドラフトし、捨てる。受理 = ドラフトの argmax == 幹の logits から引いたトークン。
影モードの 7 組 (短文脈 6 + 12K instruct 1) で、出たトークン列は素の decode と一致した。

| プロンプト | サンプラ | 受理 | 率 | ドラフト steady 中央値 (平均) ms | ドラフト最初 ms | 幹 steady 中央値 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| code | greedy | 187 / 199 | 0.940 | 14 (14) | 20 | 142 |
| code | instruct | 182 / 199 | 0.915 | 14 (14) | 20 | 135 |
| tool | greedy | 84 / 84 | 1.000 | 14 (14) | 21 | 141 |
| tool | instruct | 38 / 39 | 0.974 | 14 (14) | 29 | 150 |
| explain | greedy | 157 / 199 | 0.789 | 14 (14) | 21 | 137 |
| explain | instruct | 144 / 187 | 0.770 | 15 (15) | 19 | 135 |
| long (12K) | greedy | 173 / 199 | 0.869 | 18 (18) | 84 | 157 |
| long (12K) | instruct | 158 / 199 | 0.794 | 19 (19) | 100 | 160 |

ドラフト 1 回の内訳 (steady 中央値 ms): 短文脈 wall 14 = pre 2 [GPU 1]・route 1・routed 5 [GPU 4]・head 5。
12K は wall 18〜19 = pre 6〜7 [GPU 5]・route 1〜2・routed 5 [GPU 4]・head 5 (密な注意が 11.7K を見る分が pre に出る)。

12K の prefill: 幹 133.1 / 132.6 s、MTP 11.4 / 11.2 s (チャンク 2048 ごとに 1.7〜2.1 s)。
wired の最大 13.34 / 13.18 GB、file-backed の最小 2.75 / 2.83 GB、Swapouts 0 (memlog 越し)。

## 4. 投機ループ n_max 1 (09 §5-4)

### 4-1. ループ

`--q38-mtp spec`: 1 ステップ =
1. MTP: 前のステップで受理した行 (受理なら [d, h_pos]) + [y, h_{pos'-1}] を 1〜2 行で通し、最後の行の argmax を d とする。
2. 幹: [y, d] を T=2 で通す (`allLogits`)。1 行目から t0 を引く。
3. t0 == d なら受理: 2 行目から t1 を引き、2 トークン進む。違えば t0 だけ出し、`rollbackToFirst()` で状態を「y だけ入れた後」に戻す。

受理判定は llama.cpp と同じ「target のサンプル == ドラフトの argmax」なので、instruct でも出るトークンの分布は素の decode と同じ
(乱数の消費もトークン 1 つに 1 回で同じ)。

### 4-2. 巻き戻し

位置で持つもの (幹の KV、インデクサの生の鍵とブロック鍵、MTP の KV) は、次の forward が同じ位置を書き直すので何もしない
(ブロック鍵は 4 トークン目を含むバッチで作り直される、`attention` の `firstBlock`)。戻すのは再帰的な状態だけ:

- GDN の状態: `q38_gdn_step` に `snap` / `snap_t` を足し、1 トークン目の後の状態を別バッファに書く。巻き戻しはバッファの入れ替え (コピーなし)。
- GDN の conv 履歴: forward の前に 36 層ぶん host にコピー (1 層 123 KB)、各層の qkv の 1 行目を blit で保存。戻すときは 1 行ずらして qkv を足す。
- PLE の conv 履歴と `plePrev`: 前の値を host に持ち、1 行ずらして 1 トークン目の `pleNormed` を足す。

`snapshotFirst` を立てた forward (T が 2 以上 `gdnMinTokens` 未満、直列の GDN step) だけが保存する。

### 4-3. 中立性 (短文脈 3 本、greedy、200 トークン)

`--q38-mtp reject` は同じ検証 (T=2) を回して毎ステップ棄却する対照。

| プロンプト | 素: steady tok/s (中央値 ms) | 全部棄却: steady tok/s、1 ステップ = ドラフト + 検証 + 巻き戻し ms | 投機: ステップ・受理・トークン/ステップ・steady tok/s、1 ステップ ms |
| --- | --- | --- | --- |
| code | 6.15 (162) | 5.74、167 = 13 + 155 + 0.1 | 103・96 (0.932)・1.93・8.98、208 = 19 + 188 |
| tool | 5.92 (164) | 5.84、166 = 13 + 153 + 0.1 | 42・42 (1.000)・2.00・8.69、218 = 19 + 201 |
| explain | 6.46 (148) | 5.67、167 = 14 + 153 + 0.2 | 112・87 (0.777)・1.78・8.57、197 = 19 + 179 |

- 3 本とも **素・全部棄却・投機のトークン列が一致**。
- 素と全部棄却では、各ステップで選んだトークンの logit (6 桁で出力) が全ステップ一致 (199・84・199 ステップ)。
  T=2 の検証の 1 行目は T=1 と区別がつかず、毎ステップの巻き戻しの後も続く。
- 投機のドラフトが 19 ms なのは、受理の後は 2 行 ([d, y]) で通すため。
- この表の 3 つの腕は逐次で回したので、腕の間の速さは §5 で見る。

## 5. 端から端の比較 (09 §5-5)

(走行中。腕を ABBA で交互、instruct、seed = 反復番号、200 トークン、短文脈 3 本と 12K 3 本 × 反復 2。)

## 6. 足したもの・変えたもの

| 種類 | パス |
| --- | --- |
| ランナー | `exportHidden` / `hidden(row:)`、`mtpForward` / `mtpHidden(row:)` / `lastMTPProfile`、`attention(dense:)`、`nCap` を呼び出しごとに、`routedGemm` の展開カーネル引数と `batchRows` での確保、`sharedAndRouter` / `resizeBatch` / `embed` (切り出し)、`snapshotFirst` / `rollbackToFirst` |
| Metal | `moe_q4k_dequant_gate_up_f32`、`moe_mxfp4_dequant_down_f32` (`moe_ggml.metal`)、`q38_gdn_step` の `snap` / `snap_t` |
| 検査 | `--qwen38-mtp-dump` (+ `Scripts/qwen38/mtp_reference.py`)、`--qwen38-generate` (`--q38-mtp off|shadow|spec|reject`、`--q38-sampler greedy|instruct`、`--q38-seed`) (`Qwen38MTPCheck.swift`) |
| スクリプト | `Scripts/qwen38/chat_prompts.py` (プロンプトの描画と符号化、生成列の復号) |
| スクリプト (git 管理外) | `scratch/qwen38/prompts/` (トークン列)、`gen10.sh`・`spec10.sh`・`e2e10.sh` と各出力、`long-gen10.tokens` (§1・§3 の 12K で使った列。long はソースから作るので、以降の編集で中身が変わる) |

## 7. 落とし穴

- **`mtpForward` と `forward` は同じ `logits` バッファを返す。**幹の logits を持ったまま MTP を回すと上書きされる。呼び出し側で配列にコピーする (09 §7 と同じ規則)。
- **MTP の入力の h は 1 つ前の位置のもの。**行 p は (x_p, h_{p-1}) で x_{p+1} を予測する。prefill ではチャンクの頭の行が前のチャンクの最後の h (`pendingH`) になる。
- 密な注意は T ≥ 32 で T × 12 × n の scores を持つ。12K で MTP の prefill を 2048 行で通すと 1.2 GB になるので 256 行ずつ。
- `snapshotFirst` は直列の GDN step (T < `gdnMinTokens`) でしか効かない (precondition)。
- zsh では `${VAR:-a b c}` が単語に分かれない (`${=VAR:-a b c}`)。spec10 の最初の走行はこれで 1 本も回らなかった。
- Metal のカーネル関数の中に `static constant` の表は置けない (ファイルの外側に置く)。
