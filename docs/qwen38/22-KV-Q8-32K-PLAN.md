# 22. KV を Q8_0 にして 32K を今のメモリの点に収める (計画)

書いた日: 2026-09-15。対象は [21](21-TOOL-LOOP-TRIAGE.md) の後の Qwen3.8 経路 (ランナー `Qwen38Runner.swift`、カーネル `Metal/Qwen/qwen38.metal`、参照器 `Scripts/qwen38/reference_forward.py`)。
表記は 01 と同じ (実測 / 導出 / 未確認)。この文書は計画で、§1 以外の数字はすべて導出。実測は 23 以降に書く。

## 0. 結論

1. **12K は物理の壁ではない。**文脈に比例して伸びるメモリは KV・インデクサ鍵・MTP 層の KV だけで、それを全部 float32 で持っている (`Qwen38Runner.swift:636` の `capacity * Hkv * D * 4`)。
   1 トークン 60.9 KB、12K で 0.75 GB、32K で 2.00 GB。今の 12K の運用点 (wired 14.3〜14.6 GB、file-backed 1.4〜1.65 GB、[16 §0](16-WEIGHT-DIET.md)) に float32 のまま +1.25 GB は入らない (prev10 / n_max 2 で打ち切った点と同じ、13・14)。
2. **K・V を Q8_0、インデクサ鍵を F16 にすると 1 トークン 18.0 KB、32K で 0.59 GB。今の 12K (0.75 GB) より 0.16 GB 少ない。**メモリの点を動かさずに 32K に行ける。
3. 文脈に比例するほかの費用は小さい: prefill の注意の一時領域は QSA 予算 (2,048 トークン) で頭打ち、インデクサのスコア `idxScores` は 32K で 67 MB、decode の速さは QSA のため文脈にほぼ依らない (09 §3-1: 4K 166 ms、12K 182 ms)。伸びるのは MTP 層の密な注意 (ドラフト 1 回に prefix 全部を読む) だけで、これは測る (§5)。
4. **Gemma で同じ会話に問題が出なかったのは、Gemma が 32K で動いているから** (`AppModelKind.defaultContextTokens`、KV は FP16、sliding-window 層はリング)。32K は Gemma と同じ土俵に戻す手で、A-2 (履歴のツール結果、21 §9) を直す手ではない。両方要る。
5. 順序: **F32 / Q8_0 を実行時に選べる形で Q8_0 を入れる → 12K で一致検査と ABBA → 32K の prefill・decode・ツールループ → 既定を Q8_0・32K に**。F16 を経由しない (Q8_0 と同じ箇所を触るので、二度やる理由が無い)。

## 1. 現状の数字 (コードと 09・16 の実測から)

注意層は 48 層のうち `full_attention_interval` = 4 で 12 層 (`isLinear`)、H = 24、Hkv = 2、D = 256。インデクサは idxHeads = 4、idxD = 128、ブロック 4 トークン。MTP 層 (blk.48) は密な注意で KV を持つ。すべて float32 の `storageModeShared`。

| 中身 | 1 層 1 トークン | 層 | 1 トークン | 12,288 | 32,768 |
| --- | ---: | ---: | ---: | ---: | ---: |
| K + V (float32) | 4,096 B | 12 | 49,152 B | 604 MB | 1,611 MB |
| インデクサ raw 鍵 (float32、capacity 行) | 512 B | 12 | 6,144 B | 75 MB | 201 MB |
| インデクサ block 鍵 (float32、capacity/4 行) | 128 B | 12 | 1,536 B | 19 MB | 50 MB |
| MTP 層 K + V (float32) | 4,096 B | 1 | 4,096 B | 50 MB | 134 MB |
| **合計** | | | **60,928 B** | **749 MB** | **1,997 MB** |

09 §1 の計器 (capacity 8,194 で「KV + インデクサ鍵 466 MB」) と、MTP 層を除いた 56,832 B × 8,194 = 466 MB が一致する。

文脈に比例するほかのバッファ: `idxScores` = maxBatch × (capacity/4 + 1) × 4 B (2048 × 8,193 × 4 = 67 MB @32K)、`selection` = T × nCap × 4 B (nCap ≤ 予算 + 3 なので 67 MB @T = 2048)、`attnScores` = T × G × nMax × 4 B (nMax ≤ 予算なので約 200 MB @T = 2048、09 の 201 MB)。`attentionSgemm` の `attnKG` / `attnVG` は n × D で n ≤ 予算。**どれも 32K で増えない。**

## 2. 設計

### 2-1. 保持の型

| 中身 | 型 | 1 トークン | 32K |
| --- | --- | ---: | ---: |
| K・V (12 層) | Q8_0 (ggml の `block_q8_0`: half d + int8 × 32、34 B / 32 要素) | 12 × 2 × 8 ブロック × 34 B = 13,056 B | 428 MB |
| MTP 層 K・V | Q8_0 | 1,088 B | 36 MB |
| インデクサ raw 鍵 | F16 | 12 × 256 B = 3,072 B | 101 MB |
| インデクサ block 鍵 | F16 | 12 × 64 B = 768 B | 25 MB |
| **合計** | | **17,984 B** | **589 MB** |

- 1 行 (1 トークン × 1 kv ヘッド、256 要素) = 8 ブロック = 272 B。行の並びは今と同じ `[pos][hkv][D]`。
- **K は RMS-norm + RoPE の後に量子化する** (llama.cpp の K cache と同じ。`q38_attn_prep` が今 in place で掛けている、`qwen38.metal:467`)。V は gemv の出力をそのまま量子化。
- インデクサ鍵は選択にしか使わない (値には入らない) が、Q8 だと平均 → RMS → RoPE の前に誤差が入って選択が変わる。F16 なら 32K で 126 MB で足りるので F16 に留める。raw 鍵を capacity 行から「直近の maxBatch + 4 行のリング」に縮める案は、prompt cache の巻き戻し (18・19) で部分ブロックの raw 鍵が消えるので、この段では**やらない** (§6)。
- Q4 系は K 側の劣化が知られている (llama.cpp の `-ctk q4_0` の経験則、このモデルでは未確認)。Q8_0 で足りる (32K で 589 MB) ので候補から外す。

### 2-2. 型の切り替え

`Q38_KV_TYPE=f32|q8_0` (既定は検証が終わるまで `f32`)。Metal は function constant で K/V の読み書きを分け、ランナーは行のバイト数 (`kvRowBytes`) と PSO の組を型で持つ。ABBA の腕にするために両方を残す。既定を `q8_0` にした後も F32 は検査 (`--q38-compare-logits` の対照) のために残す。

### 2-3. 触る箇所

| 箇所 | 今 | 変更 |
| --- | --- | --- |
| `attention()` の K・V の gemv (`:641-642`) | `y: kc / vc` に float を直接書く | `y: kTmp / vTmp` (T × Hkv × D の float)。V は `q38_kv_quantize` で vc へ。K は次の行 |
| `q38_attn_prep` (`qwen38.metal:467`) | kcache を読んで RMS + RoPE、in place | kTmp を読み、q (float) と kc (Q8_0) を書く。`compareSelected` の K 行の退避 (`:679`、`rowBytes = Hkv * D * 4`) は kTmp を退避する形に |
| `q38_attn_score` (`:516`) / `q38_attn_mix` (`:629`) | kcache / vcache を float で読む (decode の host lane、MTP の密な注意) | ブロックを dequant して読む。1 レーンが dims l, l+32, … を読むので、レーン l はブロック k の要素 l を読む形。d はブロックごとに 1 回 |
| `q38_attn_gather_kv` (`:602`) / `q38_attn_gather_kv_list` (`:840`) | cache → float の attnKG / attnVG | dequant しながら書く (sgemm 側は float32 のまま) |
| `attentionHost(dense:)` (MTP) | 上と同じカーネル | 同じ変更で足りる |
| `q38_idx_block_key` (`:660`) / `q38_idx_score` (`:694`) | raw 鍵・block 鍵を float | half で読み書き。`indexer.k_proj` の gemv は iTmp (float) へ、`q38_kv_convert_f16` で ik へ |
| `memoryReport` (`:1467`) | KV + インデクサ鍵の合計 | そのまま (長さが変わるだけ) |
| `rollback(keep:)` (`:1613`)・チェックポイント (18) | KV は位置ごとで触らない | **変更なし** (行が位置で決まる性質はそのまま) |
| 参照器 `reference_forward.py` | K・V を float32 の配列 | `--kv-type q8_0`: RoPE 後の K と V を Q8_0 に往復 (d = max\|x\| / 127 を fp16 に丸め、q = round(x / d) を int8 に clamp、x' = d × q)。インデクサ鍵は fp16 に往復 |
| 検査 CLI | `--qwen38-decode` / `--qwen38-prefill` / `--qwen38-generate` / `--qwen38-prefill-bench` | 環境変数 `Q38_KV_TYPE` を読む。新しい参照 dump を `ref-*-q8.log / .logits` として作る |
| server | `-c 32768` は `supportedContextSizes` に既にある | 変更なし |
| app | `AppModelKind.contextOptions` の `.qwen38` は `[.fourK, .eightK, .twelveK]` (`AppModelKind.swift:128`) | `.sixteenK`・`.thirtyTwoK` を足す。既定 (`defaultContextTokens`) は §5 の結果を見て 32K に |

丸めは ggml (`quantize_row_q8_0_ref` の `roundf`、0.5 は 0 から遠い側) に合わせ、Metal は `round()`、numpy は `copysign(floor(|x| + 0.5), x)` (`np.rint` は偶数丸めで合わない)。d の fp16 化は両側で同じ (Metal `half`、numpy `astype(float16)`)。**これを合わせないと参照との一致検査が成り立たない。**

## 3. 手順

| 順 | 中身 | 確かめ方 |
| ---: | --- | --- |
| 1 | 参照器に `--kv-type q8_0`。France / Fuji (予算 16) / Fuji PLE の dump を q8 で作り直す (`ref-france-q8.*`、`ref-fuji-idx16-q8.*`、`ref-fuji-ple-q8.log`) | dump の top-1 列が f32 の dump と何個違うかを記録する (数字だけ) |
| 2 | Metal: `q38_kv_quantize`・`q38_kv_convert_f16`、score / mix / gather / gather_list / idx_block_key / idx_score の Q8_0 / F16 読み | 単体: 乱数の K・V 行を量子化 → dequant して、numpy の往復とバイト一致 (`--q38-kv-quant-check`、新設)。1,024 行 × 8 ブロック |
| 3 | ランナー: kTmp / vTmp / iTmp、`Q38_KV_TYPE`、行バイト数、PSO の組、`compareSelected` の退避 | `Q38_KV_TYPE=f32` で 07 §3 の 19 本 (`scratch/qwen38/checks07.sh`) が今と同じ値で PASS (F32 経路が壊れていない) |
| 4 | `Q38_KV_TYPE=q8_0` で 19 本を q8 の dump に対して | 全部 PASS (`worstLogit < 1e-3`、top-1 一致 0 個違い)。`Q38_ATTN_MPS_MIN_T=0 / 16` の両経路、`--q38-chunk 4 / 16 / 53` |
| 5 | 12K の一致: `--q38-compare-logits` で f32 と q8 の 12,000 トークン prefill 直後の logits の相対誤差、greedy 200 トークンのトークン列 (code / explain / tool) | 数字だけ書く (比較の対照は f32、腕を交互に) |
| 6 | 12K の ABBA (`scratch/qwen38/w3.sh` の形、2 腕 = f32 / q8、spec、公式サンプラ、seed = パス) | steady tok/s・wired 最大・file-backed 最小・Swapouts・トークン列。**期待 (導出): wired −0.5 GB、速さは変わらない** |
| 7 | 32K の土台: `prompt-code32k.tokens` (09 §6 と同じ 39,025 トークンの先頭 32,768) を作り、`--qwen38-prefill-bench --q38-tokens 32000 --q38-chunk 2048` を memlog 越しに | prefill 秒 (導出 約 6 分 = 32,768 / 89.8 tok/s)、チャンクごとの秒が 12K の 21〜23 s と同じ幅か、wired 最大・file-backed 最小・Swapouts 0 |
| 8 | 32K の decode: `--qwen38-generate` を 32K prefix の後で off / spec、幅測定 T = 1 / 2 | steady ms/トークン、**MTP ドラフトの ms** (密な注意が 12K の 14〜19 ms からどう伸びるか)、受理率、wired |
| 9 | server `-c 32768` + app に 32K の選択肢 → `TsugumiToolLoopCheck` を 21 §9 と同じ 6 会話・`--web-store` 再生・guarded.sh | 21 §9 と同じ列。2 ターン目の超過が消えるか、3 ターン目以降がどこで超えるか (A-2 の判断材料) |
| 10 | 既定を `q8_0`、app の既定文脈を 32K に。18 のチェックポイント検査 (`tool spec`) を 32K で 1 本 | 一致 (KV は位置ごとなので変わらないはず、未確認) |

## 4. 検査で見るもの (合否の基準)

- **参照との一致**: q8 の参照 dump に対して `worstLogit < 1e-3`、top-1 の違い 0。これは「同じ量子化を同じ丸めで行っている」ことの検査で、品質の検査ではない。
- **品質 (f32 と q8 の差)**: 順 5 の相対誤差とトークン列の一致長。**n = 1 のセルは数字だけ書き、解釈を書かない。**基準を先に決めておく: greedy 200 トークンの列が code / explain / tool の 3 本で一致するなら、Q8_0 の劣化はこの用途では見えないと判断して進む。1 本でも途中で分かれたら、分かれた位置の f32 / q8 の top-2 の差を記録し、ユーザーに見せる (進退は決めない)。
- **メモリ**: 順 6 で q8 の wired が f32 より小さいこと、順 7 で 32K q8 の wired 最大が 12K f32 (16 §0: 14.28〜14.57 GB) を超えないこと、Swapouts 0。
- **速さ**: 順 6 の steady tok/s の比。dequant の費用が乗る側 (score / mix は decode の host lane) なので、遅くなる可能性はある。遅くなったら量だけ書く。

## 5. 測る前に分かっていること (導出) と、分からないこと (未確認)

| | 導出 | 未確認 |
| --- | --- | --- |
| メモリ | 32K q8 = 589 MB、12K f32 = 749 MB。wired の点は今以下 | 32K の decode で expert ビューの wired (13: 6〜7 GB) が同じか。KV が小さくなった分を expert が埋めるだけなら wired は下がらない (それでも Swapouts は増えない) |
| prefill | 約 6 分 (90 tok/s、チャンクごとに一定)。毎ターンの増分は prompt cache で今と同じ | 32K でチャンクの秒が一定か (インデクサのスコアは capacity/4 に比例するが小さい) |
| decode | QSA なので文脈に依らない | 32K の実測は無い (12K まで)。MTP の密な注意は prefix に比例 (12K → 32K で注意部分 2.7 倍) |
| ツールループ | 2 ターン目の 1 ラウンド目 8.2K〜12.3K (21 §9) は 32K に収まる | 3 ターン目以降。日本語ページ 1 枚 1,300〜3,300 トークン、1 ターンに 2〜4 枚で、A-2 のまま 4〜6 ターンで 32K に着く |
| 品質 | llama.cpp の `-ctk q8_0 -ctv q8_0` は一般に劣化が見えないとされる | このモデル (Hkv = 2、D = 256、Q2 の重み) で同じかは順 5 で見る |

## 6. 判断が要ること

- 順 5 で greedy の列が分かれたときの扱い (§4 に「ユーザーに見せる」とだけ書いた)。
- app の 32K を既定にするか、選択肢に足すだけにするか。速さは変わらず、cold な prefill は 6 分 (導出) なので、16K を間に置くかも含めて。
- インデクサ raw 鍵のリング化 (32K で 101 → 約 1 MB)。prompt cache の巻き戻し先の部分ブロック 3 行をチェックポイントに含めれば安全だが、18 の形式を変える。この計画の外。
- KV を mmap のファイルに置いて追い出せるページにする案 (SSD 置き)。利得の上限は KV の総量 (q8 で 0.59 GB) で、expert と同じページキャッシュを取り合う。**この計画で足りなければ考える。**

## 7. 落とし穴 (先に知っておくこと)

- **丸めが 2 種類ある。**ggml は `roundf` (0.5 を 0 から遠い側)、numpy の `np.rint` と Metal の `rint` は偶数丸め。両側を `roundf` 相当に揃える (§2-3)。ずれると `worstLogit` の検査だけ落ちて原因が見えにくい。
- **K は RoPE 後の値を量子化する。**`q38_attn_prep` が in place で掛けている今の形を、kTmp → (q, kc) の形に変える。`compareSelected` の退避 (`:679`) は float の行バイト数を前提にしているので、kTmp の退避に書き換える。
- `q38_attn_score` のレーン割り (dims l, l+32, …) は 32 要素ブロックと直交している。レーン l がブロック k の要素 l を読む形になり、d はブロックごとにレーン全員が同じ値を読む。ブロックの並びで読む形に変えるなら score と mix の両方。
- `attentionSgemm` は float32 の MPS 行列を前提にしている。KV 側は gather で float に戻すので触らない。
- MTP 層の KV も同じカーネルを通る (`attention(dense: true)`)。`kBlocks: capacity` で全 prefix を読むので、32K では 1 ドラフトあたり q8 で 32,768 × 1,088 B ≈ 36 MB を読む (f32 なら 134 MB)。
- 32K の token ファイルはまだ無い (`prompt-code16k.tokens` は 16,384 個)。09 §6 の手順で 32,768 個を作る。
- app の `contextOptions` のコメント「16K 以上は 18 GB でスワップ (09)」は float32 の KV の話。書き換える。
- 見張り: `guarded.sh` の新しい条件 (直近 60 秒 +32,768 ページ超、全体 +65,536 超、21 §8)、`caffeinate -i`、腕は交互、走行の最初の数分は RSS・Swapouts を見る。
