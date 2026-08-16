# PLAN_VISION — Vision (画像入力) の実装

作成: 2026-08-16
M3 Pro 18GB / macOS 15.7.5 / `macos15-support` ブランチ
前提: **PLAN_QAT は受入完了** (`RESULTS_QAT.md`、QAT チェックポイント採用、既定 48 スロット)。
入力: `docs/investigations/PLAN_QAT_REVIEW_AND_VISION_FEASIBILITY.md` (実現可能性調査)、
本 PLAN 作成時に取得した **上流の参照実装ソースとリモート実測** (§1 / §2)。

表記は PLAN.md / PLAN_QAT.md と同じ: **実測** / **導出** / **未確認**

### ターゲット (ユーザー指示、2026-08-16)

| 対象 | 方針 |
| --- | --- |
| `TurboFieldfareCLI` | **対応する** (`--image`) |
| `TurboFieldfareServer` (OpenAI 互換) | **対応する** (`image_url` content part) |
| `TurboFieldfareApp` (Mac GUI) | **放置。**本 PLAN のスコープ外。ビルドが壊れないことだけ守る |
| 音声 / 動画 | **やらない** (§9)。ただし「黙って壊れる」ことは許さない = 明示的に拒否する |

---

## 0. 結論を先に (この PLAN で何が確定し、何が残っているか)

| # | 論点 | 状態 |
| --- | --- | --- |
| 1 | vision の重みは入手できるか | **解決 (実測)。**Google QAT リポジトリに bf16 356 本 / 1,145,588,832 B。range 取得可 (§1-1) |
| 2 | その重みは手元の QAT テキスト重みと同系統か | **解決 (実測、本 PLAN の新規事実)。**ローカル snapshot の bf16 テンソルが Google QAT リポジトリと**バイト一致**。派生元が同一と確定した (§1-2) |
| 3 | tower のアルゴリズムは何か | **解決 (実測)。**`transformers v5.6.2` の `models/gemma4/` を読み、全段を特定した (§2)。「SigLIP 系だろう」という推測は不要になった |
| 4 | 画像 1 枚 = 280 トークン固定か | **違う (実測)。**280 は**上限**で、実枚数はアスペクト比で決まる (§2-1)。調査資料の「280 固定」は訂正 |
| 5 | 最大の実装障害 | ~~**prefill チャンクの上限 128 トークン**~~ **解消済み** (2026-08-16)。上限は 2048、既定も 2048 になった (`docs/investigations/PREFILL_THROUGHPUT.md` §7)。画像スパン (最大 280) は 1 チャンクに収まるので、双方向 attention の前提は満たせる。プランナ側でスパンを割らない保証は別途要る (§4-5) |
| 6 | 最大の性能リスク | tower は画像 1 枚あたり **3.5 TFLOP** (280 トークン時、**導出**)。TTFT に直撃する (§3-2) |
| 7 | 最大の正しさリスク | 画像の前処理 (bicubic + antialias) は**上流とビット一致しない**。カーネルのバグと前処理の差を**分離できる検証系**が要る (§6) |

---

## 0-A. 本 PLAN 作成後に変わった前提 (2026-08-17)

本 PLAN は `docs/investigations/PREFILL_THROUGHPUT.md` の **§7-1 時点** (78.5 pp) を
前提に書いた。その後 §7-5 / §7-8 / §7-9 が入り、**prefill の実装は別物になった**
(2478 トークンで **231 pp**、GPU 8.11 s)。本 PLAN の設計判断のうち
**4 つが陳腐化し、1 つは追い風になった**。以下が正、本文の該当箇所より優先する。

| # | 本文の記述 | 現状 (**実測**) | 影響 |
| --- | --- | --- | --- |
| 1 | §4-5-a「`maxChunkTokens = 128`、画像スパン 280 は**必ず割れる**」 | 上限 2048、**既定も 2048** | **問題そのものが消えた。**スパン ≤ 280 は既定チャンクに丸ごと収まる。スクラッチ +20 MB も KV +0.03 GB も**不要**。ただし「境界に跨がらせない」責任はプランナに残る (§0-A-1) |
| 2 | §4-4 #2「`prefill_dequant_int4_qmm_f16_block` の bf16 版」 | その scalar QMM は**もう主経路ではない**。`prefill_int4_qmm_simdgroup_f16` (64×64 タイル / `simdgroup_matrix` / 4 simdgroup) が既定で **3.53 TFLOP/s** | 写す雛形が変わる。**bf16 版は int4 版より簡単** — dequant もグループも要らず、`Bs` を bf16→half で埋めるだけ (§0-A-2) |
| 3 | §3-2「S=280 で 3〜10 s/画像」「実効 0.41 TFLOP/s」 | 射影の実効は **3.53 TFLOP/s**、MoE で 2.17 | 3.54 TFLOP ÷ 3.5 TFLOP/s ≈ **1 s/画像** (**導出**)。§7 ゲート 4 の「10 s を超えたら 140 に落とす」はほぼ発火しない見込み |
| 4 | §4-4 #5 / §4-5-b「`attention_prefill_causal_tiled` から派生」「`last_exclusive` に手を入れる」 | 既定は `attention_prefill_causal_qblock_d256` / `_d512` (`RuntimePrefillAttentionPath.causalQBlock`)。tiled は**フォールバック**に降格 | 双方向マスクは **qblock の `window_last[j]` / `window_first[j]`** に入れる。tiled 側にも同じ意味で入れる (フォールバックが黙って causal に落ちるのを防ぐ、§0-A-3) |
| 5 | §8「V3 の bf16 QMM 実測が 0.1 TFLOP/s 台なら中止」 | その水準は scalar QMM 時代の値 | 中止線を **1.0 TFLOP/s** に引き直す (§0-A-4) |

### 0-A-1. チャンク境界 — 「広げる」から「割らせない」だけに縮んだ

既定 2048 ≥ 280 なので、`PrefillChunkPlanner.spans` に画像スパンを渡し、
**スパンを跨ぐ境界を作らない**ようにするだけでよい。却下案だった
「`allowedPrefillChunkTokens` を太らせる」も、採用案だった
「スクラッチを max(chunkTokens, 280) に広げる」も**どちらも不要**。
2048 トークンのチャンクの中に 280 のスパンが 1 つ入るだけなので、
プランナは「境界がスパン内部に落ちるなら境界をスパン先頭まで前倒しする」で足りる。
スクラッチも KV リングも**現行のまま 1 バイトも増えない**。

> 例外: `--prefill-chunk-tokens 32/64/128` を明示指定した実行では
> スパン (最大 280) がチャンク幅を超える。この場合は
> **そのチャンクだけスパン長まで広げる**か、明示的にエラーにする。
> V5 で決める。既定経路には影響しない。

### 0-A-2. bf16 タイル GEMM は int4 版より簡単

`prefill_int4_qmm_simdgroup_f16` の K ループは
「重み 32×64 を threadgroup に FP16 で dequant して置く」ところが本体で、
bf16 重みならその dequant が**単なる型変換**になる。
scale/bias の索き (`(k0 + kk) / kPrefillGroupSize`) も、group 32/64 の
両対応も、**まるごと消える**。§4-4 #2 が「group 概念に依存しない」と
書いていた性質は、int4 版が既に獲得したので**差分がさらに小さい**。

なお §7-8 の教訓として、**この経路は fp16 で重みを持つので上流と bit 一致しない**
(重み 1 個につき丸めが 1 回増える)。vision は元から前処理で一致しないので
新しい問題ではないが、§6-1 層 B の閾値はこれを織り込む。

### 0-A-3. 双方向マスクの実装位置

`attention_prefill_causal_qblock_impl` はクエリ `j` ごとに
`window_first[j]` / `window_last[j]` を持ち、キーループはその和集合
(`block_first`..`block_last`) を回して各クエリで範囲外を `continue` する。
**画像スパンはこの 2 つの配列を書き換えるだけで表現できる** —
スパン内のクエリは `window_last[j] = min(kvValidCount, spanEnd)`。
`block_last` は自動的に広がるので、キーループの構造には手を入れない。

§2-7 の「スパン ≤ 280 < window 1024 なので `first` の緩和は不要」は
qblock でもそのまま成立する (**導出**、V4 で参照比較により裏を取る)。

### 0-A-4. 追い風: TTFT の見通し

§3-2 の 3 つの対策のうち **2 (bf16 GEMM のタイル化) は既に済んだ資産の流用**になり、
**3 (tower をチャンクループの外に出す) は元から設計に入っている**。
残る 1 (`--image-tokens`) は出すが、既定 280 を下げる判断は**発火しない見込み**。
§7 ゲート 4 の「実測して記録する」は変えない — 見込みは見込みでしかない。

### 0-A-5. テキスト回帰のベースラインが動いた

§6-4 は `RESULTS_QAT.md` の QAT ベースラインと比べると書いてあるが、
そのベースライン (37.2 pp) は **231 pp** に置き換わっている。
V5 / V6 のテキスト回帰は **`PREFILL_THROUGHPUT.md` §7-9 の 231 pp / GPU 8.11 s /
peak 6.64 GB** に対して測る。decode 側 (`bench.sh ja`) は QAT ベースラインのまま。

---

## 0-B. V0 完了 (2026-08-17、**実測**)

§5-V0 の出口条件を満たした。参照系は動いており、V1 以降に進んでよい。

| 成果物 | 実体 |
| --- | --- |
| `scratch/vision-venv/` | uv + Python 3.12.11 / torch 2.13.0 (CPU) / **transformers 5.6.2** / torchvision 0.28.0 / pillow |
| `scratch/vision-weights/vision.safetensors` | **356 テンソル / 1,145,588,832 B**。§1-1 の予測と**厳密一致**。payload SHA-256 `4b30208c77e9a5dd…` |
| `scratch/vision-fixtures/` | 3 画像 × soft token {70, 280} = **6 ケース**、195 MB |
| `Scripts/vision/fetch_vision_weights.py` | index SHA-256 / テンソル数 / バイト数 / 全 BF16 を**検査してから**書く。46 GB の shard 1 から 1.15 GB だけを 6 回の Range 要求で取る |
| `Scripts/vision/make_test_images.py` | 合成画像 3 枚。写真ではなく**このリポジトリだけから再現できる**ことを優先した |
| `Scripts/vision/dump_vision_fixtures.py` | 26B のテキスト側を**一度もロードせずに** tower + projector を組む (§5-V0 の想定どおり) |

### 0-B-1. 参照は float32 で取った (**判断**)

重みは bf16 だが、参照は float32 で回している。
**参照は比較対象より精度が高くなければ意味がない** — bf16 参照だと
参照自身の丸めが許容に混ざり、§6-1 層 B が検出したい写し間違いを隠す。

### 0-B-2. 「280 は上限」が実測で確定した

§2-1 の主張 (付録 A で調査資料を訂正した点) を、6 ケース全部が裏付けた。
**どのケースも 280 にも 70 にもならない:**

| 画像 | max_soft_tokens | patch 格子 | patch 数 | **soft token 数** |
| --- | ---: | --- | ---: | ---: |
| square-512 (512×512) | 70 | 24×24 | 576 | **64** |
| square-512 | 280 | 48×48 | 2304 | **256** |
| tall-480x1200 | 70 | 15×39 | 585 | **65** |
| tall-480x1200 | 280 | 30×78 | 2340 | **260** |
| wide-1024x768 (1024×768) | 70 | 27×21 | 567 | **63** |
| wide-1024x768 | 280 | 57×42 | 2394 | **266** |

> **本文 §2-1 の計算例は誤りだった。**「1024×768 → factor≈0.79 → 816×576 →
> 51×36 patch = 1836 → 204 soft token」と書いたが、正しくは
> factor = √(2520·16² / (1024·768)) = **0.9057**、912×672、**57×42 = 2394 patch、266 soft token**。
> 「280 固定は誤り」という結論は変わらない (むしろ**強まった**) が、
> 途中の数字は上の表で置き換える。

### 0-B-3. fixture の形式

Swift から読む素朴なバイナリ (`TFVFIX01` + dtype u32 + ndim u32 + dims u32[] + payload)。
1 ケースにつき `pixel_values` / `position_ids` / `patch_embed` /
`layer{0,13,26}` / `pooled` / `soft_tokens` の 8 本 + `manifest.json`。

信号があることも確認した (§6-3 が要求する「参照側に信号があること」、**実測**):
`layer26` の rms 74.3 が `pooled` で 1.21 に落ちる — `std_scale`/`std_bias` に
よる標準化 (§2-5) が効いている証拠。NaN はどのテンソルにもない。
`soft_tokens` の rms が **1.11** である点が §2-6 の罠を裏づける:
テキスト埋め込みは √2816 = 53.1 倍されるので、
**画像位置に同じスケールを掛けると 53 倍ずれる**。

### 0-B-4. V0 で追加検証した本文の主張 (**実測**、上流ソースを直接読んだ)

本 PLAN §2 は取得したソースの読解だった。今回ローカルに 5.6.2 が入ったので照合した。

| 主張 | 判定 |
| --- | --- |
| §1-3 `vision_config` 全項目 | **一致** (config.json を再取得して照合) |
| §1-3 `processor_config.json` (`do_normalize: false` 含む) | **一致** |
| §2-1 前処理の式・patch 並び (行優先) ・patch 内 (py, px, c) | **一致** (`convert_image_to_patches` の `transpose(1,3,2,4,0)`) |
| §2-2 patch embedder (`2(x-0.5)` → Linear → 位置テーブル 2 本の和) | **一致** |
| §2-4 2D RoPE (spatial_dim 36、`inv_freq[j] = 100^(-2j/36)`、両次元同一) | **一致** (`compute_default_rope_parameters` に「両次元が同一周波数」と明記されている) |
| §2-5 pooler (`(x/3) + (pw/3)·(y/3)` の並び → ×√1152 → 標準化) | **一致**。加えて**プーリングの累算は float32** で行われる (`hidden_states.float()`) |
| §2-6 projector (no-scale RMSNorm → Linear) と soft token が √hidden を**受けない**こと | **一致** |
| §2-7 双方向マスクが sliding 層のみ | **一致**。さらに `text_config.use_bidirectional_attention == "vision"` を確認 — この分岐に入ることが**設定で確定**した |
| §4-6 `--image-tokens {70,140,280}` | 上流の `_SUPPORTED_SOFT_TOKENS` は **(70, 140, 280, 560, 1120)**。280 に絞るのは我々の選択であって上流の制約ではない (`processor_config` の既定が 280) |

---

## 0-C. V1 完了 (2026-08-17、**実測**)

§5 の V1 (前処理 + トークン列、GPU なし) の出口条件を満たした。

| 追加したもの | 中身 |
| --- | --- |
| `Sources/TurboFieldfare/Vision/VisionGeometry.swift` | `get_aspect_ratio_preserving_size` + patchify の移植。退化アスペクト比の救済分岐も含む |
| `.../VisionImagePreprocessor.swift` | ImageIO で復号 → CGContext で目標サイズに縮小 → `/255` しつつ (py, px, c) 順に並べ替え → `[P, 768]` の fp16 |
| `.../VisionPrompt.swift` | マーカー ID の解決、`<\|image\|>` → `boi + image×n + eoi` の展開、スパン算出、**メディアマーカーの拒否** |
| `Sources/TurboFieldfareValidation/Support/Fixtures/VisionFixtures.swift` | fixture リーダ + **NaN-safe な `relativeError`** (§6-3)。V3/V4 の KernelCheck からも使う |
| テスト 17 本 (4 スイート) | 下記 |

`Scripts/test.sh`: **722 テスト / 131 スイート、12 issue**。
内訳は `PREFILL_THROUGHPUT.md` §7-7 の陳腐化 5 件と**完全に同じで、新しい失敗はない**
(705 → 722 は今回の +17)。

### 0-C-1. 幾何は「一致」、画素は「近い」— 分けて測った

前処理は上流とビット一致しない (§4-3) が、**整数の部分は一致する**。
そこを混ぜないよう、テストを 2 本に割った。

- **幾何は厳密一致**: 6 ケース全部で patch 格子 / patch 数 / soft token 数が
  参照ダンプと**完全一致**。加えて「pooled セルはちょうど 9 patch を受ける」
  「両辺が 48 の倍数」「patch 数 ≤ 予算」を 36 通りのアスペクト比で検査。
- **画素は差を測る** (§6-1 層 C、**実測**、値域 [0,1] の平均絶対差):

| ケース | 整列時 | **転置した対照** | 分離 |
| --- | ---: | ---: | ---: |
| square-512-s70 | 0.00329 | 0.2958 | 90× |
| square-512-s280 | 0.00266 | 0.2681 | 101× |
| tall-480x1200-s70 | 0.00811 | 0.3515 | 43× |
| tall-480x1200-s280 | 0.00164 | 0.3536 | **216×** |
| wide-1024x768-s70 | 0.01148 | 0.3532 | 31× |
| wide-1024x768-s280 | 0.00159 | 0.3530 | **222×** |

**CoreGraphics の縮小と torchvision の antialias bicubic の差は平均 0.0016〜0.011。**
テスト画像は高周波のリング模様を意図的に入れてある (リサイズが実際に効く条件) ので、
写真ではこれより小さくなるはず (**未確認**)。

> **対照を測ったことが本体。**§6-3 の「壊すと FAIL することを確認する」を
> ここで先に一度やった: 同じ測り方で patch 格子を転置すると
> **31〜222 倍**に跳ね、通ったばかりの閾値 0.05 を確実に超える。
> 閾値 0.05 は「別のリサンプラは通すが、並びの間違いは通さない」ところに置いた。

### 0-C-2. 塞いだ穴 — 画像なしの `<|image|>`

`applyChatTemplate` / `encodeTextContinuation` が
`<|image|>` `<|image>` `<image|>` `<|audio|>` `<|video|>` (と開閉の異形) を
**エラーにする**ようになった (`VisionPromptAssembler.rejectMediaMarkers`)。

これまでは特殊 ID が普通の埋め込みとして通り、**tower も走らないまま
「見たかのような流暢な答え」が出ていた**。出力が壊れて見えないので
警告では足りず、エラーにするしかない。
マーカーは長い順に走査するので、`<|image|>` が `<|image>` として誤報されない
(これもテストで固定した)。

`ServerPromptCache.matchTextContinuation` は拒否を**キャッシュミス扱い**にする。
ここで投げると「キャッシュの問題」として報告されてしまうので、
通常の encode 経路に型付きエラーを出させる。

### 0-C-3. 新規事実: 双方向グループは boi/eoi を**含まない** (**実測**)

§2-7 / §4-5-b は「同グループ (= 同じ 1 枚の画像) のトークン同士」と書いていたが、
その境界を詰めていなかった。上流を読んで確定した:

```
processing_utils.py:1665   mm_token_types[isin(input_ids, self.image_ids)] = 1
processing_utils.py:596    self.image_ids = [image_token_id]        # ← 258880 だけ
```

つまり双方向になるのは **`<|image|>` × n の範囲だけ**で、
**`<|image>` (255999) と `<image|>` (258882) は普通の causal トークン**。
`VisionImageSpan.tokenOffset` を boi の**次**から始めているのはこのため。
V5 のマスクもこの範囲で入れる。

---

## 0-D. V2 完了 (2026-08-17、**実測**)

§5 の V2 (フォーマット拡張 + repacker) の出口条件を満たした。
**GPU も 26B の実機ロードも使っていない**ので、以下はすべて合成データでの実測である。

| 追加したもの | 中身 |
| --- | --- |
| `GTurboFormatV1` | `knownFlags += "visionTower"`、`versionMinorVision = 1`、`visionWeightsPath` |
| `GTurboManifestVisionV1` | manifest の optional `vision` セクション (§0-D-1) + 整合検査 |
| `VisionConfig` (`ModelTypes.swift`) | ランタイム側の期待値。`ManifestReader.validateVision` が field 単位で照合 |
| `VisionModelSource` / `VisionSourcePin` | Google リポジトリのピン (repo / revision / index SHA / 356 本 / 1,145,588,832 B / パリティ 3 本 / tower config) |
| `VisionSourceLoader` | index・config.json を取得して**ピンと照合**し、必要な shard のヘッダだけを Range で取る |
| `VisionRepackPlanner` | **config から期待テンソル表を生成**して突き合わせ、resident v1 レイアウトを組む (§0-D-2) |
| `RoutingSourceByteProvider` | 1 つのコピー計画を shardID の名前空間で 2 つのソースに振り分ける |
| `--include-vision` | CLI フラグ。既定は無効 |
| テスト 23 本 | format 11 / ManifestReader 4 / repack 6 / CLI 2 |

`Scripts/test.sh`: **745 テスト / 132 スイート、12 issue**。
issue の内訳は `PREFILL_THROUGHPUT.md` §7-7 の陳腐化 5 件と**完全に同じで、新しい失敗はない**
(722 → 745 は今回の +23)。

### 0-D-1. `manifest.vision` は §4-1-a に 3 項目足した

`weightsPath` / `tensorCount` / `payloadBytes` を追加した。前者はランタイムが
ファイル名をハードコードしないため、後ろ 2 つは「index が主張する中身」と
「`files` が主張するファイルサイズ」を**別々に検証できる**ようにするため
(payload > ファイルサイズは encode 時に弾く)。
`sourceRevision` は**解決済みコミット**で、ピンと一致しなければ拒否する。

### 0-D-2. 「向こうが送ってきた 356 本」を信用しない

§4-2 は「`RepackAudit.tensors_dropped_multimodal` に除外一覧が出ているから
一次情報はある」と書いたが、**これは誤りだった (実測)**。
ローカル QAT snapshot の 1279 テンソルは**全部 `language_model.` で、
multimodal テンソルは 1 本もない**。除外一覧は空で、拾うべき名前の情報源にならない。

代わりに **`vision_config` から期待テンソル表を生成**する方式にした:

```
2 (patch embedder) + 13 × 27 (層) + 2 (std) + 1 (projector) = 356
```

356 という数がピンと一致することも、各テンソルの shape が config から
導けることも検査する。したがって「上流が黙って 1 本増やした / 形を変えた」は
インストール時に落ちる。テストでは shard のヘッダから `std_bias` を 1 本だけ
消した合成リポジトリを作り、`missing: vision_tower.std_bias` で落ちることを固定した。

> **この式を本物の重みに当てて確認した (実測、2026-08-17、ネットワーク不要)。**
> V0 で取得済みの `scratch/vision-weights/vision.safetensors` の
> safetensors ヘッダに対して同じ式を展開したところ、
> **356 名すべてが一致 (欠けも余りもゼロ)、全 shape 一致、全 BF16、
> 合計 1,145,588,832 B がピンと一致**した。
> 合成データだけで通したのではなく、production のピンが本物を再現している。

### 0-D-3. パリティ検査 — §1-2 の手作業を毎回の自動検査にした

§1-2 は SHA-256 の先頭 16 桁だけを記録していたので、**全 64 桁を取り直して
`VisionModelSource.pin.parityTensors` にピンした** (**実測**、ローカル snapshot から):

| テンソル | SHA-256 |
| --- | --- |
| `language_model.model.norm.weight` | `134bc0ecd2a53f98…3013e95f43` |
| `…layers.0.input_layernorm.weight` | `978017393fdf7a41…f5eeaefa77` |
| `…layers.7.post_feedforward_layernorm.weight` | `5889b58a3573b37a…d9671455a10` |

インストール時に**テキスト側と vision リポジトリ側の両方**をハッシュし、
ピンと、そして互いと突き合わせる。3 つのどれが食い違っても中止する。

> **検出力の裏取り (§6-3 の手続き)。**別シードで作った合成チェックポイントから
> tower を組み、元のスナップショットに対してインストールさせる試験を入れた。
> この tower は**自分のピンとは整合している**ので、テキスト側との比較でしか
> 気づけない。実際 `the two checkpoints differ` で落ちる。
> 「壊すと落ちる」ことを確認していない検査は入れていない。

### 0-D-4. `--source-snapshot` でも tower はネットワークから来る

テキストはローカル、tower は HTTP という **2 本差し**なので、
スナップショット用イニシャライザにも session / token / リトライ設定が要るようになった
(`--include-vision` のときだけ効く)。コピー計画側は `shardID` に
`vision:` 名前空間を付けて 1 つの計画に同居させ、
`RoutingSourceByteProvider` が振り分ける。**チェックポイント (resume) も 1 本のまま**で、
vision 込みの計画は fingerprint が変わるので、text-only の途中状態を
vision 付きで resume しようとすると「計画が変わった」として弾かれる。

### 0-D-5. 退行なしの担保

- **manifest のバイト一致**: `Tests/TurboFieldfareFormatCompatibility` の
  凍結フィクスチャ (manifest / layout / resident index の SHA-256) が**そのまま通る**。
  `vision` は optional なので、text-only の JSON には**キー自体が現れない**
  (これも単体テストで固定した)。
- **fingerprint**: vision があるときだけ 2 フィールド追加する。text-only の
  計画 fingerprint は 1 ビットも変わらない。
- **既定の出力物**: `--include-vision` なしのインストールで `vision/` が作られず、
  vision リポジトリへのリクエストが **0 件**であることをテストで固定した。

### 0-D-6. 旧ランタイムの拒否について (正直な限界)

「旧バイナリが flag で落ちる」は**このリポジトリ内では直接試験できない**
(現行ビルドは `visionTower` を既知フラグとして知っているため)。
機構そのもの (未知フラグ → `ModelError.unknownFlag` で `Model.load` が失敗) は
既存テストと今回足した誤記フラグのテストで押さえてある。
V6 の受入で、**vision 付き `.gturbo` を旧タグのビルドに食わせて実際に落ちること**を
一度だけ実機で確認する (§7 に追加)。

---

## 1. ソース側の事実確認 (**実測**、2026-08-16)

### 1-1. vision 重みの所在と内訳

`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`
(revision `f1e06dc520982d9b9edd76859fdb7ab209449949`、gated: false、
index SHA-256 `907826a6e46ff454272bd6db1fee629d5531a2303be22986d825a0871d7dc7a7`)。
shard 1 の safetensors ヘッダを Range 取得して直接数えた。

**vision テンソル 356 本 / 合計 1,145,588,832 B (1.146 GB)、全て BF16。全て shard 1 にある。**

| テンソル (層は N で代表) | dtype | shape | bytes |
| --- | --- | --- | ---: |
| `model.vision_tower.patch_embedder.input_proj.weight` | BF16 | [1152, 768] | 1,769,472 |
| `model.vision_tower.patch_embedder.position_embedding_table` | BF16 | [2, 10240, 1152] | 47,185,920 |
| `…encoder.layers.N.self_attn.{q,k,v,o}_proj.linear.weight` | BF16 | [1152, 1152] | 2,654,208 × 4 |
| `…encoder.layers.N.self_attn.{q,k}_norm.weight` | BF16 | [72] | 144 × 2 |
| `…encoder.layers.N.mlp.{gate,up}_proj.linear.weight` | BF16 | [4304, 1152] | 9,916,416 × 2 |
| `…encoder.layers.N.mlp.down_proj.linear.weight` | BF16 | [1152, 4304] | 9,916,416 |
| `…encoder.layers.N.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight` | BF16 | [1152] | 2,304 × 4 |
| `model.vision_tower.{std_scale,std_bias}` | BF16 | [1152] | 2,304 × 2 |
| `model.embed_vision.embedding_projection.weight` | BF16 | [2816, 1152] | 6,488,064 |

1 層 = 40,375,584 B、× 27 層 = 1,090,140,768 B。残りが patch embedder (48,955,392 B) と
projector (6,488,064 B) と std (4,608 B)。合計は上の 1,145,588,832 B と厳密に一致する。

> `position_embedding_table` が **[2, 10240, 1152] の 3 階テンソル**で 47 MB ある点に注意。
> `GTurboResidentIndexEntryV1.shape` は `[UInt32; 4]` なのでそのまま格納できる (**実測**)。

### 1-2. ローカル QAT snapshot と Google QAT リポジトリは同系統 (**実測**、新規事実)

調査資料 §3-1-b は「現行ピン (`mlx-community/…-it-4bit`) の vision 重みは Google QAT の
ものと**不一致**」を示した。これは今も正しい。ただしそれだけでは
「では Google の vision をうちの QAT テキストと組んでよいのか」に答えていない。

そこで**手元の `scratch/qat-aligned-snapshot/` と Google QAT リポジトリで、
量子化されない BF16 テンソルのバイト列を直接比較した**:

| テンソル | ローカル (mlx-aligned) | Google QAT (range 取得) | 判定 |
| --- | --- | --- | --- |
| `…model.norm.weight` [2816] | `134bc0ec…` | `134bc0ec…` | **一致** |
| `…layers.0.input_layernorm.weight` | `978017 39…` | `97801739…` | **一致** |
| `…layers.7.post_feedforward_layernorm.weight` | `5889b58a…` | `5889b58a…` | **一致** |

(SHA-256 先頭 16 桁。テンソル名は mlx 変換で `model.language_model.X` →
`language_model.model.X` に付け替えられているが中身は同一。)

→ **`qat-q4_0-mlx-aligned` は `google/…-qat-q4_0-unquantized` の変換物である**ことが
実測で確定した。したがって **Google リポジトリの vision 重みは、いま採用している
テキスト重みと同一チェックポイント由来**であり、組み合わせて問題ない。
調査資料が「唯一の経路」と書いた前提は、推定ではなく事実になった。

### 1-3. トークンと設定 (**実測**)

| 項目 | 値 | 出所 |
| --- | --- | --- |
| `boi_token_id` | 255999 = `<\|image>` | config.json / tokenizer.json |
| `image_token_id` | 258880 = `<\|image\|>` | 同上 |
| `eoi_token_id` | 258882 = `<image\|>` | 同上 |
| `vision_soft_tokens_per_image` | 280 (**上限**、§2-1) | config.json |
| chat template の画像出力 | content part `{"type":"image"}` → `<\|image\|>` を 1 個 | `chat_template.jinja:322-324` |
| ローカル snapshot の `config.json` | `vision_config` **ブロックだけが落ちている** (image/boi/eoi の ID と 280 は残っている) | 実測 |

`vision_config` (Google リポジトリの config.json から全文取得、**実測**):

```
hidden_size 1152 / num_hidden_layers 27 / num_attention_heads 16 /
num_key_value_heads 16 / head_dim 72 / intermediate_size 4304 /
patch_size 16 / pooling_kernel_size 3 / position_embedding_size 10240 /
rope_parameters {rope_type: default, rope_theta: 100.0} /
rms_norm_eps 1e-6 / hidden_activation gelu_pytorch_tanh /
standardize true / use_clipped_linears false / default_output_length 280
```

`processor_config.json` (**実測**、調査資料で「未取得」だったもの):

```
image_processor: do_convert_rgb true / do_resize true / resample 3 (BICUBIC) /
                 do_rescale true / rescale_factor 1/255 / do_normalize false /
                 image_mean [0,0,0] / image_std [1,1,1] /
                 patch_size 16 / pooling_kernel_size 3 / max_soft_tokens 280
```

**`do_normalize: false`** が効く。平均/分散の正規化は前処理ではなく**モデル内**で
行われる (`std_scale` / `std_bias`、§2-4)。前処理は「リサイズして 255 で割る」だけ。

---

## 2. 参照実装の仕様 (**実測**、`transformers v5.6.2` の `src/transformers/models/gemma4/`)

推測ではなく上流のソースを読んで確定させた。移植対象はこの 6 段。
本 PLAN のカーネル設計はすべてここに紐づく。

### 2-1. 前処理 (`image_processing_gemma4.py` / `image_processing_pil_gemma4.py`)

```
max_patches   = max_soft_tokens * pooling_kernel_size^2   # 280 * 9 = 2520
side_mult     = pooling_kernel_size * patch_size           # 3 * 16 = 48
factor        = sqrt(max_patches * patch_size^2 / (H * W))
target_H      = floor(factor * H / 48) * 48
target_W      = floor(factor * W / 48) * 48                # 0 になる辺の救済あり
resize(bicubic, antialias=True) → rescale(1/255) → patchify(16)
patch i の並び  = 行優先 (y * pw + x)、1 patch のベクタは (py, px, c) の順で 768 要素
position id     = (x, y)
soft token 数   = pw * ph / 9        # ← 画像ごとに変わる。280 は上限
```

> **調査資料の「画像 1 枚 = 280 トークン」は誤り (**実測**)。**
> 正方形に近い画像なら 48×48 の格子に丸められた結果 280 未満になる。
> 例: 1024×768 → factor≈0.79 → 816×576 → 51×36 patch = 1836 → **204 soft token**。
> `processing_gemma4.py:150` は `boi + image_token * n + eoi` を **n = 実数**で埋める。
> **280 を固定値として書き込む実装は全部間違いになる。**

バッチのために `max_patches` までゼロ詰め (position は (-1,-1)) するが、
**本 PLAN は画像を 1 枚ずつ処理する**ので padding は発生しない (§4-4)。
これで padding マスク・`one_hot` プーリング・可変長 attention マスクが全部消える。

### 2-2. patch embedder (`Gemma4VisionPatchEmbedder`)

```
x = 2 * (pixels - 0.5)                       # [0,1] → [-1,1]
h = Linear(768 → 1152, bias なし)(x)
h += position_embedding_table[0, x_pos] + position_embedding_table[1, y_pos]
```

位置埋め込みは one-hot 行列積として書かれているが、**実体は 2 本のテーブル引きの和**。
行列積として実装する理由はない。

### 2-3. encoder layer × 27 (`Gemma4VisionEncoderLayer`)

**テキスト側の Gemma 4 層と同じサンドイッチ構造**で、MoE がないだけ:

```
h += post_attention_layernorm( attn( input_layernorm(h) ) )
h += post_feedforward_layernorm( mlp( pre_feedforward_layernorm(h) ) )
```

- `Gemma4RMSNorm` は **`weight` をそのまま掛ける** (`1 + weight` ではない)。
  ランタイムの `prefill_rmsnorm_bf16w_block` と同じ規約 (**実測**)。
- attention: 16 head × 72、**MHA (kv も 16 head)**、`scaling = 1.0`、`is_causal = False`。
  q_norm / k_norm は RMSNorm(72)、**v_norm はスケールなし RMSNorm** —
  テキスト側とまったく同じ構成で、ランタイムには既に「スケールなし v_norm」の
  概念がある (`Model.swift` のコメント、**実測**)。
- MLP: `down(gelu_tanh(gate(x)) * up(x))`、`use_clipped_linears: false` なので
  `Gemma4ClippableLinear` は素の Linear。重み名の `.linear.` はその残骸。

### 2-4. RoPE — 2 次元 (`apply_multidimensional_rope`)

ここがテキスト側と唯一構造的に違う。head_dim 72 を **前 36 = x 用 / 後 36 = y 用**に
割り、各半分に NeoX 形式の RoPE を掛ける。周波数は**両次元とも同一**:

```
spatial_dim = head_dim / 2 = 36
inv_freq[j] = 100 ^ (-2j / 36),  j = 0..17
x 側: 前 36 チャネルを (i, i+18) ペアで回転、角度 = pos_x * inv_freq[i]
y 側: 後 36 チャネルを同様に、角度 = pos_y * inv_freq[i]
```

既存の `prefill_rope_default_neox_block` は「1 head 全体に 1 つの position」を仮定して
いるので、**そのままは使えない**。ただし「half_dim ペアで回す」中身は同一なので、
`prefill_rope_apply_neox_pair` を 2 回呼ぶ薄いカーネルで足りる (§4-4)。

### 2-5. pooler + 標準化 (`Gemma4VisionPooler` / `Gemma4VisionModel`)

```
k = sqrt(P / S)                       # padding なしなら常に 3
pooled[(y/3)*(pw/3) + (x/3)] = mean over the 3x3 patch block
pooled *= sqrt(1152)                  # = 33.9411...
pooled  = (pooled - std_bias) * std_scale
```

出力の並びは pooled 格子の行優先。**平均のあとに sqrt(hidden) を掛ける順序**を守ること
(先に掛けても数学的には同じだが fp16 の丸めが変わる)。

### 2-6. projector と埋め込み差し替え (`Gemma4MultimodalEmbedder` / `Gemma4Model.forward`)

```
soft = Linear(1152 → 2816, bias なし)( RMSNorm_no_scale(pooled) )
inputs_embeds = scaled_word_embedding(input_ids)      # ← テキストは sqrt(2816) 倍される
inputs_embeds[image positions] = soft                 # ← soft は倍されない (masked_scatter)
```

> **`sqrt(hidden)` はテキスト埋め込みだけに掛かる。**ランタイムの
> `prefillEmbed.encode(..., outScale: sqrtHidden)` は全トークンに掛けるので、
> **画像位置は上書きで潰す**必要がある。ここを取り違えると 53 倍ずれた埋め込みが
> 入り、「なんとなく変な出力」になって目視では気づけない (§6 の分離検証が要る理由)。

### 2-7. 画像スパンの双方向 attention (`create_causal_mask_mapping`)

```python
mask_kwargs         = {...}                       # ← or_mask なし
sliding_mask_kwargs = mask_kwargs.copy()
sliding_mask_kwargs["or_mask_function"] = token_type_ids_mask_function(...)
return {"full_attention":    create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs)}
```

**双方向化は sliding 層にだけ入り、full attention 層 (5/30 層) は素の causal のまま** (**実測**)。
Gemma 3 では両方に入っていたので、上流のバグの可能性がある (**未確認**)。

判断: **参照実装に合わせる** (sliding 層のみ双方向)。理由は、受入で比べる相手が
その参照実装だから。両方に入れる版は関数フラグで切り替えられるようにしておき、
V4 の参照比較で「どちらが上流と一致するか」を**測って**決める。推測で決めない。

同グループ (= 同じ 1 枚の画像) のトークン同士は、causal 順序も sliding window も
無視して相互に見える。画像スパン長 ≤ 280 < slidingWindow 1024 なので、
**実効的には「未来方向だけ開ける」で等価**になる (**導出**、§4-5 で使う)。

---

## 3. 見積り

### 3-1. メモリとディスク (**導出**、実測値ベース)

| 項目 | 現状 (QAT / 48 slots) | vision 追加後 |
| --- | ---: | ---: |
| resident (テキスト) | 1.51 GB | 1.51 GB |
| resident (vision) | — | **+1.15 GB** |
| expert cache 30×48×3,719,168 | 5.36 GB | 5.36 GB |
| KV @4K | 0.28 GB | 0.28 GB (画像はテキスト枠を消費するだけ) |
| tower の作業バッファ (2520 patch) | — | **+0.10 GB** (§3-3) |
| `ExpertCacheBudget` の合計 | 7.15 GB | **8.30 GB** (推奨 12.88 GB 内) |

実 peak は QAT 実測 6.0 GB (`RESULTS_QAT.md`) に対し **7.2 GB 前後**と見込む (**導出**)。
ディスク: `.gturbo` が +1.15 GB (15.5 GB → 16.7 GB)。ダウンロードは range 取得 1.15 GB。

`ExpertCacheBudget.residentBytes` は `residentIndex.header.residentSize` を読むだけなので、
vision を**別ファイル**にする場合はそのぶんを明示的に足す必要がある (§4-1、変更点)。

### 3-2. tower の計算量 (**導出**、最重要リスク)

soft token 数 S、patch 数 P = 9S。1 画像あたり:

| S | P | qkvo | mlp | attention | 1 層計 | **27 層計** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 280 | 2520 | 26.8 G | 75.0 G | 29.3 G | 131 GFLOP | **3.54 TFLOP** |
| 140 | 1260 | 13.4 G | 37.5 G | 7.3 G | 58.2 GFLOP | **1.57 TFLOP** |
| 70 | 630 | 6.7 G | 18.7 G | 1.8 G | 27.3 GFLOP | **0.74 TFLOP** |

> **この節の実効値は §0-A で置き換わった。**以下の 0.41 TFLOP/s は
> `PREFILL_THROUGHPUT.md` §7-1 時点の値で、§7-8 の
> `prefill_int4_qmm_simdgroup_f16` により**射影は 3.53 TFLOP/s** になっている。
> 3.54 TFLOP ÷ 3.5 TFLOP/s ≈ **1 s/画像**が現在の見積り (**導出**、§0-A-4)。
> 対策 2 は済んだ資産の流用、対策 3 は元から設計に入っているので、
> 残る判断は対策 1 (`--image-tokens`) だけになった。

比較対象 (**実測**、`RESULTS_QAT.md`): テキスト prefill は 510 tok / 10.0 s。
active 4B とすると実効 **0.41 TFLOP/s**。ただしテキスト prefill は expert I/O を
含む値なので、重みが常駐する tower はこれより効率が出るはず (**未確認**)。

→ **S=280 で 3〜10 s/画像**と幅を持って見積もる (**導出**)。TTFT に直撃する。
対策は 3 つあり、V3 の**実測で**選ぶ:

1. `--image-tokens {70,140,280}` を出す (上流の `_SUPPORTED_SOFT_TOKENS` に従う)。
   既定は 280 だが、遅すぎるなら 140 を既定にする判断を V6 で行う。
2. bf16 GEMM をタイル化する (現状のテキスト prefill は素朴な QMM ブロック
   カーネル止まりなので、tower だけ先に良くしても全体では効かない可能性がある —
   **OPTIMIZATION_JOURNEY の「局所でなく全体で測る」**)。
3. tower の実行を 1 回にまとめ、prefill チャンクのループから外に出す (§4-5)。

> **先に測る。**V3 の出口条件に「実形状 (1152×1152 / 1152×4304、P=2520) の
> bf16 QMM 単体の実測 GFLOP/s」を入れる。ここが 0.1 TFLOP/s 台なら
> 設計を戻す (§8 の中止条件)。

### 3-3. tower の作業バッファ (**導出**、P=2520)

hidden / normed / q / k / v / attn_out = 各 2520×1152×2B = 5.8 MB → 34.8 MB。
mlp の gate/up/act = 各 2520×4304×2B = 21.7 MB → 65.1 MB。合計 **約 100 MB**。
S=280 の最大形状で確保し、それ以下の画像では使い残す (再確保しない)。

---

## 4. 設計

### 4-1. `.gturbo` フォーマット拡張 — v1 に「追加」し、旧ランタイムには**明示的に拒否させる**

調査資料の指摘どおり、`manifest.arch` は固定フィールドで vision の置き場がない。
方針:

| 決定 | 内容 | 根拠 |
| --- | --- | --- |
| vision の重みは **別ファイル** `vision/vision_weights.bin` | `model_weights.bin` には入れない | `Model.load` は `model_weights.bin` 全体を**毎回 eager に SHA-256 する** (`Model.swift:455`、**実測**)。同居させるとテキストのみの起動が +1.15 GB ぶん遅くなる。別ファイルなら**初回の画像入力時に遅延検証**でき、テキスト専用ワークロードのコストがゼロになる |
| その中身は **既存の ResidentIndex v1 形式をそのまま再利用** | ヘッダ + エントリ表 + 文字列表 + ペイロード | `GTurboResidentIndexCodec` は名前付き汎用テンソル目録で、bf16 dtype と 4 階 shape を持つ (**実測**)。新しいコーデックを書く理由がない |
| manifest に **`vision` セクション** (optional) を追加 | §4-1-a | `arch` に混ぜると `validateArch` の field 単位比較が壊れる |
| **`flags.visionTower = true`** を `knownFlags` に追加 | v1.1 | 旧バイナリは `ModelError.unknownFlag` で**起動時に落ちる** (**実測**、`ManifestReader.decode:103`)。「黙って画像を無視して部分的に動く」を構造的に禁止できる。調査資料 §3-4 が求めた性質そのもの |
| `versionMinor` は 0 → **1** | vision 付きの manifest のみ | 拒否は flag が担うので minor は表示的意味 (`decode` は `minor >= 0` を通す、**実測**) |

#### 4-1-a. `manifest.vision` の内容

```
{ "hiddenSize": 1152, "numLayers": 27, "numHeads": 16, "numKVHeads": 16,
  "headDim": 72, "intermediateSize": 4304, "patchSize": 16,
  "poolingKernelSize": 3, "positionEmbeddingSize": 10240,
  "ropeTheta": 100.0, "rmsNormEps": 1e-6,
  "hiddenActivation": "gelu_pytorch_tanh", "standardize": true,
  "maxSoftTokens": 280, "weightDType": "bf16",
  "imageTokenID": 258880, "boiTokenID": 255999, "eoiTokenID": 258882,
  "sourceRepo": "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
  "sourceRevision": "f1e06dc520982d9b9edd76859fdb7ab209449949" }
```

`sourceRepo` / `sourceRevision` を持たせるのは、**テキストと vision で由来が違う**
(§4-2) からで、`sourceSnapshotHash` 1 個では表現できない。

`VisionConfig` (ランタイム側の期待値、`ArchConfig` と同じ扱い) を作り、
`ManifestReader.validateVision` が field 単位で照合する。
`Model.validateRuntimeSchema` に vision 用の `requireBF16Matrix` 検査を足す
(Phase B で作った関数がそのまま使える、**実測**)。

### 4-2. repacker — dual source (ローカル text + リモート vision)

ローカル snapshot に vision はない (**実測**、1279 テンソル)。全 51.6 GB を落とし直すのは
無駄なので、**既存の HTTP Range 経路をそのまま使い、vision テンソルのバイト範囲だけ取る**。

```
swift run -c release TurboFieldfareRepack \
  --output scratch/gemma4-qat-vision.gturbo \
  --source-snapshot scratch/qat-aligned-snapshot \
  --include-vision                       # ← 追加。既定は無効 (現行の出力と 1 バイトも変わらない)
```

- ~~`--include-vision` のとき、`RepackPlanner.classify` の `.excludedMultimodal` を
  「除外」から「vision 収集」に反転させる。**除外の一覧は既に `RepackAudit` に
  `tensors_dropped_multimodal` として出ている** (**実測**) ので、
  「何を拾うべきか」の一次情報は既にある。~~
  **この項は誤りだった (§0-D-2、2026-08-17)。**ローカル snapshot に multimodal
  テンソルは 1 本もないので除外一覧は空で、情報源にならない。
  実装は `vision_config` から期待テンソル表 (356 本) を生成して照合する方式にした。
  `classify` にも `RepackAudit` にも手を入れていない。
- vision の byte provider は `HTTPRangeSourceByteProvider` (既存)。
  Phase C で `SourceByteProvider` がプロトコル化済みなので、
  **text = `LocalSnapshotByteProvider` / vision = HTTP という 2 本差し**で済む (**実測**)。
- 指紋: `SourceFingerprint.knownFingerprints` に Google リポジトリの index SHA-256
  (`907826a6…`) を追加し、revision `f1e06dc5…` をピンする。
  **§1-2 のバイト一致検査 (norm.weight 3 本) を repack 時にも実行する**。
  これが「テキストと vision の由来が同じ」を毎回機械的に保証する唯一の手段になる。
- audit / 受領証 (`verified-install.json`) に vision ファイルを追加する。

`--include-vision` を付けない既定の出力は**バイト単位で現行と同一**であること
(manifest の flags も増えない) を V2 の出口条件にする。

### 4-3. ランタイム: 画像の前処理 (Swift、CPU)

依存は増やさない。macOS SDK の ImageIO + vImage (Accelerate) で完結する。

```
CGImageSource で読み込み → sRGB / 8bit RGB に統一 (do_convert_rgb 相当)
→ target サイズ算出 (§2-1 の式をそのまま Swift に写す)
→ vImageScale_ARGB8888 or CGContext 補間で bicubic 縮小
→ Float16 に展開しつつ /255 し、(py, px, c) 順の patch ベクタに並べ替える
→ [P, 768] の fp16 バッファ + P (patch 数) + (pw, ph)
```

**位置 ID は保持しない。**padding がないので `x = i % pw`, `y = i / pw` で
カーネル内から計算できる (§2-1)。

> **リサイズは上流とビット一致しない (**未確認 → 事実上不可能**)。**
> torchvision の `antialias=True` bicubic と Apple の実装は係数もクランプも違う。
> ここは §6 で「前処理を通さない比較」を用意して**切り分ける**。
> 一致させようとして自前 bicubic を書くのは筋が悪い (上流も PIL / torchvision で
> 既に 2 実装あり、どちらとも完全一致はしない)。

### 4-4. ランタイム: tower の推論

新規 `Sources/TurboFieldfare/Metal/Vision/vision.metal` + `Kernels/Vision/`。
既存プリミティブと同型のものは**同じ構造で書き、同じ検証系に載せる**。

| # | カーネル | 内容 | 既存との関係 |
| --- | --- | --- | --- |
| 1 | `vision_patch_embed_block` | `2(x-0.5)` → [768→1152] GEMM → 位置テーブル 2 本を加算 | 新規 (GEMM 部は #2 と共通化) |
| 2 | `vision_bf16_qmm_f16_block` | `Y[t,n] = Σ_k W[n,k]·X[t,k]`、W は bf16、X/Y は fp16 | ~~`prefill_dequant_int4_qmm_f16_block` の bf16 版~~ → **`prefill_int4_qmm_simdgroup_f16` の bf16 版** (§0-A-2)。64×64 タイル / 4 simdgroup / K タイル 32 の構成をそのまま使い、dequant を型変換に置き換える。**group 概念も scale/bias の索きも丸ごと消える** |
| 3 | `vision_rmsnorm_bf16w_block` | 既存の `prefill_rmsnorm_bf16w_block` を D=1152 で呼ぶだけ | **流用 (新規カーネルなし)** |
| 4 | `vision_qk_norm_rope2d_block` | head ごとの q/k RMSNorm(72) + v の no-scale RMSNorm + 2D RoPE | `prefill_rope_apply_neox_pair` を x/y で 2 回呼ぶ。既存 `PrefillQKVEpilogue` の構造を写す |
| 5 | `vision_attention_full_tiled` | 非 causal・全可視・scale 1.0 の attention | ~~`attention_prefill_causal_tiled` から派生~~ → **`attention_prefill_causal_qblock_impl` から派生** (§0-A-3)。tiled は既にフォールバックに降格している。ただし **headDim 72 は `kElemsPerLane × 32` に載らない** (72 / 32 = 2.25) ので、レーン割り当てはそのままでは使えない。V3 で決める |
| 6 | `vision_mlp_act_block` | `gelu_tanh(gate) * up` | 既存 `prefill` の gelu と同じ式 (`prefill.metal:27`) |
| 7 | `vision_pool_project_block` | 3×3 平均 → ×sqrt(1152) → `(h-bias)*scale` → no-scale RMSNorm → [1152→2816] | 新規 (GEMM は #2) |

- tower は **prefill の 1 回だけ**走る。decode 経路は 1 行も変わらない。
- 実行タイミング: `prefillChunked` に入る**前**に全画像を処理し、
  `[S, 2816]` の fp16 バッファ (画像ごと) を作る。チャンクループの中で
  tower を呼ばない (§3-2 の対策 3)。
- 重みは vision resident buffer から `TensorView` で引く。Model に
  `visionQProj(layer:)` 等のアクセサを足す (テキスト側と同じ形)。

### 4-5. ランタイム: prefill への統合 (**最大の設計課題**)

#### (a) チャンク境界が画像スパンを割ってはいけない

> **この項は §0-A-1 で置き換わった。**以下の「128 なので必ず割れる」は
> `PREFILL_THROUGHPUT.md` §7-2 で上限も既定も **2048** になったため成立しない。
> スパン (最大 280) は既定チャンクに丸ごと収まるので、
> **スクラッチの拡張も KV リングの拡張も不要**。
> 残る仕事は「境界をスパン内部に落とさない」だけで、
> 却下案 (`allowedPrefillChunkTokens` を太らせる) の懸念も消えている。
> 以下は記録として残す。

`PrefillRuntimeConfig.maxChunkTokens = 128`、
`RuntimeConfiguration.allowedPrefillChunkTokens = [32, 64, 128]`、
`PrefillChunkScratchLayout` も 128 で clamp (**実測**、3 箇所)。
画像スパンは最大 280 なので**必ず割れる**。割れると、スパン後半の K/V が
まだ KV キャッシュに書かれていない状態で前半の attention を計算することになり、
双方向にしようがない (`copyPrefillKVToCache` → attention の順で、
`kvValidCount` はチャンク末尾まで、**実測**)。

**採る案: チャンクプランナを画像スパン対応にする。**

```
PrefillChunkPlanner.spans(tokenCount:startPosition:chunkTokens:imageSpans:)
  - テキスト区間は従来どおり chunkTokens (既定 128) で刻む
  - 画像スパンは「1 スパン = 1 チャンク」で丸ごと出す (最大 280 トークン)
  - スクラッチは max(chunkTokens, maxImageSpanTokens) で確保
```

- `maxChunkTokens` の定数は 128 のまま (テキストの既定挙動は不変)。
  スクラッチ側の clamp だけ「画像がある実行では最大 280」に広げる。
- 影響: prefill スクラッチ 13 MB → 33 MB (**導出**)、
  FP16 KV リング容量 `slidingWindow + chunkTokens` = 1152 → 1304
  (画像のある実行のみ、KV +約 0.03 GB、**導出**)。
- **却下した案**: `allowedPrefillChunkTokens` に 320 を足して全体を大きくする。
  テキストのみの実行まで KV とスクラッチが太り、`bench.sh` の回帰が動く。
  「vision を足したらテキストが遅くなった」は最悪の失敗の仕方。

#### (b) 双方向マスク

> **実装位置は §0-A-3 に移った。**既定経路は
> `attention_prefill_causal_qblock_d256` / `_d512` なので、
> 下の `last_exclusive` ではなく **`window_first[j]` / `window_last[j]`** を
> 書き換える。`block_last` は自動的に広がるのでキーループの構造は不変。
> tiled 側 (フォールバック) にも同じ意味で入れる — 入れないと
> フォールバック時に黙って causal に落ちる。
> **範囲は `<|image|>` × n だけ** (boi/eoi は含まない、§0-C-3)。

`PrefillAttentionParams` に `visibleEndBuffer` (device `uint*`、長さ = queryCount) を足す:

```metal
uint last_exclusive = min(p.kvValidCount, abs_q + 1u);
if (p.hasSpanMask) last_exclusive = min(p.kvValidCount, span_end[t]);   // ← 画像内は末尾まで
```

- `span_end[t]` は既定で `abs_q + 1`、画像スパン内のクエリだけスパン末尾 (排他)。
  §2-7 の「実効的に未来方向だけ開ける」に基づく。スパン ≤ 280 < window 1024 なので
  `first` (window 下端) の緩和は不要 (**導出**、V4 で参照比較により裏を取る)。
- **sliding 層にだけ適用**する (§2-7)。full attention 層は `hasSpanMask = false`。
- TensorOps 2D 経路 (macOS 26 / Apple10) は `hasSpanMask` のとき**無効化する**。
  この機体では元から未使用 (**実測**) だが、macOS 26 で黙って causal に落ちるのを防ぐ。
  MPP を group 32 で塞いだときと同じ扱い (PLAN_QAT §3-1)。

#### (c) 埋め込みの差し替え

```
prefill_embed_lookup_int4_block(..., outScale: sqrt(2816))   # 既存のまま
vision_scatter_soft_tokens_block(hidden, soft, offsets, count)  # 追加 (fp16 コピー)
```

チャンク内の画像位置に、**sqrt(hidden) を掛けずに**書き込む (§2-6)。
scatter はチャンク開始位置からの相対で行う。

#### (d) トークン列の生成

`applyChatTemplate` は現状 text-only を明示している (`Tokenizer.swift:305-307`)。
`Message.content` を `[ContentPart]` (text / image) に拡張し、
テンプレートと同じく画像を `<|image|>` 1 個としてレンダリングしたうえで、
**処理系側で `boi + image_token × n + eoi` に展開する** (§2-1、`processing_gemma4.py:150`)。
展開後のトークン列と、各画像スパンの `(startOffset, count)` を一緒に返す。

> **同時に塞ぐ穴 (**実測**、調査資料 §3-1-c):** いまはユーザーが本文に
> `<|image|>` と書けば、特殊 ID が**普通の埋め込みとして通って黙って崩れる**。
> 画像が添付されていない `<|image|>` / `<|audio|>` / `<|video|>` は
> **エラーにする** (V1 で入れる。vision 本体より先に入る安全策)。

### 4-6. 入口 (CLI / Server)

**CLI**

```
--image <path>            # 複数回指定可。--messages-file の最後の user ターンに付く
--image-tokens {70,140,280}   # 既定 280。上流の _SUPPORTED_SOFT_TOKENS に従う
```

`--messages-file` の JSON も content parts を受ける形に拡張する
(`{"role":"user","content":[{"type":"text",...},{"type":"image","path":...}]}`)。
文字列 content は従来どおり動く。

**Server (OpenAI 互換)**

`OpenAITextPart` を `OpenAIContentPart` に拡張し、`image_url` を受ける:

```json
{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
```

- **`data:` URI のみ受ける。`http(s)://` の取得はしない** (V6 時点)。
  サーバがユーザー入力の URL を取りに行くのは SSRF そのもので、
  ローカル推論サーバに必要な機能ではない。明示的に 400 を返す。
- 上限 (画像枚数 / デコード後ピクセル数 / base64 バイト数) を設定可能にし、
  超過は 413 相当のエラーにする。
- `usage.prompt_tokens` は soft token を含んだ実トークン数を返す (既存の数え方のまま)。
- **`--prompt-cache-mode single-prefix` は画像を含むリクエストで無効化する。**
  `ServerPromptCache` はメッセージのテキスト等価性で当てている (**実測**) ので、
  同じ文言で画像だけ違うリクエストが**誤ヒットする**。画像対応のキー設計は
  スコープ外にし、「画像があればキャッシュしない・publish もしない」で塞ぐ。

### 4-7. 触らないもの

decode 経路、MoE、KV レイアウト、sampling、expert streaming、group 32/64 の両対応。
tower は prefill の前段でしか動かないので、**§5-0 のテキスト回帰が動いたら
それは実装ミスの証拠**として扱う。

---

## 5. Phase 分解

| Phase | 内容 | 出口条件 |
| --- | --- | --- |
| **V0** | 参照系の固定 | 下記 §5-V0 — **完了 (2026-08-17、§0-B)** |
| **V1** | 前処理 + トークン列 (GPU なし) | 参照 fixture と patch 一致 (許容 §6-1)。画像なしの `<\|image\|>` が**エラーになる** — **完了 (2026-08-17、§0-C)** |
| **V2** | フォーマット拡張 + repacker | `--include-vision` で `.gturbo` が出来、**旧バイナリが flag で拒否する**。`--include-vision` なしの出力が現行とバイト一致 — **完了 (2026-08-17、§0-D)**。ただし旧バイナリでの実拒否確認は V6 に持ち越し (§0-D-6) |
| **V3** | tower カーネル + 単体検証 + **性能実測** | `TurboFieldfareKernelCheck` に vision ケース追加で全 PASS + **検出力の裏取り**。bf16 QMM の実測 GFLOP/s を記録 |
| **V4** | tower 統合 (画像 → soft token) | 参照実装の `pooler_output` と相対誤差 ≤ §6-2 の閾値 |
| **V5** | prefill 統合 (スパン・マスク・scatter) | 実画像で説明が成立。**テキストのみの回帰 ±1%** (§5-0 相当) |
| **V6** | CLI / Server の入口 + 受入 | §7 のゲート、`RESULTS_VISION.md` |

### 5-V0. 参照系の固定 (**これを飛ばさない**)

PLAN_QAT で最も高くついた発見は「評価ハーネスが暗黙に固定していた設定が、
評価結果そのものを決めていた」だった (`RESULTS_QAT.md` §1)。vision では
**カーネルのバグ・前処理の差・チェックポイントの素の性能**の 3 つが
すべて「なんとなく変な説明文」に見えるので、参照系がないと判定不能になる。

```
scratch/vision-venv/            uv + torch (CPU) + transformers==5.6.2 + pillow
scratch/vision-weights/         Google QAT リポジトリから vision 356 本だけ range 取得 (1.15 GB)
scratch/vision-fixtures/        参照ダンプ (下記)
```

`Gemma4VisionModel` + `Gemma4MultimodalEmbedder` を vision_config と 356 本の重みだけで
単体構築できる (**導出**、`Gemma4Model.__init__` は tower を独立に組む)。
**26B のテキスト側をロードする必要はない。**

ダンプする中間結果 (テスト画像 3 枚 × soft token 70/280):

| fixture | 用途 |
| --- | --- |
| `pixel_values` [P,768] / `pw,ph` | 前処理の突き合わせ (V1) と、**前処理を迂回**した tower 検証 (V4) |
| patch_embedder 出力 [P,1152] | §2-2 の検証 |
| layer 0 / 13 / 26 の出力 [P,1152] | 層ごとの発散点の特定 |
| pooler 出力 [S,1152] | §2-5 の検証 |
| `pooler_output` (projector 後) [S,2816] | **V4 の出口条件そのもの** |
| 素の JPEG/PNG | end-to-end (V6) |

**出口条件:** 上記が生成でき、Swift 側から読める素朴なバイナリ形式で保存されていること。
これが出来ないうちは V3 以降に進まない (§8)。

---

## 6. 検証プロトコル

### 6-1. 三層に分けて、混ざらないようにする

| 層 | 比べるもの | 目的 | 許容 |
| --- | --- | --- | --- |
| **A. カーネル単体** | GPU カーネル vs CPU 参照 (ハーネス内の素の二重ループ) | 実装バグの検出 | 相対誤差 ≤ 1e-3 (bf16×fp16 累算の既存基準、PLAN_QAT §5-A と同じ桁) |
| **B. tower 全体 (前処理を迂回)** | 参照の `pixel_values` を**そのまま食わせた**うちの soft token vs 参照の `pooler_output` | 「アルゴリズムの写し間違い」の検出。**リサイズ差が混入しない** | 相対誤差 ≤ 2e-2 (fp16 で 27 層積む前提、V4 で実測してから確定) |
| **C. end-to-end** | 素の画像 → うちの soft token vs 参照 | 前処理 (bicubic) 差の実害を測る | **B との差分**として報告。閾値は設けず、超えたぶんはリサイズ差と特定する |

**C だけを見て判断しない。**C が悪いとき、B が通っていればリサイズ、
B も落ちていればカーネル、と一意に切り分かる。これが §4-3 で
「ビット一致を諦める」と書ける根拠になっている。

### 6-2. ハーネスが暗黙に固定しているもの (先に列挙する)

`RESULTS_QAT.md` §1 の教訓。V0 の時点で書き出し、V6 の判定でも参照する:

- soft token 数 (70/140/280) — 画像ごとに変わる (§2-1)
- リサイズのバックエンド (torchvision / PIL / vImage / CGContext) と antialias
- `--thinking on|off` (画像課題でも推論の有無で答えが変わる)
- 双方向マスクを sliding 層のみに入れるか両方か (§2-7)
- 温度 (ベースライン測定は temp 0、`bench.sh` の既定は 1.0 — PLAN_QAT の落とし穴)

### 6-3. 検出力の裏取り (**必須**)

PLAN_QAT で 7 箇所のジオメトリバグを潰したとき、価値があったのは
「PASS した」ことではなく「**壊すと FAIL する**ことを確認した」ことだった。
vision でも同じ手続きを取る:

| わざと壊す | 期待 |
| --- | --- |
| 2D RoPE の x/y を入れ替える | B が FAIL |
| pooler の並びを列優先にする | B が FAIL |
| soft token に sqrt(2816) を掛ける | B が PASS / **C の実機出力が壊れる** → C 側の検出力の確認 |
| 双方向マスクを外す (causal のみ) | B は無関係 / **実機の説明品質で差が出るか**を観察 (出なければ §2-7 の判断自体を再検討する材料) |

比較関数は **NaN-safe な `relativeError` を使う** (`RelError.compute` は
`max(0, .nan) == 0` で NaN を PASS にする、`PLAN_QAT.md` §4 で顕在化済み)。
コマンドバッファのエラー検査と「参照側に信号があること」の検査も同様に入れる。

### 6-4. テキストのみの回帰 (V5 / V6 の各段階)

```bash
TEMP=0 MAXNEW=384 ./bench.sh ja      # 3 回インターリーブ中央値、48 スロット
```

> **prefill 側のベースラインは §0-A-5 で差し替わった。**pp は
> `PREFILL_THROUGHPUT.md` §7-9 の **231 pp / GPU 8.11 s / peak 6.64 GB**
> (2478 トークン) に対して測る。下の `bench.sh ja` は decode の話なので
> QAT ベースラインのままでよい。

`RESULTS_QAT.md` の QAT ベースラインに対し **±4% 以内** (実際には ±1% を期待。
tower は off-path なので、動いたら実装ミス)。
`load=` の秒数も見る — vision を別ファイルにした狙い (§4-1) が効いているかは
**テキストのみ実行の load 時間が変わらないこと**で確認できる。

---

## 7. 受入ゲート (V6)

| # | ゲート | 基準 |
| --- | --- | --- |
| 1 | 正しさ (B) | §6-1 の閾値内 |
| 2 | 目視 | 実写 3 枚 + 図表 1 枚 + 日本語の質問で、**画像を見ていないと書けない内容**が出る (色・数・位置関係を問う) |
| 3 | テキスト回帰 | §6-4 で ±4% 以内、`load` 時間が現行と同水準 |
| 4 | TTFT | 画像 1 枚 (S=280) の TTFT を**実測して記録する**。事前の合格線は引かない (§3-2 の見積り幅が大きすぎる) が、**10 s を超えたら既定 soft token を 140 に落とす判断を V6 で行う** |
| 5 | メモリ | footer の peak < 12 GB、`ExpertCacheBudget` が 48 スロットを通す |
| 6 | Server | `data:` URI で 200、`http(s)` URI で 400、画像ありでプロンプトキャッシュが**publish されない**こと |
| 7 | 起動 | `--verification trusted-install` / `full-sha256` の両方で exit 0 |
| 8 | 退行なし | `--include-vision` なしで作った `.gturbo` が現行とバイト一致 (V2 の出口条件の再確認) |
| 9 | 旧ランタイム拒否 | vision 付き `.gturbo` を **vision 以前のタグでビルドしたバイナリ**に渡し、`unknown v1 flag` で exit != 0 になることを実機で 1 回確認する (§0-D-6) |

記録は `RESULTS_VISION.md` に PLAN §6 準拠 (commit / ハード / コマンド / exit code /
footer 全文 / プロトコルからの逸脱すべて)。

---

## 8. 中止条件

- ~~**V0 の参照ダンプが作れない** → 以降に進まない~~ **解除 (2026-08-17)。**
  §0-B で 6 ケース分の参照ダンプが取れた。「カーネルのバグ」と
  「チェックポイントの素の性能」を分離する手段は確保できている。
- **V3 の bf16 QMM 実測が ~~0.1~~ 1.0 TFLOP/s を下回る** → S=280 で 3.5 s 超になる。
  カーネル設計を戻すか、既定 soft token を落とすかを決めるまで V4 に進まない。
  **線を 10 倍上げた**のは、比較対象が scalar QMM (0.59 TFLOP/s) ではなく
  タイル化後の射影 (3.53 TFLOP/s) になったため (§0-A-5 ではなく §0-A の 5 行目)。
  0.1 TFLOP/s 台は今や「何かを致命的に間違えた」水準で、中止条件として機能しない。
- **V5 でテキストのみの回帰が ±4% を超える** → 設計を見直すまで先に進まない。
- **双方向マスクが実装できない構成しか取れない** (チャンク上限を上げるとテキストが
  退行する等) → causal 近似で通すのではなく、いったん報告して判断を仰ぐ。
- 12 GB 予算を守れない構成しかない → 報告して停止。

---

## 9. 明示的にやらないこと

- **音声・動画。**トークン (`<|audio|>` 258881 / `<|video|>` 258884) と
  `audio_config` は存在するが、tower も前処理も別物。**明示的に拒否する**だけ入れる。
- **Mac GUI (`TurboFieldfareApp`)。**ユーザー指示。ビルドが通り、
  テキストで従来どおり動くことだけ守る。
- **tower の量子化。**bf16 のまま常駐させる。int4 化は QAT 由来でない再量子化に
  なるので品質リスクが読めない (調査資料 §3-2 と同じ判断)。
- **プロンプトキャッシュの画像対応。**画像があれば無効化する (§4-6)。
- **サーバからの画像 URL 取得。**`data:` URI のみ。
- **tower の最適化。**V3 で実測し、必要なら既定 soft token を下げる。
  カーネルのチューニングは本 PLAN の外 (PLAN_QAT §8 と同じ線引き)。
- **`image_processor` のビット一致再現。**§4-3 / §6-1 のとおり分離して測る。

---

## 付録 A: 本 PLAN で調査資料を訂正した点

| 調査資料の記述 | 本 PLAN での訂正 | 根拠 |
| --- | --- | --- |
| 「画像 1 枚 = 280 soft token」 | **280 は上限。**実数はアスペクト比依存 (`pw*ph/9`) | §2-1 (**実測**、`processing_gemma4.py`) |
| 「280 soft token の生成方法は**未確認**」 | 全段確定 (patch→27 層→3×3 平均→√1152→標準化→no-scale RMSNorm→projector) | §2 (**実測**) |
| 「`processor_config.json` は未取得」 | 取得済み。`do_normalize: false` が要点 | §1-3 (**実測**) |
| 「vision ウェイトはチェックポイント間で共有できない」 | 正しい。**加えて**、ローカル QAT snapshot が Google QAT リポジトリの変換物であることをバイト一致で確定した | §1-2 (**実測**、新規) |
| 「双方向 attention」 | 参照実装では **sliding 層のみ**。full attention 層は causal のまま | §2-7 (**実測**) |
| 「vision テンソルは resident に入れられる」 | 入れられるが**入れない**。`model_weights.bin` は毎回 eager に SHA-256 される | §4-1 (**実測**) |
| 「チャンク prefill との相互作用は未確認」 | ~~上限 128 < スパン 280 で**必ず割れる**~~ 上限は 2048 になった。既定チャンクにスパンは収まるが、境界に跨がらせない責任はプランナ側に残る | §4-5 (**実測**) |

## 付録 B: リモート観測ログ (2026-08-16)

- `GET /api/models/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` —
  sha `f1e06dc520982d9b9edd76859fdb7ab209449949`、lastModified 2026-07-20、gated false、
  shard 2 本 (合計 51,611,872,412 B)、`processor_config.json` あり。
- `model.safetensors.index.json` (103,196 B) — SHA-256 `907826a6e46ff4…`、
  1,013 テンソル、vision 356 本は全て shard 1。
- shard 1 の safetensors ヘッダ (先頭 129,808 B、Range 取得) — vision 合計
  1,145,588,832 B。内訳は §1-1。
- `model.language_model.{norm,layers.0.input_layernorm,layers.7.post_feedforward_layernorm}.weight`
  を Range 取得し、ローカル `scratch/qat-aligned-snapshot/` の対応テンソルと
  SHA-256 比較 → **3 本とも一致** (§1-2)。
- 参照実装: `huggingface/transformers` タグ `v5.6.2` の
  `src/transformers/models/gemma4/{modeling,processing,image_processing,image_processing_pil}_gemma4.py`
  を取得して読解 (§2)。
