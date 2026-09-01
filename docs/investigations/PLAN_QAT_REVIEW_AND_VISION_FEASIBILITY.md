# 調査: PLAN_QAT の再発明チェックと Vision 拡張の実現可能性

作成: 2026-08-17
スコープ: 調査のみ。ソース・ランタイム既定・モデルファイルは一切変更していない
(リポジトリは QAT 改造の別タスクが進行中。本調査は読み取りとリモート取得のみ)。
表記は PLAN.md に合わせる: **実測** / **導出** / **未確認**
リモート事実は 2026-08-17 時点の Hugging Face API / HTTP Range 取得による
ピン留めなしの観測。リビジョンや内容は上流で変わりうる。

---

## 1. 結論 (要約)

| 問い | 結論 |
| --- | --- |
| PLAN_QAT は OPTIMIZATION_JOURNEY の車輪の再発明か | **していない。**両文書の目的が直交し、実験記録にも QAT / group 32 / bf16 ルーターの前例はない (**実測**)。ただし PLAN が踏襲すべき教訓のうち 1 件 (アライメント) は明示されていないので §2-3 に注意点として残す |
| Vision 機能は原理的に追加可能か | **可能。**重みは DL 元の Google QAT リポジトリ自体が全量保持している (**実測**、§3-1)。ただし配布状態 (.moepack v1 = text-only) からの拡張であり、**ワイヤフォーマット拡張を含む本格的な機能追加**になる (§3-3)。前提の一部 (vision ウェイトのチェックポイント間共有) は実測で崩れているため、重みは必ず QAT リポジトリから取ること (§3-1-b) |

---

## 2. PLAN_QAT × OPTIMIZATION_JOURNEY — 再発明チェック

### 2-1. 両文書の目的は直交している

- `docs/OPTIMIZATION_JOURNEY.md`: **同じモデルを速く動かす**ための実験史
  (explicit reads、persistent MoE、ベクタ化、split-KV、chunked prefill、sampling)。
  前提は「モデルは 14.3 GB のまま」。
- `PLAN_QAT.md`: **チェックポイントの載せ替え** (group 32 / bf16 ルーター)。
  §8 で「最適化作業はやらない。本 PLAN は『載せる』まで。性能は測定のみ」と
  明示的に最適化を除外している。

`docs/experiments/EXPERIMENT_INVENTORY.md` を `qat|vision` で検索しても
前例はない (**実測**)。group サイズのパラメータ化、bf16 ルーター経路、
QAT チェックポイントのいずれも本リポジトリで扱われたことがない。

### 2-2. PLAN が既に journey の教訓を踏襲している箇所

| OPTIMIZATION_JOURNEY の教訓 | PLAN_QAT での対応 | 判定 |
| --- | --- | --- |
| 局所結果ではなくエンドツーエーンドで判定 («What local results missed») | §5-2 の性能ゲートは `bench.sh` の全体測定。§5-A ハーネスはあくまで出口条件 | 踏襲済み |
| サーマル/ページキャッシュの交絡 → 交互測定 («The method that worked») | §5-0 / §5-2 は 3 回インターリーブ中央値 | 踏襲済み |
| 変更の種類に応じた正確性検査 (exact / FP 再順序) | §5-A は相対誤差 + group 64 恒等性で「意味不変」を担保 | 踏襲済み |
| staged affine MPP が prefill の production 経路 | §3-1 で group ≠ 64 の MPP 無効化ゲートを明示的に入れる (macOS 15 では未使用だが macOS 26 での黙った誤動作防止) | 踏襲済み |
| packed KV の失敗 (メモリ不利 + 品質不合格) | PLAN は KV に触れない | 無関係で正常 |

### 2-3. 注意点 (再発明ではないが、journey の知見で PLAN に明示されていないもの)

1. **アライメントの罠 (最重要)。** journey「Vectorization helped when it
   respected the storage layout」の核心: 32-bit pack 読み込みは
   offset-0 のフィクスチャを通り、実テンソルの 2-byte アライメントで壊れた。
   Phase B の新カーネル `router_gemv_gemma4_bf16_r4` は bf16 行列を読む
   GEMV であり、まさにこの罠の形をしている。§5-A のケース 6 は合成入力
   (offset 0) なので、この種のバグを検出できない可能性がある。
   journey の「realistic tensor offsets part of later kernel tests」という
   運用に合わせ、bf16 ルーターを**実際の resident ファイル上の offset**
   (奇数 2-byte 境界) から読むケースを足すことを勧める。
2. **定数の書き換え時期。** journey 冒頭の「1.35 GB common / 12.9 GB pool /
   14.3 GB」は group-64 pin の数字。QAT 受入後は 1.51 GB / 14.28 GB になる
   (PLAN §2)。journey / SYSTEM_DESIGN の数値更新は PLAN のスコープ外だが、
   受入後に必要になることを記録しておく。
3. **スロット数の前提差。** journey の expert キャッシュの数字 (16-slot LFU)
   は 8 GB 機のもの。PLAN は 18 GB 機 64 スロットで測っており、ベースライン
   (RESULTS §3-4) も同スロットなので比較は公平。I/O 命中率を journey の
   数字と直接比較しないこと。

**結論: 重複なし。** PLAN_QAT は journey の未踏の領域にあり、方法論も
journey の結論に沿っている。

---

## 3. Vision 機能の追加 — 実現可能性

### 3-1. 前提事実

#### a. 重みは DL 元の Google QAT リポジトリ自体にある (**実測**、最重要)

`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` を HF API + Range 取得で
直接確認した:

- **vision テンソル 356 本、合計 1,145,588,832 B ≈ 1.15 GB (bf16)。**
  `model.vision_tower.*` (tower 本体) + `model.embed_vision.embedding_projection.weight`
  [2816, 1152] (テキスト側 hidden への projector)。
- shard 1 (49.9 GB) 内に散在 (`embed_vision` は先頭、tower は末尾付近)。
  ただし現行 repacker と同じ **range リクエスト方式なら 50 GB全文の DL は不要**
  で、必要な range 群 (~1.15 GB + ヘッダ) だけ取れる。
- `vision_config` (実測、全文取得): SigLIP 系の構成 — 27 層 / hidden 1152 /
  head 72 × 16 / intermediate 4304 / patch 16 / rope_theta 100 /
  `position_embedding_size` 10240 / `standardize: true` /
  `default_output_length` 280。
  画像の標準化パラメータそのものがテンソルとして入っている
  (`vision_tower.std_scale` / `std_bias`、各 [1152] bf16)。
- `processor_config.json` も存在 (1,689 B、未取得)。前処理の詳細はここ。

つまり「QAT ウェイトには Vision が含まれていない」のは正確には
**mlx-community 変換 (`qat-q4_0-mlx-aligned`) が text-only に落とした**ためで、
Google 元リポには完全に入っている。ローカル snapshot
(`scratch/qat-aligned-snapshot/`、1,279 テンソル) に vision がないのも変換由来。

#### b. チェックポイント間で vision ウェイトは共有できない (**実測**)

現 pin 由来 (`mlx-community/gemma-4-26b-a4b-it-4bit`) の
`vision_tower.std_scale` / `std_bias` のバイト列を Google QAT リポジトリの
ものと Range 取得で直接比較したところ、**両者は不一致**。
Gemma 3 系の「QAT では vision tower が凍結されていて非 QAT 版と同一」という
経験則はここでは成り立たないと考えるべき。

→ **QAT テキスト重みと組む vision 重みは、必ず同一の Google QAT リポジトリ
から取得すること。** pin の shard 3 から流用する案はこの実測で閉じた。

#### c. トークナイザと設定は既に画像入力を前提している (**実測**)

- snapshot の `tokenizer.json` に画像系トークンが存在:
  `255999 <|image>`、`258880 <|image|>`、`258882 <image|>` のほか
  audio/video 系も含め 8 種 (調査資料「byte-identical tokenizer」と整合)。
- snapshot の `config.json` は `image_token_id: 258880` /
  `boi_token_id` / `eoi_token_id` / `vision_soft_tokens_per_image: 280` /
  `use_bidirectional_attention: "vision"` を保持
  (ただし `vision_config` ブロック自体は mlx 変換で落とされている)。
- 現行ランタイムに画像トークンを含むプロンプトを渡した場合、特殊 ID は
  通常の埋め込みとして通るため、**クラッシュせず黙って崩れた出力になる**
  (**導出**、構造からの帰結)。

#### d. `.moepack` v1 の収容力 (**実測**、ソース読み)

- **ResidentIndex は名前付きの汎用テンソル目録** (dtype / shape /
  offset / scale / bias。bf16 dtype=1 を持つ)。物理的には vision テンソルを
  そのまま格納できる。`Model.validateRuntimeSchema` は要求テンソルの
  存在のみ検査し、余分なエントリを拒否する検査は見当たらない。
- 一方 **manifest.arch は固定フィールド** (hiddenSize / numHeads / …) で
  vision のアーキ情報を載せる場所がなく、`flags` は既知集合のみを受入
  (未知の flag は v1 リーダが拒否)。→ **vision 対応にはワイヤフォーマットの
  バージョン管理された拡張が必須。** `Sources/MoEPackFormat/` は
  Foundation-only の固定契約なので、拡張は v1 の後方互換追加または v2 として
  明示的に設計する必要がある。
- repacker は `vision_tower.*` / `embed_vision.*` を
  `isMultimodalTensorName` で明示的に除外している (audit の
  `tensors_dropped_multimodal`)。vision 化はこの除外の opt-in 反転。

### 3-2. 原理的な可否: **可能**

理由:

1. **テキスト側の推論経路は画像でも不変。** 画像は「事前計算された 280 トークン
   分の埋め込み列」として prefill に入るだけで、decode・MoE・KV・sampling は
   一切変わらない。モデル自体がマルチモーダル学習済み
   (`Gemma4ForConditionalGeneration`) であり、能力を後付けするのではなく
   入力経路を復元する。
2. **必要な部品がすべて既知。** (a) 画像前処理 (resize + 標準化、パラメータは
   重み同梱)、 patch / conv 埋め込み、 27 層の小規模 transformer tower
   (bf16、常駐 1.15 GB)、 projector (1152 → 2816)、
   (e) prefill 埋め込み差し替え 280 位置、
   (f) 画像スパン内の双方向 attention マスク。
   (e)(f) 以外は既存の GEMV / attention プリミティブの再構成で書ける。
3. **重みが入手可能** (§3-1-a)。かつ、テキスト重みと同一 QAT リポジトリ由來で
   整合性が保証される (§3-1-b の制約を満たす唯一の経路)。

メモリ (**導出**): tower は常に常駐させるなら bf16 で +1.15 GB。
QAT 64 スロット構成の予想 peak 約 9.3 GB (PLAN §2) と合わせても
約 10.5 GB で 12 GB 予算内。tower は prefill にしか使わないため、
遅延ロード (prefill 直前に mmap/pread、生成中は解放) も選択肢。
int4 化 (~0.33 GB) も物理的には可能だが QAT 由来でない再量子化に
なるため品質リスクが強く、第一候補からは外す。

KV (**導出**): 画像 1 枚 = 280 位置 × 30 層ぶんの KV エントリ。4K コンテキスト
に数枚載せる分には既存 FP16 KV 予算内 (PLAN 1-7 の 0.28 GB @4K の枠で
画像ぶんがテキストぶんを置き換えるだけ)。

### 3-3. 必要な改造の棚卸し (スコープ見込み)

| # | 領域 | 内容 | 規模感 |
| --- | --- | --- | --- |
| 1 | `MoEPackFormat` | manifest への vision セクション (アーキ、quant スロット、soft-token 数、boi/eoi/image token id)。未知 flag を拒否する現行検証との両立 — バージョン方針の明示 | 中 (契約変更なので検証もセット) |
| 2 | repacker | `--include-vision` 相当の opt-in。**ソースが二股になる点に注意**: ローカル QAT snapshot に vision がないため、テキストはローカル snapshot、vision は Google リポジトリの range 取得という dual-source が必要。指紋 (SHA-256) の管理対象が 2 リポジトリに広がる | 中 |
| 3 | runtime: 画像前処理 | 画像デコード・resize・標準化 (`std_scale`/`std_bias` は重みから読む)。`processor_config.json` の精査が前提 (**未確認**) | 中 |
| 4 | runtime: tower 推論 | patch conv、位置埋め込み、27 層 bf16 forward、RoPE (theta 100)。既存プリミティブで構成可能。prefill 時のみ実行 | 大 (新経路) |
| 5 | runtime: prefill 統合 | 画像位置 (280/枚) の埋め込み差し替え、**チャンク prefill の境界をまたぐ画像スパンの取り扱い**、画像スパン内の双方向 attention マスク (sliding / full 両 kernel)、RoPE position の連続付与 | 大 (既存 kernel への mask 追加を含む) |
| 6 | 入力経路 | CLI `--image`、サーバ (OpenAI 互換の content parts)、Mac アプリ UI。コンテキスト会計 (1 枚 = 280 token 扱い) | 中 |
| 7 | 検証 | HF `Gemma4ForConditionalGeneration` (transformers) 出力との参照比較 + 実画像の目視ゲート。PLAN §5 の手法をそのまま適用 | 中 |

### 3-4. リスクと未確認事項

| 項目 | 状態 |
| --- | --- |
| vision 重みの QAT 整合性 | 同一リポジトリ由来で原理的に解決。ただし mlx 変換相当の処理 (bf16 なら無変換で通るはず) を要確認 (**未確認**) |
| 280 soft token の生成方法 (Gemma 3 と同じ「tower 出力 → projector → 差し替え」か) | **未確認。**`processor_config.json` と `chat_template.jinja` の画像部分の精査が必要 |
| 双方向マスクとチャンク prefill / sliding window の相互作用 | **未確認。**実装設計の初期に確定させる |
| tower 実行のパフォーマンス | 27 層 × 280 token 程度の小さな prefill 相当。詳細は未測定。journey の教訓どおり、局所でなく全体で測る |
| QAT 改造 (別タスク進行中) との順序 | **vision は QAT 受入後に着手すべき。**同時に進めると §5 の判定が交絡する。本調査時点でワーキングツリーには Phase B/C 相当の変更が既に入っている (**実測**: `requireBF16Matrix`、`QATAlignedModelSource`、`LocalSnapshotByteProvider` 等) |
| フォーマット拡張の互換性 | v1 リーダは未知 flag を拒否するため、拡張は旧ランタイムが明確に拒否できる形にする (黙って部分読み込みさせない) |

### 3-5. 結論

Vision 機能の追加は**原理的に可能**であり、最大の障害だった「重みがない」は
実測で解消した (Google QAT リポジトリが 1.15 GB bf16 で保持、range 取得可)。
ただし (1) `.moepack` フォーマット拡張、(2) tower 推論と prefill 統合という
2 つの新規経路、(3) 三入口 (CLI / Server / App) への画像入力、を含む
本格的な機能開発であり、「配布状態からの改造」の枠を確実に超える。
着手は PLAN_QAT の受入手続き (§5-2) 完了後が自然。最初の一歩は
`processor_config.json` と chat template の画像部の精査、および
フォーマット拡張 (v1.x / v2) の契約設計。

---

## 付録: 本調査でのリモート観測ログ (2026-08-17)

- `GET /api/models/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` —
  gated: false。siblings 10 ファイル (shard 2 本、計 ~51.6 GB)。
- 同 resolve `model.safetensors.index.json` (103,196 B) — weight_map 1,013
  テンソル。うち vision 系 356。shard 2 (1.70 GB) はテキスト層のみ。
- 同 resolve shard 1 の safetensors ヘッダ (Range 取得) — vision テンソルの
  合計 1,145,588,832 B、offset 範囲 0 〜 49,907,116,700。
- `mlx-community/gemma-4-26b-a4b-it-4bit` の index + shard 3 ヘッダ —
  vision 系 358 テンソル (embed_vision の .scales/.biases ぶんだけ多い =
  projector が int4 化されている)。`vision_tower.std_scale` / `std_bias`
  のバイト列を Google QAT リポジトリ側と比較 → **不一致**。
