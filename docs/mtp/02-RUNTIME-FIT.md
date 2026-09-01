# 02. 性能のベースラインと、既存ランタイムとの対応

commit `156667c` の作業ツリーを読んで確認した (**実測**)。行番号はその時点のもの。

---

## 1. 性能のベースライン (**実測**、`RESULTS_VISION.md` §3 / §4)

M3 Pro 18GB / macOS 15.7.5。MTP の期待値はこの数字の上に立てる。

| 条件 | tok/s | decode hit | io/tok |
| --- | ---: | ---: | ---: |
| `TEMP=0 MAXNEW=384 ./bench.sh ja` haiku (64 スロット、中央値) | **32.226** | 97.3〜99.2% | 1.7〜6.8 ms |
| 同 math | **28.849** | 同上 | 同上 |
| 同 story | **31.371** | 同上 | 同上 |
| 画像 1 枚 + 64 トークン生成 (48 スロット既定、footer 実文) | **19.103** | 93.0% | 20.06 ms |

画像実行の footer 内訳: `[decode/tok io=20.06ms cb1=0.69ms cb2=0.29ms head=3.59ms]`。

**MTP の利益は expert キャッシュの温度に強く依存する** (**導出**):

- 温まった状態 (hit 97〜99%、io/tok 1.7〜6.8 ms) では、I/O は 1 トークン約 31 ms の
  うち数 ms しかない。verify で I/O を償却しても上限は数 %。
- 冷えた状態 / 48 スロット / 画像同居 (hit 93%、io 20 ms/tok) では償却の余地が大きい。
- **32 tok/s = 31 ms/tok に対し、footer の io + cb1 + cb2 + head では全体を説明できない。**
  残りが「トークンごとに必ず払う固定費」なら MTP はよく効き、「トークン数に比例する計算」なら
  効かない。**これを特定するのが M0 の第一項目** (04-PHASES §1)。

メモリのベースライン: テキストのみ peak **6.64 GB** / vision 同居 **6.97〜7.01 GB**。
ドラフター int4 は +0.24 GB で、12 GB 予算に対し余白は 5 GB 近い (**導出**)。

サンプリング既定: CLI `1.0` (`Sources/TurboFieldfareCLI/Args.swift:195`)、
Server `1.0` (`Sources/TurboFieldfareServer/Core/OpenAIModels.swift:405`)、
Mac アプリ `0.2`、`bench.sh ja` は `TEMP=0`。**受入は temp 0 と 1.0 の両方で測る。**

---

## 2. そのまま使える資産 (**実測**)

| ドラフターの要件 | 既存資産 | 場所 |
| --- | --- | --- |
| int4 affine GEMV (group 64) | decode 経路の int4 GEMV 一式。ドラフターの量子化形式は現行と同じ affine group 64 | `Kernels/Quant/`、`Model.affineGroupSize` (`Model.swift:44`) |
| GQA decode attention (SWA / full、scale 1.0) | 1 トークン decode の attention。head 構成がドラフターと一致 (01 §4) | `RealForwardRunner.produceToken` (`:1569-2086`) |
| 共有 KV の読み出し | `keyView(layer:validTokenCount:)` / `valueView(...)`。**層 28/29 の KV は既にそこにある** | `KVCacheManager.swift:171-199` |
| q_norm | `Model.qNorm(layer:)`。ドラフターは q_norm のみ | `Model.swift:194` |
| `layer_scalar` + sandwich 残差 | 同名概念が存在 | `Model.swift:241`、`Kernels/Fusions/FusedLayerTail.swift:35` |
| RMSNorm (重みをそのまま掛ける規約) | `prefill_rmsnorm_bf16w_block` | `Kernels/Prefill/Primitives/PrefillPrimitives.swift:47` |
| 2 系の RoPE (θ1e4 / θ1e6 + partial 0.25) | ターゲットが両方実装済み | `ModelTypes.swift:88-92`、`Kernels/Prefill/Primitives/PrefillRoPE.swift` |
| 埋め込み × √hidden | prefill の埋め込み経路が `outScale` を持つ | `Kernels/Prefill/` |
| tie 減 lm head の GEMV | `LMHeadChainInt4` (ドラフター用は D=1024 の小型版) | `Kernels/Fusions/LMHeadChainInt4.swift:56` |
| 別ファイル重みの遅延ロード | vision tower と同じ「manifest の optional セクション → 使うときだけ開く」 | `Model.swift:118-140` |

**ドラフター 1 ステップは約 0.55 GFLOP** (**導出**)。ターゲット 1 forward (active ~4B) より
1 桁以上軽い。

### Vision で入ってそのまま流用できる仕組み (**実測**)

| 資産 | 実体 | MTP での使い道 |
| --- | --- | --- |
| prefill の span 機構 | `RealForwardRunner.swift:594-612` の `PrefillChunkPlanner.spans(...)`、chunk ごとの行制御 (`:645-690`) | **k トークン verify はこの span 機構の最小ケース**。短いチャンクを 1 本流す経路が既にある |
| manifest の optional セクション | `GTurboManifestV1.swift:105-401` の `vision` + `flags.visionTower` + `versionMinor` ゲート + 「フラグとセクションは 1 つの事実を 2 回書く」検証 | ドラフター用セクションを同じ型で足せる (03 D1) |
| 旧ランタイムの明示的拒否 | `RESULTS_VISION.md` §8: 旧ビルドが `unknown key "visionTower"` で exit 1、対照は exit 0 | 同じ検出力つきの検査を書ける |
| 追記インストーラ | `--add-vision` (`VisionAppendInstaller.swift`)。tower 1.14 GB のみ取得、テキスト側 inode 不変 | `--add-draft` はこの写像 (03 D2) |
| プロンプトキャッシュの契約 | `RawCompletion.swift:260-270` の `kvBackedTokenIDs` / `uncommittedBoundaryTokenIDs`、「画像ターンは publish しない」前例 (`:115-122`) | 投機位置を公開しない不変条件を同じ形で書ける (03 D7) |

---

## 3. 新規に書くもの — 4 点

### N1. 検証 (verify) パス — **最大の実装項目**

現状:

- decode は `produce(token:position:into:)` で**厳密に 1 トークン**
  (`Runtime/Generation/LogitProducer.swift:10`)。
- `prefillChunked` は logits を**単一バッファに 1 行だけ**書く。書くのは最終 span の
  最終行で、`writeFinalHead` と `row: t - 1` がそれを固定している
  (`RealForwardRunner.swift:611`、`:1517-1546`)。

要るもの: **k トークンを KV に追記しつつ、k 位置すべての logits を出す forward**。

部品は揃っている (**導出**):

- prefill の attention / MoE は複数トークンを処理できる。routed MoE は expert ごとに
  行をまとめてタイル GEMM に流す (`Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift`、
  `PrefillRoutedGEMMPlanner`) ので、**k トークンの expert 和集合を 1 回で読む**という
  MTP が欲しい形に既になっている。
- span 単位でチャンクを流す経路も vision で入った。
- 足りないのは head だけ: `prefillFinalRowHead.encodeLogits(row: t-1, …)` を
  **k 行ぶん回す** (greedy なら `fusionHead.encodeGreedyDecode` を k 回)。
  実測 `head=3.59ms` × k で、**bs=3 なら +7 ms/ラウンド。ここが verify の主コスト** (**導出**)。

### N2. KV の巻き戻し — 小さい

`KVCacheManager` に `advance(by:)` (`:203-209`) の対称形がない。ただし実装は軽い:

- 物理スロットは `physicalSlot = position % capacityTokens[layer]` の
  **ステートレスな写像** (`:239-241`)。リングに書き込みカーソルの状態変数はない。
- `startSlot` も保持されず `validTokenCount` から毎回導出される (`:243-248`)。
- 可変状態は `public private(set) var position` **1 個だけ** (`:61`)。

→ 巻き戻し = `position` を減らすだけ。投機トークンのバイトは残るが、attention は
`[0, validTokenCount)` しか読まない契約 (`:18-21`) なので、次に同じ位置を書けば上書きされる。

**唯一の実質的な条件** (**導出**): SWA リング容量は `slidingWindow + maxPrefillChunkTokens`
(`:81-83`)。位置 p まで受理した後に投機 k トークンを書くと物理スロット
`(p+1..p+k) % capacity` を潰すので、それが「位置 p の窓 `[p-1023, p]` に今も入っている位置」と
衝突しない条件は `capacity >= slidingWindow + k`。既定 (chunk 128 → capacity 1152) で成立し、
実際の k は 2〜4。**init で precondition として明示する** (03 D4)。

### N3. ドラフターの forward 一式

| 要素 | 内容 |
| --- | --- |
| `pre_projection` GEMV | [1024, 5632] int4。concat 入力 (embed × √2816 と last_hidden) の組み立て込み |
| 4 層 (共有 KV) | 既存 decode attention を**ターゲット層 28/29 の KV ビューに向けて**呼ぶ。q は自前 |
| `post_projection` GEMV | [2816, 1024] int4 |
| ドラフター lm head | 埋め込み [262144, 1024] を線形として使う GEMV |
| **post-norm hidden の取り出し** | `LMHeadChainInt4.encodeGreedyDecode` (`LMHeadChainInt4.swift:56-70`) は `hidden` と `normWeight` の両方を取り、RMSNorm を**カーネル内部**で行う。呼び出し側 (`RealForwardRunner.swift:1517-1546`、`:2058`) が渡すのは**正規化前**の hidden で、`model.norm` 後の hidden はどのバッファにも残らない。ドラフターに渡すには rmsnorm を 1 本追加するか、正規化済み hidden も書き出す変種が要る (2816 要素なのでコストは無視できる)。**M3 で前者を実装済み** — `produceToken` の head 直前で probe 有効時だけ 1 本足す ([13-M3-RESULTS.md](13-M3-RESULTS.md) §7) |

### N4. ループと受理規則

`runRawCompletion` の while (`RawCompletion.swift:204-258`) は
「1 トークン sample → 停止判定 → `produce` して次へ」の形。MTP はここに
**bonus → ドラフト bs−1 → verify → 受理/巻き戻し → 受理ぶんをまとめて放出**を挟む。

同時に守るものが 4 つある:

1. **停止判定**は受理列を先頭から 1 個ずつ見る。停止トークン以降は捨て、KV を戻す。
2. **stop string / detokenizer** は逐次 push が前提 (`StreamingStopMatcher`)。
   受理トークンを 1 個ずつ push すれば形は変わらない。
3. **`position` ごとの seed** (`Sampler.seedFor(config:position:)`) を非投機実行と
   同じ値に保つ。これが出力一致 (03 D5) の根拠。
4. **プロンプトキャッシュの契約** — `kvBackedTokenIDs` / `uncommittedBoundaryTokenIDs` は
   「KV が実際に持っている列」を表す。投機で一時的に進んだ位置を外に出してはいけない。

---

## 4. 触らなくてよいもの (**導出**)

decode の 1 トークン forward 本体 (cb1 → io → cb2)、MoE、expert streaming、
sampler の中身、KV レイアウト、tokenizer、vision 経路。
ドラフターはこの経路の**外側と後側**にだけ挿入される。

例外は expert I/O のスケジューリングで、verify が複数トークンを一度に流すため
先読み判断 (`rdadviseSkipUntilPosition` 等、`RealForwardRunner.swift:463-470`) の
前提が「1 トークン 1 ルーティング」から変わる (05-RISKS R1)。
