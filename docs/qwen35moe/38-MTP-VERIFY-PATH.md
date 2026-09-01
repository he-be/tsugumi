# 38. 検証パスの費用を段ごとに測り、attention を差し替えた (実測(手元)、2026-08-22)

[37](37-MTP-POSTMORTEM-PLAN.md) は「残る 1.21 倍の内訳はホスト +11 ms の
route grouping と GPU +2.7 ms の prefill attention / block router」と書き、
そこから改善案 A (幅 2 特化のハイブリッド検証パス) を組み立てた。
その内訳は [36 §4-4](36-MTP-DECODE.md) が**未確認**と明示していたもので、
[37 §1-4](37-MTP-POSTMORTEM-PLAN.md) の規則 3 は「基底を実測せずに次の文書へ
進まない」である。**だから先に段ごとに測った。**

**結論を 1 行で: 内訳は外れていた。**ホスト側の route grouping は
**0.20 ms/パス**で、11 ms ではない。1.21 倍のほぼ全部は **pre-router の
コマンドバッファの GPU 時間**で、その中身は長文脈ほど支配的になる
**プロンプト用の attention カーネル**だった。差し替えは既にある
`Attention.encodeRows` (Gemma の投機で書かれた split-KV) を 1 行ずつ呼ぶだけで、
**2,698 トークンの要約が ×0.85 から ×1.15 に反転した。**

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | 37 の内訳は当たったか | **外れた。**`readPrefillRoutes` の汎用 grouping は幅 2 で **0.25 ms/パス** (decode の routes 段は 0.017 ms/tok)。11 ms は**どこにも無い**。[37 §4-A](37-MTP-POSTMORTEM-PLAN.md) の 3 番目の弾 (routing 読み戻しの幅 ≤ 2 特化) は**やる価値が無い** (§2) |
| 2 | では 1.21 倍はどこか | **pre-router の GPU。**幅 1 で decode の +7.2 ms/パス (t2、文脈 60)。文脈を 2,698 に伸ばすと **+28.2 ms** になる — つまり**文脈に比例する項**が主犯で、それは attention である (§2-2) |
| 3 | 何が悪かったのか | **カーネル選択がまた 1 つ。**[36 §4-3](36-MTP-DECODE.md) の密射影と同じ形。`attention_prefill_causal_qblock_d512` は 1 スレッドグループに 16 クエリを詰めて `ceil(T/16) × numQHeads` を投げるので、**T=1 でも T=2 でも 16 スレッドグループが KV 全体を歩く**。decode は KV を分割して並列に歩く。同じ 2,640 位置ぶんで decode は +5.0 ms/tok、T 行経路は **+26.0 ms** (§2-2) |
| 4 | 直し方 | **既にあるカーネルを呼ぶだけ。**`Attention.encodeRows` (split-KV、`docs/mtp/24-M5.5-RESULTS.md` の投機ブロック用) に差し替えた。**新しい Metal カーネルは 0 本** (§3) |
| 5 | 幅 2 を 1 発で流さない理由 | **投機の中立性が壊れるから。**`rowsGeometry` は KV の分割を `[0, startPosition + rows)` で切るので、同じトークンが行 1 で出るか行 0 で出るかで**加算順が変わる**。force-reject 対照との完全一致が 95/96 に落ちた。**行ごとに 1 発ずつ**投げると分割はその行の位置だけの関数になり、**96/96 に戻る**。代償は 0.4 ms/パス (§4) |
| 6 | 取り分 | §5 の A/B (192 トークン、腕は交互、3 腕) |
| 7 | 触っていないもの | プロンプトのプレフィル (T=512 では query-blocked が正しい)、MoE、router、巻き戻し、受理率 (P1 も a も動かない) |

---

## 1. 計器 — 段ごとの wall と GPU

`QwenStageProfile` (`Sources/Tsugumi/Runtime/Inference/QwenStageProfile.swift`)。
`TF_QWEN_STAGE_PROFILE=1` のときだけ働く。**decode ループと T 行チャンク経路を
同じ切り方で**計る — そうしないと引き算ができない。

| 段 | 中身 |
| --- | --- |
| `preRouter` | 入力 norm + attention か delta rule + residual + post-norm + router。層の 1 本目のコマンドバッファで、ホストが routes を読むので必ず join される |
| `routes` | router の出力の読み戻しと、MoE カーネルが取る形への変換。decode は 8 本のエキスパート ID、チャンク経路は `PrefillMoEGrouping` の sortedPairs / groups / tiles |
| `plan` | スロット割り当てと読みの発行 |
| `io` | その読みを待つ時間 |
| `shared` / `routed` / `tail` | 共有エキスパート / routed エキスパート / token-major reduce と residual |
| `drain` | 層が commit したまま join していないものの合流 |
| `head` | 508 MB の lm_head |
| `draft` | MTP ヘッドのパス |

`wall` はホストの時計、`gpu` はドライバの報告する GPU 時間で、
**deferred なコマンドバッファの GPU 時間は「commit した段」に付ける** —
join するのは次の層の `preRouter` なので、そうしないと全部が preRouter に寄る。
その代わり `shared` / `routed` の `host = wall - gpu` は負になる。段は入れ子に
しないので、`wall` の総和がその経路の壁時計になる。

## 2. 測ったもの — 1.21 倍はどこに居たか (**実測(手元)**)

`bench/qwen35/mtp_stage_profile.sh`、t2 コード (プロンプト 60 tok) と
t4 要約 (プロンプト 2,698 tok)、64 トークン、腕は直列・クールダウン 10 秒。
**この節の数字は [36](36-MTP-DECODE.md) と同じコード**である
(`TF_QWEN_MTP_ROWS_ATTN=0` の腕に相当)。

### 2-1. 段ごとの引き算 (t2 コード、文脈 60)

単位は decode が ms/トークン、T 行経路が ms/パス。幅 1 は 1 パス = 1 トークン。

| 段 | 素の decode | T 行・幅 1 | 差 | T 行・幅 2 |
| --- | ---: | ---: | ---: | ---: |
| `preRouter` wall | 29.38 | 38.13 | **+8.74** | 43.91 |
| — うち GPU | 16.66 | 23.90 | **+7.24** | 26.57 |
| `routes` | 0.017 | 0.201 | **+0.18** | 0.252 |
| `plan` | 0.257 | 0.260 | +0.00 | 0.321 |
| `io` | 19.48 | 23.10 | +3.62 | 34.53 |
| MoE の GPU (`shared`+`routed`+`tail`) | 8.30 | 8.36 | +0.06 | 11.67 |
| `head` | 4.25 | 4.70 | +0.45 | 5.97 |
| `draft` | — | — | — | 6.07 |
| **合計 wall** | **54.84** | **68.09** | **+13.25** | **93.17** |

**`routes` は 0.18 ms しか増えていない。**[37 §4-A](37-MTP-POSTMORTEM-PLAN.md) が
「ホスト +11 ms の主犯」と名指しした
`readPrefillRoutes` の層ごと汎用 grouping (16 ペアのソート + 256 要素の配列 2 本 +
重複検査の Set) は、**幅 2 では丸ごと 0.25 ms/パス**である。特化しても取れる上限が
0.25 ms なので、A の 3 番目の弾は**落とす**。

増えているのは `preRouter` の GPU (+7.24) と `io` (+3.62) である。

### 2-2. 文脈を伸ばすと符号が見える (t4、文脈 2,698)

| 段 (GPU) | 素の decode | T 行・幅 1 | T 行・幅 2 |
| --- | ---: | ---: | ---: |
| `preRouter` GPU、文脈 60 (t2) | 16.66 | 23.90 | 26.57 |
| `preRouter` GPU、文脈 2,698 (t4) | 21.68 | **49.90** | **57.69** |
| **文脈 +2,640 位置ぶんの増分** | **+5.02** | **+26.00** | **+31.12** |

**同じ 1 行を同じ 2,640 位置に当てて、decode は 5.0 ms、T 行経路は 26.0 ms。
5.2 倍である。**しかも幅 2 の増分 (31.1) は幅 1 (26.0) より 5.1 ms しか大きくない —
**行あたりの限界費用は decode とほぼ同じ (5.1 対 5.0)** で、
**固定の無駄が 21 ms 乗っている**という形をしている。

原因は `PrefillAttention.encodeCausal` の投げ方である
(`Sources/Tsugumi/Kernels/Attention/PrefillAttention.swift:155`)。
head_dim 512 の specialisation は 1 simdgroup に 2 クエリ、
1 スレッドグループ 256 スレッド = 8 simdgroup なので **16 クエリ / スレッドグループ**、
グリッドは `ceil(T/16) × numQHeads`。**T=1 でも T=2 でも 16 スレッドグループ**が
KV 全体を逐次に歩く。プロンプト (T=512) では正しい形だが、検証パスの幅では
GPU が空く。[36 §4-3](36-MTP-DECODE.md) の密射影 (`t >= 8` の外側で
scalar QMM に落ちる) と**同じ種類の崖**が、attention にもう 1 つあった。

### 2-3. 残りの `io` は取得の話で、経路の話ではない

`io` は幅 1 で +3.6 ms、幅 2 で +15.0 ms 増える。幅 2 の増分は
**2 行の top-8 の和集合 (≤ 16 エキスパート) を 1 パスで取る**ぶんで、
[36 §5-2](36-MTP-DECODE.md) の相乗りの費用側そのものである。t2 では
34.53 ms/パス ÷ a=1.778 = **19.42 ms/トークン**で、素の decode の 19.48 と
**ほぼ同じ** — このタスクでは相乗りは収支ゼロだった。ここは本書では触らない。

---

## 3. 直したもの — split-KV の rows カーネルに差し替えた

**新しいカーネルは 1 本も書いていない。**`Attention.encodeRows`
(`Sources/Tsugumi/Kernels/Attention/Attention.swift:290`) が既にある —
Gemma の投機ブロック用に書かれた split-KV で、KV の範囲を最大 16 チャンクに割って
`numQHeads × numChunks` のスレッドグループに配り、decode と同じ combine で
log-sum-exp を畳む (`docs/mtp/24-M5.5-RESULTS.md` §7-1)。**decode が使っている
分割の形**であり、検証パスが欲しかったのはまさにこれだった。

- `QwenForwardRunner.chunkRowsAttention` を立てた層だけ差し替わる。
  **プロンプトのプレフィルは 1 バイトも動かない** — T=512 では query-blocked が
  正しい。`--qwen-prefill` の 4 本と負の対照 5 本は差し替え後も通る (§6)。
- 対照の腕は `TF_QWEN_MTP_ROWS_ATTN=0`。[36](36-MTP-DECODE.md) が測ったのは
  この腕である。
- `fault == .uncompactedQuery` (負の対照) は query-blocked のまま。rows 経路は
  詰めたクエリの stride を前提にするので、対照を別の経路に流したら対照でなくなる。

## 4. 幅 2 を 1 発で流さなかった理由 — 投機の中立性

最初の実装は幅 2 を **1 発**で投げた (`rows: T`)。速い。**しかし
force-reject 対照 ([36 §3-2](36-MTP-DECODE.md)) との完全一致が 96/96 から
95/96 に落ちた。**

`Attention.rowsGeometry` は KV の分割を `[kvStart, startPosition + rows)` で切る。
つまり**チャンク境界がブロックの開始位置に依存する**。あるトークンが
「パスの行 1」として出るときと「次のパスの行 0」として出るときで
`startPosition` が 1 違うので、**分割が変わり、log-sum-exp の畳む順が変わり、
接戦の argmax がひっくり返る**。投機が受理したかどうかが出力に漏れる — これは
[36 §3-2](36-MTP-DECODE.md) が証明した性質そのものの喪失である。

**直し方は行ごとに 1 発ずつ投げること** (`rows: 1`、`startPosition: start + row`)。
分割がその行の絶対位置だけの関数になるので、行 0 で出ようが行 1 で出ようが
同じビットになる。

| 腕 | force-reject 対照との一致 |
| --- | ---: |
| query-blocked (36 の腕) | **96 / 96** |
| rows、幅 2 を 1 発 | 95 / 96 |
| rows、**行ごとに 1 発** | **96 / 96** |

代償は KV をもう 1 度歩くことだが、**実測で 0.4 ms/パス** (t4 幅 2 の
`preRouter` GPU が 26.77 → 27.14)。30 ms の取り分に対して 1.3% で、
**中立性を売る理由にならない。**

なお **[36 §3-3](36-MTP-DECODE.md) の「素の decode とは答えが変わる」は変わらない**
(むしろ差し替えで別の並びになる)。ここで守ったのは「投機の受理・棄却が出力に
影響しない」という内側の性質で、対照が[設計バグを捕まえた](36-MTP-DECODE.md#3-1-対照が設計の誤りを捕まえた)
ときに効いたのはそちらである。

---

## 5. 実タスクの A/B (**実測(手元)**)

`bench/qwen35/mtp_ab.sh` に 3 本目の腕を足した。192 トークン、temp 0、
**腕は回ごとに順を回す**、クールダウン 10 秒、2 反復の中央値。
運用点 (32 スロット、mmap、pipeline on)。

- `base` — 素の decode
- `mtpqb` — `--qwen-mtp`、検証パスは **query-blocked** (`TF_QWEN_MTP_ROWS_ATTN=0`)。
  [36 §5](36-MTP-DECODE.md) が測った腕
- `mtp` — `--qwen-mtp`、検証パスは **rows** (既定)

### 5-1. 表

| タスク | `base` | `mtpqb` (36 の腕) | **`mtp` (本書)** | ×(mtp) | ×(qb) |
| --- | ---: | ---: | ---: | ---: | ---: |
| a1 エージェントのコード修正 | 15.116 | 17.774 | **18.241** | **×1.207** | ×1.176 |
| a2 エージェントのツール JSON | 14.971 | 17.366 | **17.399** | **×1.162** | ×1.160 |
| t2 コード生成 | 21.511 | 19.730 | 19.925 | ×0.926 | ×0.917 |
| t3 英語の散文 | 22.201 | 20.593 | 21.361 | ×0.962 | ×0.928 |
| **t4 2,698 tok の要約** | 16.340 | 13.113 | **18.142** | **×1.110** | ×0.802 |

単位は tok/s (decode のみ)。反復間の振れは小さい (t4 は ×1.111 / ×1.110)。
`mtpqb` の並びは [36 §5-1](36-MTP-DECODE.md) の符号も順序も再現している
(絶対値は別の日の機械なので `base` ごと上にずれている)。

**長文脈の負けが消えて、5 本中 3 本が勝ち側になった。**t4 は ×0.802 → **×1.110**、
同じ日の同じ熱環境で **+38%**。

### 5-2. 取り分の出どころは 1 つだけ — 費用は長文脈でしか動いていない

腕ごとの 1 パスの費用と受理:

| タスク | `mtpqb` verify | `mtp` verify | 差 | `mtpqb` P1 | `mtp` P1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| a1 | 98.53 ms | 98.75 ms | +0.2 | 87.3% | 89.1% |
| a2 | 102.46 | 99.20 | −3.3 | 88.6% | 83.3% |
| t2 | 86.20 | 84.80 | −1.4 | 81.1% | **81.1%** |
| t3 | 74.04 | 75.34 | +1.3 | 64.7% | 74.5% |
| **t4** | **126.79** | **91.11** | **−35.7** | 74.5% | 76.1% |

**費用が動いたのは t4 だけ** (−28%)。t2 は受理率も a も**完全に同じ**まま
1 パスが 1.4 ms 安くなった — 文脈 60 では §2-2 の予測どおり取り分が小さい。

**a1 と t3 の改善は受理率の差である。**カーネルを替えるとトークン列が変わり
(§4 の注記)、変わった列は予測しやすさも変わる。t3 は 64.7% → 74.5%、
a2 は逆に 88.6% → 83.3% に落ちている。**これは機序ではなく中身のくじ引きで、
n=2 のセルにそれ以上の解釈を載せない。**言えるのは
「費用の改善は長文脈に効き、短文脈では ~1.5 ms」までである。

---

## 6. 既存経路に触れていないことの検算

### 6-1. Phase 4 の検査 (差し替え後)

```
PASS  41 tokens, every one equal to the float32 reference
PASS  chunk 8 (3 chunks) — the same 41 tokens
PASS  routed experts on the per-pair path, chunk 512 (1 chunks) — the same 41 tokens
PASS  routed experts on the per-pair path, chunk 8 (3 chunks) — the same 41 tokens
  negative controls — each must disagree within 16 tokens:  5/5 PASS
```

チャンク 8 (T=8、`Attention.maxRows` と同じ幅) も通る — `chunkRowsAttention` は
プロンプトでは立たないので、幅が届いていても差し替わらない。

### 6-2. 投機の中立性 (§4)

`TF_QWEN_MTP_FORCE_REJECT=1` と本番の腕が **96/96 一致**。t2 コード、96 トークン。

### 6-3. `swift test`

1,350 テスト / 206 スイート。**失敗 24 件はすべて `remote HTTP 404: https://hf.test/…`**
で、[36 §6-2](36-MTP-DECODE.md) と同じ vision / draft の install fixture。
本書の変更とは無関係。

---

## 7. 残る未確認と、次の一手

> **後日 (同日) の追記 — #1 と #2 は [39](39-RESIDENCY-COMMIT.md) で片付いた。**
> #1 の `io` の正体は `MTLResidencySet.commit()` で、**背景の直列キューに
> 投げるだけ**で検証パスの `io` は 36.5 → 8.3 ms/パスになった (素の decode も
> MTP も 4 腕すべて勝ち)。#2 の `draft` 6.0 ms は**89% が GPU** で、
> コマンドバッファを融合しても上限 0.64 ms/パス — 削れる無駄ではなかった。
> #3 の t2 / t3 は 39 でも t2 が ×1.011 止まりで、まだ残っている。

1. **`io` (エキスパート取得) が最大の項になった。**t2 の素の decode で
   19.48 ms/tok = 壁時計の 36%、その中身は residency set の commit
   (`commit=18.28ms/tok`)。MTP に固有の話ではなく Phase 6 の続きだが、
   **検証パスの予算でも `preRouter` GPU を追い越した** (§2-1)。次に測るならここ。
2. **`draft` が 6.0 ms/パス**、GPU 時間は計上されていない (ドラフタは自前の
   コマンドバッファで回る)。1 トークンあたり 3.4 ms で、勝ち幅 15〜20% と
   同じ桁である。ここは計器がまだ無い。
3. **t2 / t3 はまだ負ける** (×0.93 / ×0.96)。素の decode が一番速い 2 本で、
   [36 §5-2](36-MTP-DECODE.md) の「取得の相乗りは、素の decode が遅いタスクほど
   効く」と整合する。1 と 2 が動けばここも変わる。
4. **[37 §4-A](37-MTP-POSTMORTEM-PLAN.md) の残り 2 つの弾は取り下げる** —
   routing 読み戻しの幅 ≤ 2 特化は上限 0.25 ms (§2-1)、
   「attention/GDN を decode の 1 行カーネルで 2 回」は**密射影まで含めると損**
   (幅 2 の `preRouter` GPU は 26.6 ms/パス = 14.9 ms/トークンで、decode の
   16.7 ms/トークンより既に安い。1 行カーネルを 2 回回せば重みを 2 度読む)。
   **残ったのは attention だけで、それが本書である。**
5. **文法・ツール呼び出しとの併用** ([36 §7](36-MTP-DECODE.md) #2、
   [37 §4-B](37-MTP-POSTMORTEM-PLAN.md)) は手つかず。エージェント用途では
   これが一番効く制約のまま。
6. **自動ゲート** ([37 §4-C](37-MTP-POSTMORTEM-PLAN.md)) の前提が 1 つ変わった:
   「プロンプト > 1,000 tok なら off」は**もう要らない** — 長文脈こそ勝つ側に
   回った。既定を変えるかどうかはユーザー判断。
7. **サーバー経路には出していない** (`--qwen-mtp` は CLI だけ)。

---

## 8. コードと文書の根拠 (2026-08-22 に確認した現物)

| 事実 | 場所 |
| --- | --- |
| 段ごとの計器 | `Sources/Tsugumi/Runtime/Inference/QwenStageProfile.swift` + `QwenForwardRunner.stage(_:_:)` (`TF_QWEN_STAGE_PROFILE=1`) |
| 段ごとの実測 (t2 / t4 × base / 幅 1 / 幅 2 × 2 カーネル) | `bench/qwen35/mtp_stage_profile.sh` → `bench/qwen35/mtp38/*.footer` (**実測(手元)**) |
| query-blocked の投げ方 (16 クエリ/スレッドグループ) | `Sources/Tsugumi/Kernels/Attention/PrefillAttention.swift` `encodeCausal` の `.qBlock` 分岐 |
| 差し替え先の split-KV | `Sources/Tsugumi/Kernels/Attention/Attention.swift` `encodeRows` / `rowsGeometry` (`docs/mtp/24-M5.5-RESULTS.md` §7-1) |
| 差し替えの本体と行ごとの投げ方 | `Sources/Tsugumi/Runtime/Inference/QwenPrefill.swift` `encodePrefillAttention` |
| 腕の切り替え | `QwenForwardRunner.chunkRowsAttention` / `TF_QWEN_MTP_ROWS_ATTN` |
| 3 腕の A/B | `bench/qwen35/mtp_ab.sh` (`ARMS="base mtp mtpqb"`) → `bench/qwen35/mtp38-ab/*.footer`、集計は `bench/qwen35/mtp38_report.py` (**実測(手元)**) |
| 中立性の対照 | `TF_QWEN_MTP_FORCE_REJECT=1`、`bench/qwen35/mtp38/ctrl.*.json` |
| 37 が名指しした内訳 | [37 §4-A](37-MTP-POSTMORTEM-PLAN.md) / [36 §4-4](36-MTP-DECODE.md) (**未確認**と書かれていたもの) |
