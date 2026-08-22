# 37. MTP の検討 6 本を負例として解剖し、次の改善を決める (検討 + NVMAI 照合、2026-08-22)

[30](30-MTP-HEAD-GRAFT.md) から [35](35-PREFILL-CHUNK-WIDTH.md) までの 6 本は
1 行も推論経路を書かずに MTP の取り分を見積もり、[36](36-MTP-DECODE.md) の
実装 1 日でその集計は崩れた。本書は 3 つをやる:

1. **負例の解剖** — どの数字がなぜ腐ったのか。個々の文書の誠実さは保たれて
   いたのに、なぜ積み上げが誤ったのかを、再発を防げる形の規則に落とす (§1)
2. **実験で初めて見えた機序** — 事前検討のどのモデルにも項が無かったもの (§2)
3. **NVMAI 再照合と改善案** — [32](32-NVMAI-ADOPT.md) の後にもう一度
   `~/LLM/NVMAI` を読み、採用済み・未採用・不採用を仕分けた上で、
   [36 §7](36-MTP-DECODE.md) の残件を潰す順を決める (§3〜§4)

**結論を 1 行で:** 移送してよいのは品質の量 (受理率・routing) だけで、費用の量は
実行経路の性質だから移送できない。改善の本丸は**幅 2 特化のハイブリッド検証パス**
(A) で、文法併用 (B) が用途を開き、自動ゲート (C) が既定 on に近づける。

> **後日 (同日) の訂正 — [38](38-MTP-VERIFY-PATH.md) を先に読むこと。**
> §4-A の内訳 (ホスト +11 ms = route grouping、GPU +2.7 ms) は
> **段ごとの計器で外れた**: grouping は 0.25 ms/パス、1.21 倍のほぼ全部は
> `preRouter` の GPU で、その中身は**文脈に比例する attention** だった。
> A の 3 つの弾のうち生きていたのは attention だけで、それは
> [38 §3](38-MTP-VERIFY-PATH.md) で片付いた (t4 ×0.802 → ×1.110)。
> **本書は「実測せずに書いた見積もりは 7 本目でもまだ外れる」という負例を
> 1 本足したことになる** — §1-4 の規則 3 が自分自身に効いた形である。

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | 何が数字を腐らせたか | **費用の量の移送** (§1)。[33 §3-7](33-MTP-ACCEPTANCE.md) は比をプレフィル経路で測って decode の予算に掛け (基底が 1.52 倍違う)、[32 §1-4](32-NVMAI-ADOPT.md) は NVMAI の和集合費用モデルを写した (通貨が違い、符号まで逆)。一方**品質の量は移送に成功している** — off-policy 受理率も履歴 ablation も実機を当てた |
| 2 | なぜ 6 本続いたか | **proxy が数字をくれるから** (§1-3)。各文書は注意書き付きの正直な数字を出せてしまい、注意書きは連鎖の中で失われた。基底を暴いた幅 1 対照は **MTP のヘッド無しでも書けた**実験で、順序が逆なら 33 §3-7 は書かれなかった |
| 3 | 実験でしか見えなかった機序 | 3 つ (§2)。**勝敗は受理率でなく取得の相乗り**が決める (MTP はタスク間の tok/s を平準化する機構)、**費用はカーネル選択の崖 1 つ**に居た (`t >= 8` の外側)、**対照実験が設計バグと数値差を仕分けた** |
| 4 | NVMAI から新しく採るもの | **draft へのプロンプト履歴のチャンク相乗り** (`prefillChunkedWithMTP`、§3-2)。[36 §2-2](36-MTP-DECODE.md) が「釣り合わない」と閉じたのは**別経路として**の話で、NVMAI は相乗りとして実装済み。しかも [36 §2-3](36-MTP-DECODE.md) の洞察でフルブロックは要らず k/v だけでよい — NVMAI より安く「full」腕 (+2.35pt) が取れる。ほかにサーバー選択ロジック・prompt cache 排他・メモリ予算の器 |
| 5 | NVMAI から採らないもの | フルブロックの draft prefill (k/v で足りる)、150k 語彙の CPU argmax (融合 multi ヘッドが上)、**「MTP は損」という結論そのもの** (通貨依存。§1-2) |
| 6 | 改善の本丸 | **A: 幅 2 特化のハイブリッド検証パス** (§4-A)。attention/GDN は decode の 1 行カーネルを 2 回逐次、MoE だけ 2 行の和集合 grouping で相乗りを残す。[36 §7](36-MTP-DECODE.md) の #3 (残る 1.21 倍) と #4 (長文脈) と #1 (出力差) を 1 つの設計で潰す |
| 7 | 用途を開くのは | **B: 文法・ツール呼び出し併用** (§4-B)。GEN-7 再採点の 2 行化。Gemma 側で 2026-08-22 に緑になった文法込み投機の作法を写す。エージェント運用点 (a2 + `--tools`) はここが繋がって初めて実利になる |
| 8 | 運用を変えるのは | **C: 自動ゲートと途中離脱** (§4-C)。force-reject 対照が本番とトークン列完全一致 ([36 §3-2](36-MTP-DECODE.md)) なので、**生成途中で MTP を切っても出力は変わらないことが保証済み**。実効倍率を窓で監視して負けたら落ちる |

---

## 1. 負例の解剖 — 移送の成功と失敗

30〜35 が積んだ数字はどれも**移送された数字**である: 測れない量の代わりに近くの
量を測り、係数や比だけを持って来る。移送には成功例と失敗例があり、境界線は
きれいに引ける。

### 1-1. 移送に成功したもの — モデルとタスクの性質

| 量 | 出どころ | 実機 ([36](36-MTP-DECODE.md)) | 判定 |
| --- | --- | --- | --- |
| off-policy 教師強制の受理率 | [33 §2-1](33-MTP-ACCEPTANCE.md) (CPU 参照器、bf16) | 予測 85.3 / 64.9 / 77.0 → 実機 81.1 / 64.7 / 74.5 (t2/t3/t4)。並びも一致 | **移った** |
| プロンプト履歴を捨てる代償 | [36 §2-2](36-MTP-DECODE.md) の ablation (npz 再利用、モデル再実行なし) | t3 で 64.92% と予測、実機 64.7% | **移った** |
| 幅 2 / 幅 1 の費用**比** | [33 §3-7](33-MTP-ACCEPTANCE.md) (プレフィル経路) | 1.27〜1.40 と予測、実機 1.307 ([36 §4-1](36-MTP-DECODE.md)) | **比は移った** |

受理率・routing・hidden の統計は、bf16 の CPU 参照器で測ろうが 4bit の実機で
測ろうが移る。**モデルとタスクの性質だから**である。

### 1-2. 移送に失敗したもの — 実行経路の性質

**失敗 1: 同じ機械の中の経路またぎ ([33 §3-7](33-MTP-ACCEPTANCE.md))。**
幅 2/幅 1 の比をプレフィル経路で測り、decode の予算に掛けた。比は上の表の
とおり正しかった。しかし掛ける先の「幅 1」がプレフィル経路では素の decode の
**1.52 倍** ([36 §4-1](36-MTP-DECODE.md)) で、+15〜29% の予測は −16%〜+20% の
現実になった。**同じ機械の中ですら、経路が違えば費用の基底は移らない。**

**失敗 2: 機械またぎで符号まで逆 ([32 §1-4](32-NVMAI-ADOPT.md))。**
NVMAI の `verifyGreedyPair` の実測 — 「検証費用はエキスパートの**和集合**で
決まり (幅 2 で 1.585x)、射影を速くしても救えない。損益分岐は p > 0.585」 —
を写した。32 自身が「機構は写せるが数字は写せない」と注意書きしながら、
**符号が写らない**ことは見抜けなかった:

- NVMAI の通貨は SSD の `pread` である。和集合 1.585x はそのまま**読みの費用**。
- 本線の通貨はホスト写像 ([27 §9](27-PHASE6-THROUGHPUT.md)) である。同じ
  「同一エキスパートに routed した行は 1 回の重み読みに相乗りする」構造が、
  **1 回の取得を 1.8 トークンで割る利得**として現れた ([36 §5-2](36-MTP-DECODE.md))。
- p > 0.585 の閾値モデルは本線の勝敗の符号を 1 つも予測しない。t4 は
  p = 74.5% で負け (×0.843)、a1 は 87.3% で勝つ (×1.146)。どちらも閾値の上。
- 「射影を速くしても救えない」は本線では端的に誤り。カーネル 1 本の差し替えが
  GPU を 60.9 → 27.4 ms/tok にした ([36 §4-3](36-MTP-DECODE.md))。

### 1-3. なぜ 6 本続いたか

proxy が数字をくれるからである。npz、NVMAI のソース、プレフィルのベンチ —
どれも推論経路を書かずに、それぞれ**注意書き付きの正直な数字**を出せた。
腐ったのは個々の数字ではなく積で、`正しい比 × 間違った基底` を誰も検算して
いない。注意書きは連鎖の中で失われる。

皮肉なのは、基底を暴いた幅 1 対照 (`TF_QWEN_MTP_NO_DRAFT=1` — 1 行を
T 行経路に流すだけ) が **MTP のヘッドが 1 行も無くても書けた**実験であること。
これを最初にやっていれば 33 §3-7 は書かれなかった。

### 1-4. 規則に落とす

1. **費用の量は proxy から積まない。品質の量は積んでよい。**受理率・routing・
   hidden 統計はモデルの性質で移送できる。ms・tok/s・費用比の基底は実行経路の
   性質で移送できない — 同じ機械の中の経路またぎでも。
2. **比を掛ける前に、掛ける先の絶対値を実測する。**([`ratio-vs-base-mismatch`]
   として記憶済み。) 比の測定より基底の測定が先。
3. **「実装ゼロで書ける対照」を検討フェーズの必須項にする。**幅 1 対照のように、
   本命機構が無くても土俵の費用だけは測れることが多い。検討文書が費用を
   見積もるとき、その基底を実測せずに 2 本目の文書へ進まない。

なお検討 6 本が無駄だったわけではない。二重バッファの設計
([33 §3-6](33-MTP-ACCEPTANCE.md))、pre-final-norm の決着 ([32 §1-1](32-NVMAI-ADOPT.md))、
深さ 1・幅 2 の決め打ち、履歴 ablation — **設計と品質の検討は全部生きて実装に
入った**。腐ったのは費用の見積もりだけで、それはまさに実行経路の詳細
(カーネル選択、dispatch、同期) に依存する量だった。

---

## 2. 実験で初めて見えた機序

事前検討のどのモデル (32 の I/O モデル、33 の比のモデル) にも項が無かったもの。

### 2-1. 勝敗を決めるのは受理率ではなく、取得の相乗り

[36 §5-2](36-MTP-DECODE.md)。MTP 腕の tok/s は t4 を除き ×1.18 の幅しか
動かないのに、素の decode は ×1.47 動く。MTP は「速いタスクを速くする」機構では
なく**「遅いタスクを平準化する」機構**である — 1 パスが 1 回のエキスパート取得を
1.8 トークンで割るので、エキスパート局所性の悪いタスク (= 素の decode が遅い
タスク) ほど勝つ。エージェント形で勝ち、散文で負ける符号の並びはこれ 1 つで
説明される。クロスタスクの tok/s の**分散構造**を見て初めて出る事実で、
1 タスクの測定を何回重ねても出ない。

### 2-2. 費用はカーネル選択の崖 1 つに居た

[36 §4-3](36-MTP-DECODE.md)。`QwenPrefillInt8QMM.usesTiledPath` の `t >= 8` の
外側に検証パス (1〜2 行) が必ず落ち、coalesce しない scalar QMM
(`qwen_int8_qmm_f16_block`) を踏んでいた。FLOPs にも和集合にも比にも、
この項は無い。**費用の見積もりが実装詳細に負ける典型**で、§1-4 の規則 1 の
実例そのもの。

### 2-3. 対照実験が設計バグと数値差を仕分けた

[36 §3](36-MTP-DECODE.md)。force-reject 対照 (`TF_QWEN_MTP_FORCE_REJECT=1`) が
「パス前スナップショットへの復元 = 復元しすぎ」を degenerate ループとして
捕まえ (§3-1)、修正後は対照と本番の**トークン列完全一致**で投機の中立性を
証明した (§3-2)。その上で残った素の decode との出力差を「投機のバグではなく
T 行カーネルと decode カーネルの加算順」と診断できた (§3-3)。対照が無ければ
誤診していた。**機構を入れるときは、機構を無効化する腕を同時に作る** —
これも検討では出ず、実装して初めて価値が確定した作法である。

---

## 3. NVMAI 再照合 — 採用済み・未採用・不採用

[32](32-NVMAI-ADOPT.md) の後、[36](36-MTP-DECODE.md) を実装した目でもう一度
`~/LLM/NVMAI` を読んだ (2026-08-22、`StreamingMTP.swift` 364 行 /
`RealForwardRunner.swift` の MTP 系 / `ServerInference.swift`)。

### 3-1. 報告済み・採用済み (再掲のみ)

| 要素 | 記録 | 状態 |
| --- | --- | --- |
| pre-final-norm hidden | [32 §1-1](32-NVMAI-ADOPT.md) | 採用 |
| checkpoint/restore | [32 §1-2](32-NVMAI-ADOPT.md) → [33 §3-6](33-MTP-ACCEPTANCE.md) | **ポインタ交換に改良して採用** (コピー 0 回、NVMAI は blit) |
| interleaved A/B の作法 | [32 §7](32-NVMAI-ADOPT.md) | bench に採用 |
| `MTPStatistics` (受理率 / emit/パス) | [32 §1-6](32-NVMAI-ADOPT.md) | 相当物あり (P1 / a) |
| p > 0.585 の判定線 | [32 §1-4](32-NVMAI-ADOPT.md) | 33 のゲートに使用。**符号予測としては外れた** (§1-2) |
| sidecar 8 スロット streaming の実測 | [32 §1-5](32-NVMAI-ADOPT.md) | fallback として記録のみ |

### 3-2. 新しく見つけた未採用 — draft へのプロンプト履歴のチャンク相乗り

[36 §2-2](36-MTP-DECODE.md) は「full と gen の差 2.35 ポイントに 2 本目の
prefill 経路は釣り合わない」と閉じた。**その前提が NVMAI の実装で崩れる。**

NVMAI の `prepare` は `prefillChunkedWithMTP`
(`sources/NVMAI/Runtime/Inference/RealForwardRunner.swift:1346`) を呼ぶ:
本体プレフィルの**チャンクごと**に、そのチャンクの hidden を読み戻して sidecar の
`advanceMTP` (`predictNext: false`) に流し込み、チャンク境界の 1 行は `carry` で
持ち回る。**別経路ではなく相乗り**であり、draft の KV はプロンプト全域を持って
生成に入る。

しかも NVMAI はフルブロック (attention + MoE + 1 層 MoE のエキスパート I/O) を
プロンプト位置にも回しているが、[36 §2-3](36-MTP-DECODE.md) の洞察のとおり
**出力を誰も読まない位置は (k, v) しか要らない**。つまり本線の版は
fc (BF16 GEMV) → `input_layernorm` → k/v 射影だけでよく、attention も MoE も
508 MB のヘッドも走らない。チャンク内なら T ≥ 8 でタイル QMM の内側である。
**NVMAI より安い形で、ablation の「full」腕 (平均 P1 +2.35pt、t3 は +4.2pt) が
取れる** (机上。費用は実装後に実測)。

### 3-3. 新しく見つけた未採用 — 周辺の器 3 つ

| 要素 | 場所 | 中身 |
| --- | --- | --- |
| サーバー選択ロジック | `sources/NVMAIServer/Core/ServerInference.swift:1085` | pure greedy **かつ** `プロンプト + maxNewTokens ≤ draft 文脈` のときだけ MTP、外れたら素の runner に静かに落ちる。YaRN は構築時に明示拒否 (`yaRNUnsupported`) |
| prompt cache との排他 | 同 `:451` | MTP 有効時は prompt cache を強制 off。「MTP は第 2 の KV ストリームを持ち、両状態を snapshot に含めるまで排他」の注記付き。[34](34-PROMPT-CACHE-ESTIMATE.md) を作るときの先回りの地雷除去 |
| `StreamingMTPMemoryPlan` | `StreamingMTP.swift` 冒頭 | 部品別の明示予算 (常駐テンソル / expert cache / draft KV / rollback / scratch)、超過は**構築時エラー**、環境変数は fail-closed |

### 3-4. 不採用

- **フルブロックの draft prefill** — §3-2 のとおり k/v で足りる。
- **150k 語彙の CPU argmax** (`StreamingMTPDecoder.argmax`) — 本線の融合
  multi ヘッド (`qwen_lm_head_greedy_int8_rows_chunk_raw_multi`) が上。
- **「MTP は損 (~0.85x)」という結論そのもの** — 通貨依存 (§1-2)。NVMAI の
  doc コメントは NVMAI の機械では正しく、本線では符号が逆。

---

## 4. 改善案 — 効き順

### A. 幅 2 特化のハイブリッド検証パス ([36 §7](36-MTP-DECODE.md) #1 #3 #4 を 1 設計で)

残る 1.21 倍の内訳 (ホスト +11 ms = `readPrefillRoutes`
(`Sources/TurboFieldfare/Runtime/Inference/QwenPrefill.swift:800`) の層ごと汎用
grouping、GPU +2.7 ms = prefill attention + block router) と、t4 の長文脈負け
(2,900 位置 × 2 行の T 行 attention) は、**全部「幅 2 なのに汎用 T 行機械を
使っている」ことに由来する**。attention/GDN と MoE は分離できる:

- **attention / delta rule は decode の 1 行カーネルを 2 回、逐次**。行 1 は
  行 0 の k/v 書き込み後に走る。delta rule は [33 §3-6](33-MTP-ACCEPTANCE.md) で
  既に T=1 × 2 分割済み (代償 0.28%)。長文脈でも decode と同じ費用になり、
  t4 の負け筋が消える。
- **MoE だけ 2 行の和集合 grouping** で取得の相乗り (§2-1 の利得の源) を残す。
- **routing 読み戻しは幅 ≤ 2 特化**: `makeTokenExpertPairs` の汎用 grouping を
  やめ、topK 8 × 2 行の和集合 (≤ 16) を decode 式のスロット付けで直接組む。
- 副産物: attention が decode カーネルになるので **§3-3 の出力差が縮む**
  (密射影は既に T=1 でビット一致)。残るは router と MoE 縮約の加算順だけ。

机上の算術 (実測で置き換えること): 幅 1 のパスが 72.0 → 59.4 ms 級に落ち、
幅 2 の増分が今と同じ 29 ms なら 1 パス 88 ms、a = 1.81 で t2 は 20.6 tok/s
(×1.22)。[36 §4-4](36-MTP-DECODE.md) の言うとおり全タスクが勝ち側に回る計算。

### B. 文法・ツール呼び出しとの併用 ([36 §7](36-MTP-DECODE.md) #2)

エージェント用途で一番効く制約。`SpeculativeError.constrained`
(`QwenSpeculativeDecode.swift:164`) の排他を外すには、GEN-7 の融合ヘッド再採点を
行 index/stride 付きにして 2 行分の `normalizedHidden` を埋め、draft の提案と
検証行の両方に文法マスクを当てる。NVMAI は「制約せずパースする」設計
(`StreamingToolCallParser`) なので参照にならない。代わりに **Gemma 側で
2026-08-22 に緑になった文法込み投機の作法**を写す。A/B の a2 (ツール JSON、
×1.199) はまさにこの運用点で、ここが繋がって初めてエージェント経路の実利になる。

### C. 自動ゲートと途中離脱

force-reject 対照が本番とトークン列完全一致 ([36 §3-2](36-MTP-DECODE.md)) —
つまり**生成途中で MTP を切っても出力は変わらないことが保証済み**である。

- ターン開始時: プロンプト > 1,000 tok なら off (A が入るまでの暫定)、
  制約が要るターンは off (B が入るまで)。
- 生成中: `MTPStatistics` 相当の窓で実効倍率を監視し、負けていれば素の decode に
  落ちる。途中で on に戻すのは draft KV の追いつきが要るので後回し
  ([36 §2-3](36-MTP-DECODE.md) の k/v-only 追いつきで書ける)。
- サーバー結線 ([36 §7](36-MTP-DECODE.md) #6): NVMAI の選択ロジック +
  prompt cache 排他 (§3-3) をそのまま。

これで「既定 off の使われない機能」から「既定 on でも損しない機能」に変わる。

### D. draft のプロンプト履歴 — チャンク相乗り、k/v-only (§3-2)

費用はプレフィルの 1% 未満と読む (机上)。取り分は実測済みの +2.35pt
(t3 は +4.2pt、[36 §2-2](36-MTP-DECODE.md))。§2-1 の機序からして単独の効果は
小さいが、A で t3 が損益分岐線上に来たときの後押しとして安い。優先度は A/B の後。

### E. sidecar の常駐 453 MB を任意化

NVMAI は 8 スロット streaming で差なしを実測済み (interleaved、6.202 vs
6.008 tok/s、[32 §1-5](32-NVMAI-ADOPT.md))。本線は mmap なので「residency set の
commit をやめてページキャッシュに任せる」だけの形もある。18 GB 機で約 400 MB を
返せる。RAM が苦しくなったときの引き出しとして、フラグ 1 本ぶんの実装。

### F. 将来の prompt cache ([34](34-PROMPT-CACHE-ESTIMATE.md)) に排他を先に敷く

snapshot に draft KV を含める拡張は後回しでよい。まず NVMAI と同じ
「MTP on なら cache off」を結線しておけば地雷にならない (§3-3)。

---

## 5. 測っていないもの

- A の机上の算術 (§4-A) は幅 2 の増分 29 ms が据え置きという仮定を含む。
  和集合の grouping を特化した後の増分は実測するまで分からない。
- D の「プレフィルの 1% 未満」は fc + k/v 射影の形からの読みで、未実測。
- B の文法マスク 2 行化の費用 (再採点 2 回ぶん) は未実測。
- C の監視窓の幅・閾値は未設計。

## 6. コードと文書の根拠 (2026-08-22 に確認した現物)

| 事実 | 場所 |
| --- | --- |
| draft へのプロンプト履歴のチャンク相乗り | `~/LLM/NVMAI/sources/NVMAI/Runtime/Inference/RealForwardRunner.swift:1346` `prefillChunkedWithMTP` (carry で境界 1 行を持ち回る) |
| NVMAI の advanceMTP がフルブロックを回すこと | 同 `:817` (`predictNext: false` はヘッドを飛ばすだけで `executePrefillChunk` は走る) |
| 和集合費用モデルと p > 0.585 | 同 `:713` 付近の `verifyGreedyPair` doc コメント (**実測(NVMAI)**) |
| サーバー選択ロジック / prompt cache 排他 | `~/LLM/NVMAI/sources/NVMAIServer/Core/ServerInference.swift:1085` / `:451` |
| メモリ予算の器 | `~/LLM/NVMAI/sources/NVMAI/Runtime/Generation/StreamingMTP.swift` `StreamingMTPMemoryPlan` |
| 本線の汎用 grouping (ホスト +11 ms の主犯とみるもの) | `Sources/TurboFieldfare/Runtime/Inference/QwenPrefill.swift:800` `readPrefillRoutes` |
| 文法との排他 | `Sources/TurboFieldfare/Runtime/Inference/QwenSpeculativeDecode.swift:164` |
| 実機の A/B・費用分解・対照 | [36](36-MTP-DECODE.md) §3〜§5 (**実測(手元)**) |
| 移送の成否の判定材料 | [32 §1-4](32-NVMAI-ADOPT.md) / [33 §2-1, §3-7](33-MTP-ACCEPTANCE.md) / [36 §2-2, §4-1, §5](36-MTP-DECODE.md) |
