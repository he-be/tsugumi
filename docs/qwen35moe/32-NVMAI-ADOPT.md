# 32. NVMAI からの移植 — 何を丸ごと写すか

**検討 + 実測(NVMAI)。**2026-08-22 起草。`~/LLM/NVMAI` は**同じ
Ornith 1.5 35B-A3B** を動かす成熟した Swift/Metal ランタイムで
(Apache-2.0、`sources/` 以下)、本計画が「できない」と結論した 2 つ —
**投機デコードの巻き戻し** ([26](26-PHASE8-SERVER.md)) と
**prompt cache** ([26](26-PHASE8-SERVER.md) で `cache_n` 常に 0) — を、
どちらも**「巻き戻さず、確定点を checkpoint して restore する」**形で解いている。
本文書はその実装を読んで (コードと設計文書、コミット履歴)、写す対象・写さない対象・
写す順を決めるための材料を置く。

**表記:** 本文書の **実測(NVMAI)** は「NVMAI のソースまたは設計文書に
数字と手続きが書かれており、実装がそこに在ることを確認した」もの。
**NVMAI の機械の数字であり、この M3 Pro の数字ではない。**特に NVMAI の
expert I/O は `F_NOCACHE` + bounded `pread` で、本ランタイムの mmap とは
**待ちの通貨が違う** ([28 §1](28-PREFETCH-IDEAS.md) と同じ注意)。機構は写せるが、
数字は写せない。

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | 一番大きい持ち帰り | **MTP の損益分岐が測る前に分かった** (§1-4、実測(NVMAI))。検証パスの費用は行数ではなく**エキスパートの和集合**で伸びる: 幅 2 で 12.68 experts/層 = **1.585 倍**。対する受理 57.4% の利得は 1.574 tok/pass。**相殺して、残りの固定費で約 0.85 倍の損。**損益分岐は **受理率 p > 約 0.585**。`mtp_acceptance.py` の判定線はこれ |
| 2 | §4-3 の未決着 | **決着** (§1-1)。draft に渡す本体 hidden は **pre-final-norm** (`model.norm` の**前**)。NVMAI の動く実装が明記している |
| 3 | 「再帰状態を巻き戻せない」 | **投機には別解がある** (§1-2)。KV は append-only でカーソルだけ戻し、GDN 状態は**確定行直後の値を GPU 上の影バッファに取り、棄却時に blit で書き戻す**。巻き戻しではなく checkpoint/restore |
| 4 | prompt cache | **「持たない」の根拠が半分消える** (§2)。巻き戻し不要な形 — 次の要求が前回の**厳密な延長** — だけを狙う snapshot-restore 型なら再帰状態と両立する。NVMAI は KV+GDN を 1 本の payload に落として復元している。**エージェントのツールループはまさにこの形** |
| 5 | decode の残りの待ち | **hit-fixup が [27 §9](27-PHASE6-THROUGHPUT.md) の写像待ちに重なる** (§3)。routed phase-1 を「常駐分を即 commit、miss だけ後から fixup」に割る。32 スロットのヒット率 74.9% ぶんの GPU 仕事を待ちの陰に入れられる |
| 6 | サンプラ (R3) | **写せるが、先に頭が要る** (§4)。NVMAI は mlx-lm 順 (Top-P → Top-K → 温度) の GPU サンプラを持つ。本線の障害物はサンプラ不在ではなく**融合ヘッドが logits を書かない**こと ([19](19-LM-HEAD-INT8.md)) |
| 7 | 測定作法 | **逐次スイープは禁止** (§5、実測(NVMAI))。ページキャッシュの温めで**偽の +11%** が出る。interleaved A/B で測るか、測らないか。[31](31-PREFETCH-CHEAPER.md) 以降の A/B に既定として足す |
| 8 | 写さないもの | v4.2 の event-driven Metal I/O (NVMAI 自身が資格審査で落とした)、予測先読みの実装 (NVMAI では昇格失敗、こちらは [31](31-PREFETCH-CHEAPER.md) が既に上)、thinking 切り替え (既に持っている) (§6) |
| 9 | ライセンス | Apache-2.0。コードを写すときは `THIRD_PARTY_NOTICES.md` に記載を足す |

---

## 1. MTP — Phase 7 の器 (`StreamingMTP.swift`)

出典: `sources/NVMAI/Runtime/Generation/StreamingMTP.swift` (364 行) /
`sources/NVMAI/Runtime/Inference/RealForwardRunner.swift` の
`captureSpeculativeCheckpoint` / `rollbackSpeculativeCheckpoint` /
`verifyGreedyPair` / `advanceMTP` / `rewindMTP` /
`sources/NVMAI/Runtime/KVCache/GDNStateManager.swift` の speculative 系。
NVMAI の MTP は Qwen3.6 のネイティブヘッドで、本線の差し替え済みヘッド
([30 §6](30-MTP-HEAD-GRAFT.md)、ドナーが同じ Qwen3.6) と**同じ形の sidecar**
(1 層 MoE、256 エキスパート、embedding / lm_head は本体と共有)。

### 1-1. draft に渡す hidden は pre-final-norm — §4-3 の決着

`mtp_acceptance.py` は「本体 hidden が `model.norm` の前か後かは未決着なので
両方引く」としていた ([29 §3-4](29-MTP-PREFETCH-OUTLOOK.md))。NVMAI の
`TargetPairVerification.hiddenRows` は
**"Two contiguous FP16 pre-final-norm target hidden rows"** と明記し、
`verifyGreedyPair` の実装もそうなっている: `scratch.hidden` (最終 norm 前) を
blit で退避してから、logits 用には `prefillFinalRowHead` が **norm 込み**で
別に畳む。つまり **draft への入力は `model.norm` の前、logits は後。**
`mtp_acceptance.py` の両引きは、pre 側が本命という予想で読んでよい
(ただし検算としての両引き自体は安い。捨てなくてよい)。

### 1-2. 巻き戻しの別解 — checkpoint/restore

構造は 3 つの部品に分かれ、**どれも「過去に戻る」演算を持たない**:

| 状態 | 手 | NVMAI の実装 |
| --- | --- | --- |
| full-attention KV (本体) | **append-only。カーソルだけ戻す** | `SpeculativeInferenceCheckpoint` は `position` 1 個の値型。棄却時 `kv.rewind(to: position + 1)` — 確定行は残す |
| GDN 状態 (本体 30 層) | **確定行直後の値を影バッファに取る** | 2 トークン検証バッチのカーネルが `snapshotGDNAfterFirstToken: true` で **1 トークン目を流した直後の状態**を speculative バッファに書き、棄却時は `encodeSpeculativeRestore` が blit で本物へ書き戻す |
| draft の KV (sidecar 1 層) | **カーソルだけ戻す** | `rewindMTP(to:)`。棄却された提案の 1 行は次のパスが上書きする |

本ランタイムへの写像は NVMAI より**簡単**になる:
`RecurrentStateManager` は 30 層ぶんを `stateBuffer` / `convBuffer` の
**連続 2 本**で持っているので (NVMAI は層ごとに別バッファ)、影も 2 本、
blit も 2 回で済む。

**→ blit すら要らないと分かった ([33 §3-6](33-MTP-ACCEPTANCE.md)、実測)。**
`qwen_delta_rule` は `stateIn` / `stateOut` を元から別引数で取るので、
投機行の出力先を第 2 バッファにして**受理時にポインタを入れ替える**だけでよい
(コピー 0 回、棄却時は何もしない)。カーネルへの引数追加も不要。払うのは
T=2 を T=1 の 2 回に割る代償 **0.28%** だけである。**番号の大きい 33 が正。**

### 1-3. ループの形 — 2 トークン検証バッチ

`advance(boundaryToken:)` 1 回の中身:

1. draft が `(境界トークン, 本体 hidden)` から次トークンを 1 個提案
   (`advanceMTP`, `predictNext: true`)
2. 本体が `[境界, 提案]` の **2 行を prefill 経路で 1 パス**流す
   (`verifyGreedyPair`)。40 層 1 回の走査から **2 つの argmax と
   2 つの pre-norm hidden** が出る
3. 一致なら 2 トークン emit、draft には受理トークンを追記
   (`predictNext: false` — 提案は要らない、KV を揃えるだけ)。
   不一致なら draft KV を 1 行 trim、本体を restore、1 トークン emit

計器は `MTPStatistics` (受理率と **emit/本体パス** の 2 つ) をそのまま写す。
greedy 限定は明示エラー (`StreamingMTPError.greedyOnly`)。R3 の
「受理して無視」とは整合する (どうせ greedy)。

### 1-4. NVMAI 自身の実測 — 現行の形では損

**実測(NVMAI)** (`verifyGreedyPair` の doc コメント、Qwen3.6-35B-A3B、
40 層、topK=8/256):

| 検証幅 | experts/層 (和集合) | 費用 |
| --- | ---: | ---: |
| 1 (通常 decode) | 8.00 | 1.000x |
| 2 (`verifyGreedyPair`) | 12.68 | **1.585x** |
| 13 | — | 5.18x |
| 42 | — | 11.25x |

sparse MoE では検証パスの費用が**行数ではなくエキスパートの和集合**で決まる
(同じエキスパートに routed した行は 1 回の重み読みに相乗りする —
prefill の expert-major grouping がまさにそれ)。受理率 57.4% の利得
1.574 tok/pass に対し費用 1.585x — **相殺し、パスあたりの固定費で
約 0.85 倍の損**。幅を広げても利得は 1/(1−p) = 2.35 で頭打ち、
和集合は伸び続けるので**幅 2 が最も損益分岐に近く、それでも届かない**。
レバーは検証経路の高速化ではなく**受理率のみ**: **p > 約 0.585** が下限。

本線への読み替え: 費用の通貨は違う (NVMAI は SSD 読み、こちらは
ホスト写像 [27 §9](27-PHASE6-THROUGHPUT.md)) が、**和集合で伸びる構造は
写像のページ数にも同じに効く**ので、損益分岐の形は移る。したがって:

- **`mtp_acceptance.py` (M0) の判定線を p = 0.585 に置く。**深さ 1 の
  P1 がここに届かないなら、Phase 7 の decode 高速化としての取り分は無い —
  **カーネルを 1 本も書く前に閉じられる**
- 届かなくても [29](29-MTP-PREFETCH-OUTLOOK.md) の**先読みの神託としての
  MTP** (draft の routed 先で次トークンのエキスパートを名指す) は別勘定で
  生きている。そこは受理率ではなく名指しの再現率が通貨
- NVMAI の受理 57.4% は**ネイティブヘッド**の値。差し替えヘッド
  ([30](30-MTP-HEAD-GRAFT.md)) がこれを上回る理由は無いので、
  期待値は低く持つ

### 1-5. sidecar のエキスパート常駐 — 対抗する実測が 1 つ

[03 §6](03-DESIGN.md) は **draft の 256 エキスパート全常駐 (453 MB、
draft I/O ゼロ)** とした。NVMAI は逆に **8 スロットの streaming** で流し、
interleaved A/B (各 4 標本、warmup 捨て) で **8 スロット 6.202 tok/s (sd 0.26)
vs 32 スロット 6.008 (sd 0.35) — 差なし**を測っている (実測(NVMAI)、
`StreamingMTPMemoryPlan.expertSlots` のコメント)。全常駐の方が単純で
この機械の RAM にも載るから設計は変えなくてよいが、**RAM が苦しくなったら
約 400 MB を返せる**裏付けとして記録する。

### 1-6. 写すものの対応表

| NVMAI | 本ランタイム | 備考 |
| --- | --- | --- |
| `StreamingMTPDecoder` (target/draft の 2 runner 所有、prepare/advance) | 新規。`QwenForwardRunner` を 2 つ持つ兄弟型 | `LogitProducer` 相当の結線は `RunQwen.swift` / `QwenServerSession` の生成ループ |
| `verifyGreedyPair` (T=2 prefill + 2×argmax + hidden 退避) | `QwenPrefill` のチャンク経路に幅 2 の特化を足す | 融合ヘッドは logits を書かないが argmax は書く — **2 行版の頭**が要る ([25](25-CLI-TOOLS.md) のマスク付き再畳みと同族) |
| `snapshotGDNAfterFirstToken` | `qwen_delta_rule` に第 2 状態出力を足す | [15](15-PHASE2-GDN.md) の状態書き出しの複製 |
| `encodeSpeculativeRestore` | `RecurrentStateManager` に影 2 本 + blit | 連続バッファなので NVMAI より簡単 |
| `kv.rewind(to:)` | `KVCacheManager` にカーソル巻き戻しを足す | ring 化した領域の位置管理と要整合 |
| `MTPStatistics` | そのまま | 受理率 / emit/パス |
| `StreamingMTPMemoryPlan` (部品別の明示予算 + 超過は構築時エラー) | そのまま | 予算値はこの機械で引き直す |

---

## 2. prompt cache — snapshot-restore 型 (「持たない」の再考)

出典: `sources/NVMAI/Runtime/KVCache/InferenceStateSnapshot.swift` /
`RealForwardRunner.captureInferenceState` / `GDNStateManager.restoreSnapshot` /
`sources/NVMAIServer/Core/ServerPromptCache.swift` / `ServerPromptStateStore.swift`。

[26](26-PHASE8-SERVER.md) の「持たない」の根拠は**巻き戻せない**ことだった。
NVMAI の解は巻き戻さない: **生成が終わった時点の状態を丸ごと写真に撮り、
次の要求が前回の厳密な延長だったときだけ復元して続きから prefill する。**

- **payload**: KV セグメント列 + GDN セグメント列 (状態 + conv tail を層順に
  連結) + `position`。descriptor は版番号と各セグメント長を持ち、復元時に
  レイアウト全体を検査してから memcpy — 形が 1 バイトでも違えば落ちる
- **一致判定** (`ServerPromptCache.match`): (S12) レンダ済みプロンプトの
  先頭が entry の KV-backed トークン列と一致する直接ヒットと、
  「前回の入力 + 前回の assistant 応答 + 続き」という**会話延長ヒット**の
  2 段。tools の一致も要求。`uncommittedBoundaryTokenIDs` (KV に入っていない
  境界トークン、常に 1 個) を継ぎ目に挟んで再レンダとの食い違いを防ぐ
- **domain 束縛**: モデル ID・snapshot hash・runtime profile hash・文脈長・
  KV 形式・テンプレート SHA256。どれか 1 つでも違えば別世界としてミス
- **publish 条件**: `endOfTurn` / `toolCalls` / `maxTokens` で止まった
  完全な 1 ターンのみ。stop-string で切ったものは載せない
- **保管**: `ServerPromptStateStore` がメモリ層 (既定 1 エントリ) を持つ

**→ 取り分と欠品は [34](34-PROMPT-CACHE-ESTIMATE.md) が机上で出した。**
以下の読み替えのうち **(a) は訂正される** — capture/restore に**コピーは要らない**
([34 §3-1](34-PROMPT-CACHE-ESTIMATE.md))。「前回までのトークン数 × prefill の ms/tok」
という取り分の式も、prefill に **1.30 秒の床**があるぶん過大である
([34 §1-1](34-PROMPT-CACHE-ESTIMATE.md))。**番号の大きい 34 が正。**

本ランタイムへの読み替え:

- **器は既にある。**Gemma 側の `Sources/TurboFieldfareServer/Core/ServerPromptCache.swift`
  が同じ役割を持っている。足りないのは (a) `QwenForwardRunner` +
  `RecurrentStateManager` の capture/restore (連続 2 バッファなので
  memcpy 2 回 + KV)、(b) `QwenServerSession` の `cache_n` 結線
- **費用**: 1 エントリ = 固定 62.8 MiB + conv tail + full-attention 10 層
  ぶんの KV × position。既定 1 エントリなら [26](26-PHASE8-SERVER.md) が消した
  `ExpertCacheBudget` の勘定に 1 行足すだけで収まるかを先に計算する
- **効き所**: エージェントのツールループ (前回プロンプト + ツール応答の延長)
  と多ターン会話。[27](27-PHASE6-THROUGHPUT.md) で TTFT を 2.4 → 1.5 秒まで
  詰めた文脈で、**多ターンの 2 回目以降は prefill を丸ごと払い直している**のが
  現状。取り分は「前回までのトークン数 × prefill の ms/tok」で、実タスク
  ([27 §2](27-PHASE6-THROUGHPUT.md) の 4 本) から先に机上で出せる
- **[26](26-PHASE8-SERVER.md) の記述の更新が要る**: 「prompt cache は無い」を
  「延長一致のみの snapshot-restore 型」に。`cache_n` が 0 でなくなるのは
  延長ヒット時だけ。**巻き戻しは今後も無い** — 途中で分岐した会話は全ミス

---

## 3. decode の hit-fixup (v4.1) — 写像待ちに常駐分を重ねる

出典: `docs/v4.1-expert-streaming-engine.md` (NVMAI の生産既定)。

routed phase-1 を**常駐 (hit) 分と miss 分に割り**、hit 分と shared を
即 commit → miss は予約済みスロットへ非同期に読み → **fixup コマンドが
miss 位置だけ** phase-1 を畳み → phase-2 の縮約は元の top-k 順で全行を消費。
スロットは `empty → loading → resident (+pinned)` の明示状態を持ち、
**lease が世代を GPU の最終消費まで pin する**。all-hit / all-miss の層は
単純な全相経路のまま。旧経路は環境変数で A/B に残す
(`NVMAI_DECODE_EXPERT_EXECUTION=barrier`、未知値は fail-closed)。

本ランタイムでの意味:

- [27](27-PHASE6-THROUGHPUT.md) の pipeline は**遅延 join と shared の重ね**
  まで。**hit した routed エキスパートの計算はまだ miss の読み
  (= ホスト写像 [27 §9](27-PHASE6-THROUGHPUT.md)) を待っている。**
  32 スロットのヒット率 74.9% ([27 §6](27-PHASE6-THROUGHPUT.md)) ぶんの
  phase-1 を待ちの陰に入れられる
- 既存の 3 コマンドバッファ pipeline と直交ではない — join の位置が変わる。
  `TF_QWEN_PIPELINE` と同格の switch で旧経路を残し、[27 §2](27-PHASE6-THROUGHPUT.md)
  と同じ実タスク 4 本で A/B する
- 教訓も 1 つ写す: NVMAI の「全 phase-1 が I/O を待つ barrier」は設計ではなく
  **Swift 配列の値意味論のバグ** (空の snapshot を取ってから埋めていた) だった。
  hit/miss 分割の配列受け渡しはここが再発点

---

## 4. サンプラ (R3 の解消) — 障害物はヘッド

出典: `sources/NVMAI/Runtime/Generation/Sampler.swift`。

NVMAI は温度 0.6 / Top-K 20 / Top-P 0.95 (Qwen 推奨既定) の GPU サンプラを
持つ。写す価値があるのは:

- **mlx-lm 順の truncation**: Top-P を全語彙の分布から先に取り、Top-K で
  上限を掛け、温度は最後の draw にだけ効かせる — 参照実装と一致する順
- **検証**: `presencePenalty` は 0 のみ受理 (非対応を明示エラーに)、
  「topK なしの topP < 1」も明示エラー — **静かに違うサンプラになる道を塞ぐ**
- repetition penalty の**増分履歴** (id → 出現数の辞書をパス間で持ち回す)

ただし本線の障害物はサンプラではない: **融合 LM head は logits をどこにも
書かない** ([19](19-LM-HEAD-INT8.md))。argmax しか出ないから greedy なのであって、
サンプラを写しても入力が無い。要るのは logits (または top-N 候補) を書く
ヘッドの変種で、[25](25-CLI-TOOLS.md) のマスク付き再畳みと同じ「もう 1 回
畳む」系の追加になる。**費用はヘッド 1 回 ≈ 4.0 ms/tok** ([19](19-LM-HEAD-INT8.md))
が上限の目安 (書く範囲を絞れば下がる)。MTP (§1) は greedy 限定なので、
Phase 7 を先に判定してからで遅くない。

---

## 5. 測定作法 — interleaved A/B と昇格ゲート (v4.3)

出典: `docs/v4.3-predictive-prefetch-plan.md`。中身は昇格に**失敗した**
予測先読みの記録だが、作法が良い。写すのは 3 つ:

1. **逐次スイープの禁止** (実測(NVMAI)): sidecar スロット数の逐次スイープは
   32 スロットが約 11% 勝つように**見えた**が、正体は前の構成が温めた
   ページキャッシュ。interleaved にしたら差は消えた。mmap の腕で動いている
   本ランタイムはこの誤りに**より**弱い。[31](31-PREFETCH-CHEAPER.md) 以降の
   A/B は interleaved を既定にする (クールダウン 20 秒の既存規約に追加)
2. **昇格ゲートの明文化**: 対象条件で +X%、greedy 出力のバイト一致、
   p95 非悪化、メモリ予算内 — を**実装前に**書く。
   [04](04-PHASES.md) 次の一手 #29 (先読みの既定化) の判定枠に使える
3. **実需 > 投機の優先度と staging の分離**: 投機読みは正規スロットを
   触らず専用リングに落とし、実需と重なったら join する (P0/P1)。
   [31](31-PREFETCH-CHEAPER.md) の先読みを昇格させる場合の安全側の形

---

## 6. 写さないもの

| もの | 理由 |
| --- | --- |
| v4.2 event-driven Metal I/O | NVMAI 自身が資格審査で外した (`Document Metal I/O qualification limits`)。生産既定は v4.1 のまま |
| 予測先読みの実装そのもの | NVMAI では昇格失敗 (4-bit **−3.9%**)。こちらは既に [31](31-PREFETCH-CHEAPER.md) が +2.4〜3.1% を測っており、ボトルネックが違う (SSD 帯域 vs ホスト写像) ので**こちらの結果が正** |
| thinking 切り替え | `enableThinking` を CLI / サーバーとも既に持っている |
| verified-install の receipt / golden-baseline | 役割は `--verify-install` と参照一致検査 (`--qwen-decode` ほか) が既に果たしている |

---

## 7. 写す順 (提案 — 判断はユーザー)

1. **§1-4 の判定線だけ先に**: `mtp_acceptance.py` を p=0.585 の線で回す。
   カーネル 0 本で Phase 7 の decode 側の生死が出る。死んでも §1 の
   checkpoint/restore は先読みの神託 ([29](29-MTP-PREFETCH-OUTLOOK.md)) に流用できる
2. **§2 prompt cache**: 独立で、効き所 (多ターン TTFT) が既知。机上の取り分
   計算 → capture/restore → `ServerPromptCache` の Qwen 対応の順
3. **§3 hit-fixup**: [27 §9](27-PHASE6-THROUGHPUT.md) の残りの待ちへの正攻法。
   実装は最重量なので、§5-2 の昇格ゲートを先に書いてから
4. **§4 サンプラ**: logits を書くヘッドの変種が前提。Phase 7 の判定後
5. **§5 測定作法**: コストほぼゼロ。次の A/B から適用
