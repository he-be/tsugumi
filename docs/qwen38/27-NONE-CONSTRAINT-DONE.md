# 27. 26 の実施: `none` のトークン禁止、`toolBudgetNotes` の撤去、本文の後の呼び出しの描き直し、S-2

書いた日: 2026-09-15。表記は 01 と同じ (実測 / 導出 / 未確認)。
方針は 26。走行の記録は `scratch/qwen38/branch26/m6/` (git 管理外)。

## 0. 結論

1. **`tool_choice: none` はサンプラの拘束になった (26 §2)。** 文法の代わりに `ForbiddenTokensConstraint` がツール呼び出しの開始トークン (Qwen `<tool_call>`、Gemma `<|tool_call>`) だけを候補から落とす。宣言はプロンプトに残す。Qwen3.8・Ornith・Gemma の 3 セッションと Ornith の CLI に入れた。SPEC GEN-4 を改めた。
2. **M6 の分岐点 k6 で、禁止なしなら `<tool_call>` を書く引きが、禁止つきでは 0 になった (実測、各 16 回)。** MTP on でも 0。
3. **`toolBudgetNotes` (rounds / context) を撤去した (26 §3)。** 残したのは上限の 1 行 (事実の通知) だけ。
4. **本文の後の `<tool_call>` の描き直しを直した (26 §4)。** アプリが生成した assistant 手番は、モデルが本文の後に書いた空白をそのまま置く。上流テンプレートの 2 行を差し替え、OpenAI 経路 (クライアントの手番) は今までどおり `\n\n`。
5. **S-2: 文脈の末尾で切れた呼び出しを文脈超過として報告する (26 §5)。** 3 セッションとも。
6. 全単体テスト 1,598 件緑 (§6)。

## 1. `none` の拘束

| 場所 | 変更 |
| --- | --- |
| `Runtime/Generation/ForbiddenTokensConstraint.swift` (新) | 禁止する id の集合。`allows` はそれ以外で真、`fillAllowedMask` は全部真にして id だけ偽、`accept` は何もしない、`mayEndHere` は常に真。思考の中でも同じ id を禁止する |
| `ChatGrammarBuilder` / `QwenChatGrammarBuilder` | 戻り値を `ChatConstraint` (`.grammar(ChatGrammarConstraint)` / `.forbiddenTokens([Int32])`) に。`.none` は開始トークンの禁止 (宣言が空でも同じ)。`response_format` があるときは今までどおり response format の文法 (GEN-12) |
| `ServerGenerationPlan` / `QwenGenerationPlan` | `forbiddenTokenIDs` と `forbiddenTokensConstraint()`。`isConstrained` は禁止も含む (Gemma は logits ヘッドが要る側に入る) |
| `ServerInference` (Gemma) / `QwenServerSession` / `Qwen38ServerSession` | 文法の拘束 (`grammarConstraint`、思考中の抑止に使う) と、復号ループに渡す拘束 (文法か禁止) を分けた。Gemma の fused greedy ヘッドの拒否は `plan.isConstrained` で判定 |
| `TsugumiCLI/RunQwen.swift` | `--tool-choice none` も同じ禁止 |
| `AppModel.startNextRound` | コメントを「`none` はサンプラで呼び出しを止める、上限の行は通知」に |
| `docs/serving/SPEC.md` GEN-4 | `none` は開始トークンの id を候補から落とす |

MTP の投機経路は同じ `ConstraintGate` を通るので、ドラフトが `<tool_call>` でも検証の引きで落ちる (§2 で確認)。

## 2. 分岐点 (M6 の k6、上限のラウンド)

`--qwen38-branch-probe` の manifest に `forbid` (引きに `ForbiddenTokensConstraint` を渡す) と `speculative` (MTP ヘッドつきのエンジン) を足し、出力に `draws_tool_call` (`<tool_call>` を含む引きの数) を足した。
trunk は 25 の k6-none (8,129 トークン)、案は 25 の k6-none と k6-limit、公式サンプラで 16 回 × 12 トークン。容量 8,183。

| 走行 | 案 | P(`<tool_call>`) | `<tool_call>` を含む引き | 引きの書き出し |
| --- | --- | ---: | ---: | --- |
| 禁止なし | k6-none (行なし) | 0.9999 | 16 / 16 | `<tool_call>\n<function` × 16 |
| | k6-limit (上限の行) | 0.6747 | 7 / 16 | `<tool_call>` × 7、`## M6 Mac` × 9 |
| 禁止つき | k6-none | 0.9999 | **0 / 16** | `M6 Mac mini` × 11、`検索結果を整理` × 5 |
| | k6-limit | 0.6747 | **0 / 16** | `## M6 Mac` × 16 |
| 禁止つき、MTP on | k6-none | 0.9999 | **0 / 16** | 禁止つきと同じ |
| | k6-limit | 0.6747 | **0 / 16** | 禁止つきと同じ |

- P はモデル自身の確率 (禁止の前) なので、3 走行で同じ。禁止つきの引きでは `<tool_call>` の確率が 0 になる (構成から、`GEN_4_forbiddenTokensLeaveOnlyThoseIDsOut`)。
- k6-limit の P は 25 §3 の 0.3046 と違う。25 の走行は trunk を 6 か所 (4265 / 4275 / 7201 / 7211 / 8118 / 8128) で切り、今回は 2 か所 (8118 / 8128)。25 §2-2 のチャンク境界の件に当たるが、この 2 走行の差の原因は確かめていない。
- 走行時間: trunk の prefill 88.5 / 87.7 / 97.2 s。見張り: Swapouts 最大 +2,048 / +32 / +16,280 ページ (MTP on は 60 秒窓 +13,660、止める条件の内側)。

## 3. `toolBudgetNotes` の撤去

6fb2e40 で足した `AppToolBudgetNotes`、`AppModel.toolBudgetNotes` / `toolRoundContextTokens` / `roundBudgetNote`、テスト `budgetNotesGoOnEachRoundsLastResult` / `budgetNotesGroupDigitsAndStopAtZero`、`TsugumiToolLoopCheck --budget-notes` を消した (その差分の逆適用)。
`roundBudgetReachedNote` (上限の行) と `exhaustedRoundsForbidCallsButKeepTheDeclarations` は残る。
分岐点の道具 (`--qwen38-branch-probe` / `--qwen38-path-check` / `--qwen38-restore-check`、`Scripts/qwen38/budget_branches.py`) は測定器なので残した。

## 4. 本文の後の `<tool_call>` (INV-1)

### 4-1. 記録から (実測)

- アプリは本文を**末尾の空白ごと**保存している: 21 full2 の en-swift 2 ターン目の手番は `…が必要なら再取得します。\n`。trim して `\n\n<tool_call>` を置くのは上流テンプレート (`content|trim`、127〜130 行)。
- `scratch/` の `turns.jsonl` 14 走行で、本文つきの呼び出しは Qwen3.8 で 14 回。区切りは `\n\n` 12 回 (テンプレートと一致)、`\n` 1 回 (21 §9 の件)、`\n\n\n` 1 回。Gemma の 36 回はすべて本文なし。

### 4-2. 変更

- `GFTokenizer.Message.contentIsGenerated`: content がこのランタイムの生成した文字列 (空白を含む) であること。アプリ (`RealInferenceSession.validatedChatRequest`) は assistant 手番に真を置く。サーバの要求解析は置かない。
- `QwenTokenizer`: `contentIsGenerated` で呼び出しのある手番に `tsugumi_call_separator` (content の末尾の空白) を渡し、テンプレートの 2 行を差し替える。本文ありの `'\n\n<tool_call>…'` と本文なしの `'<tool_call>…'` の前に、渡されたらその区切り、無ければ元の `\n\n` / 空。宣言の `tojson` の差し替え (`toolDeclarationTemplate`) と同じ文字列に入るので、ツールの宣言が無い要求では差し替えは効かない。
- 範囲は continuation と履歴の両方。どちらも同じ `AppChatTurn.text` から描くので、ターンの中だけ直すと、ターンの終わりに履歴へ畳んだ時点で次ターンの 1 ラウンド目が分岐する。
- 本文の先頭の空白 (テンプレートの trim が落とす) は扱っていない。生成プロンプトが `\n\n` で終わるので、記録には無い。

### 4-3. 検査

- `Qwen38ToolLoopPromptTests`: 会話の本文を保存される形 (`1 件目のページを読みます。\n\n`) にし、spec.json を更新。`Scripts/qwen38/tool_loop_fixture.py` で描き直した t1r1〜t3r1 は変わらず、Swift の描画と一致。
- 同 `separatorIsRedrawnAsGenerated`: 区切り `\n` / なし / `\n\n\n` / ` \n` で、生成した本文 + 呼び出しが次の要求の描画の接頭辞になり、live から続く。
- 同 `clientTurnKeepsTheTemplateSeparator`: `contentIsGenerated` が偽なら `読みます。\n\n<tool_call>`、真なら `読みます。\n<tool_call>`。
- 実モデルで en-swift を流し直すことはしていない (再現は n = 1 の事象で、判定はトークン列の接頭辞)。

## 5. S-2

`ServerRequestError.generationReachedContext(stop:promptTokens:generatedTokens:reserved:maxContext:)`: 止まった理由が `maxTokens` で、prompt + 生成 + 検証行 (Qwen の MTP は 1) が文脈に達していれば `exceed_context_size_error` (`context_length_exceeded`、「prompt of N tokens and M generated tokens reached the configured context of C inside a tool call」)。
3 セッションとも、復号器が呼び出しを読み終えられなかったとき、構造化出力の失敗より先にこれを見る。要求の `max_tokens` で止まった場合は今までどおり失敗。
アプリには `invalidRequest` として届き、`structured_output_failure` の再試行には入らない。

テスト `GenerationReachedContextTests`: 21 §9 の 2 件 (12,199 + 88、12,286 + 1、12K、MTP on) が超過になる。1 トークン足りない場合・`endOfTurn`・文脈に余裕がある場合は nil。

## 6. テスト

§1〜§5 の関係スイート (`GenerationConstraintTests`、`ChatGrammarBuilderTests`、`QwenChatGrammarBuilderTests`、`QwenGenerationPlanTests`、`ServerGenerationPlanTests`、`ServerGrammarWiringTests`、`AppModelToolLoopTests`、`Qwen38ToolLoopPromptTests`、`GenerationReachedContextTests`) は緑。全体 (`Scripts/test.sh`): 1,598 件緑 (97 s)。

## 7. 残り

- **Gemma の本文の後の呼び出し (未観測、コードから導出):** 同梱テンプレートは assistant 手番の `tool_calls` とそれに続くツール結果を content より先に描く。Gemma が本文を書いてから呼び出した場合、描き直しは手番の先頭で分岐する。記録 36 回には無い。直すなら Qwen と同じく「生成した文字列を置く」側で、テンプレートの順序に手を入れることになる。
- `none` のラウンドの実会話 (26 §2-3 の 4、ja-m6ssd を 1 本) と Gemma のツールループ 1 本 (同 5) は流していない。判定は §2 の分岐点と単体テスト。
- k6-limit の P が走行で 0.30 / 0.67 と違う件 (§2)。
