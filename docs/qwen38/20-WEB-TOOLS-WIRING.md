# 20. Web 検索ツールの配線 — アプリの宣言を Qwen3.8 に通し、実運用の形で流した最初の結果

書いた日: 2026-09-14。対象は [19](19-RESPLIT-HISTORY.md) までの `Qwen38ServerSession` と Mac アプリのツールループ ([WEB_SEARCH.md](../WEB_SEARCH.md))。表記は 01 と同じ (実測 / 導出 / 未確認)。

## 0. 結論

1. アプリの Search / Online が Qwen3.8 でも宣言されるようにした (`AppModelKind.supportsTools`、Ornith 以外)。server 側のツール文法・デコーダ・`<tool_call>` の直前のチェックポイントは 17 / 18 のまま。
2. ツール宣言の描き方を上流に合わせた (17 §3-1)。swift-jinja の `tojson` はキーを辞書順・空白なし・`\/`・非 ASCII を `\uXXXX` にするので、アプリの日本語の説明が escape の列になっていた。
   `QwenToolDeclaration` が HF の `json.dumps(ensure_ascii=False)` と同じ綴り (クライアントが書いたキー順、`", "` / `": "`) で書き、テンプレートの `{{- tool | tojson }}` 1 行をその文字列に差し替える。Ornith と共有 (`QwenTokenizer`)。
3. モデル無しの検査 (`Qwen38ToolLoopPromptTests`): アプリの要求組み立て (`RealInferenceSession.validatedChatRequest`) で 2 ターンのツールループ 6 要求を描き、
   **6 本とも Python (HF jinja) の描画と文字列一致**、**ラウンド・ターンをまたいで生きている状態から続く (INV-1)**。宣言の差し替えを外すと描画一致が落ちる (負例)。
4. 実モデルで、アプリの AppModel を画面なしで回す検査 `TsugumiToolLoopCheck` を作り、6 会話 × 2〜3 ターン (Online、12K、MTP on、公式サンプラ) を流した。ユーザー指示で途中で止めた (§3)。
   - 描き直しが原因で cache が外れたラウンドは 0 (完了したラウンド間の shortfall は、下の 2 の場合を除き全部 0)。
   - **落ちた形は 2 つ**:
     1. **文脈超過のエラー** (4 ターン): ツール結果 (日本語のページ 1 枚 1,300〜3,300 トークン) を積み、次の要求が 12,288 を超える。
     2. **ラウンド上限 (6) で全部 prefill し直す** (2 ターン): アプリは上限でツール宣言を外す (`tools: []`) ので、システムブロックが変わり cached 0。宣言を戻す次のターンの 1 ラウンド目も cached 0。

## 1. 置いたもの

| 層 | ファイル | 中身 |
| --- | --- | --- |
| Tsugumi | `Tokenization/QwenToolDeclaration.swift` | 順序つき JSON (`OrderedJSON`、パースと Python の `json.dumps` 綴り)。`parametersSource` が `parameters` と同じ値ならそのキー順、無ければ辞書順 |
| Tsugumi | `Tokenizer.swift` | `FunctionDefinition.parametersSource` (クライアントが書いた JSON テキスト、省略可) |
| Tsugumi | `QwenTokenizer.swift` | テンプレートに `{{- tool \| tojson }}` がちょうど 1 行あれば `{{- tool.tsugumi_json }}` に差し替えた版で描く。無ければ従来どおり |
| app | `AppModelKind.supportsTools`、`AppModel.toolsAvailable` | Gemma と Qwen3.8。Inspector の文言を「Ornith では使えません」に |
| app | `RealInferenceClient` | ツールの `parametersJSON` を `parametersSource` として渡す |
| test | `Qwen38ToolLoopPromptTests`、`Fixtures/qwen38-tool-loop/` | §2。`spec.json` はテストが書く会話、`<label>.txt` は `Scripts/qwen38/tool_loop_fixture.py` が上流の描画で書く |
| test | `AppModelToolLoopTests.qwen38DeclaresTheToolsWithoutAThinkingBudget` | Online で 4 本を宣言し `web_search` を強制、thinking off で思考予算なし |
| check | `Sources/TsugumiToolLoopCheck/`、`Scripts/qwen38/tool_loop_conversations.json` | §3 |

server の HTTP 経路 (`ChatRequestParser`) は JSON テキストを持たないので、宣言のキー順は辞書順のまま (空白と非 ASCII は上流と同じになった)。

## 2. モデル無しの検査

会話: Online・ローカル Wikipedia あり (宣言 4 本、`AppModel.makeToolExecutor` の順)、2026-09-14 固定のシステムプロンプト。
1 ターン目 = アプリの事前引き当て (`wikipedia_lookup`) → `web_search` (強制) → 前置きの文 + `fetch_page` (強制) → 回答、2 ターン目 = `wikipedia_page {"from":2000,…}` (整数引数) → 回答、3 ターン目の 1 要求。
各ラウンドの生成は文法が許す綴り (パラメータ昇順、文字列は生、それ以外はコンパクト JSON、前置きの後は `\n\n`)。

| 検査 | 結果 |
| --- | --- |
| 6 要求の描画 == HF jinja の描画 | 一致 (差し替えを外すと 6 本とも 83 文字目で不一致: `{"function":{"description":"この Mac…` 対 `{"type": "function", "function": {"name": …`) |
| 前の要求 + 生成 + `<|im_end|>` の生きている状態から次の要求 (`Qwen38PromptCache.aligned` → `decide`) | 5 組とも `live` |

`Scripts/test.sh --filter "AppModelToolLoop|AppModelKind|Qwen38|PromptCache|DecodeService|QwenTurnRedraw|QwenChatGrammar"`: 106 件緑。

踏んでいない形: 前置きの文と `<tool_call>` の間が `\n` 1 つ (描き直しは `\n\n` にするので分岐する)、回答の端の空白 (`content|trim`)。

## 3. 実モデル (`TsugumiToolLoopCheck`、`scratch/qwen38/tools20/`)

```
Scripts/qwen38/guarded.sh <log> .build/release/TsugumiToolLoopCheck --out <dir> [--only a,b]
```

AppModel・実行器 (Serper → 自前 fetch / Jina、ローカル Wikipedia)・設定ファイル (`web-search.json`・`persona.json`) はアプリのもの。推論クライアントだけ、プロセス内の `RealInferenceClient` (DecodeService が回すのと同じセッション) に要求と診断を記録する薄い代理をかぶせた。DecodeService のソケット層は通っていない。
判定 (ターンごと): `live` = 2 ラウンド目以降と次のターンの 1 ラウンド目の cached == 前のラウンドの prompt + generated − 1、`error` 無し、`answer` 非空、`online` = 検索した・取得を試した・回答が出典を挙げる。
設定: 12,288、MTP on、thinking off、ラウンド上限 6、ページ 6,000 字、検索 8 件。各 n = 1。

### 3-1. 走行

| 走行 | 内容 | 終わり方 | wired 最大 / file-backed 最小 | Swapouts |
| --- | --- | --- | --- | ---: |
| smoke | en-url 2 ターン | 2 ターン目 5 ラウンド目 (prompt 10,690) の prefill 中に見張りが停止 | 14.63 / 1.02 GB | +7,276 ページ |
| mem1 | en-url 2 ターン (検査用のメモリ内訳ログつき、ログはコミットしていない) | 完走 | 14.38 / 1.12 GB | +140 |
| full1 | 6 会話 | ja-price の後、en-facts の 1 ラウンド目でユーザー指示により停止 | 14.66 / 1.17 GB | +96 |

### 3-2. full1 のターン

| 会話 | ターン | ラウンド | 秒 | 検索 / 取得 / Wikipedia | 結果 |
| --- | ---: | ---: | ---: | --- | --- |
| ja-news | 1 | 5 | 300 | 1 / 3 / 0 | ok (最後の prompt 10,277、回答 965 トークン) |
| ja-news | 2 | 2 | 18 | — | **文脈超過** (12,526 > 12,288) |
| en-swift | 1 | 7 | 270 | 3 / 3 / 0 | **live 不合格**: 7 ラウンド目 (上限でツールを外す) cached 0 / 4,767、prefill 63.0 s |
| en-swift | 2 | 7 | 391 | 1 / 4 / 0 | **live 不合格**: 1 ラウンド目 cached 0 / 5,952 (75.2 s)、7 ラウンド目 cached 0 / 11,229 (142.1 s) |
| en-swift | 3 | 1 | 0 | — | **文脈超過** (12,593) |
| ja-howto | 1 | 5 | 177 | 1 / 4 / 2 | **文脈超過** (14,640) |
| en-url | 1 | 4 | 155 | 2 / 1 / 0 | ok |
| en-url | 2 | 4 | 189 | 1 / 3 / 0 | ok |
| ja-price | 1 | 6 | 211 | 1 / 3 / 1 | **文脈超過** (12,369) |

完了したラウンドの decode は 6.6〜8.6 tok/s。ツール呼び出しだけのラウンドは生成 35〜85 トークン。

### 3-3. ツール結果の大きさ (次のラウンドのプロンプト増分から前の生成を引いた値、導出)

| 中身 | トークン |
| --- | ---: |
| システムプロンプト + 宣言 4 本 + 質問 | 1,627〜1,785 (1 ラウンド目の prompt、事前引き当てなし) |
| `web_search` (8 件) | 600〜850 |
| `fetch_page` 日本語 (6,000 字で打ち切り) | 1,300〜3,300 |
| `fetch_page` 英語 | 約 1,350 (6,000 字)、約 600 (1,893 字) |

### 3-4. 記録だけ (解釈は n = 1 なので書かない)

- 同じ URL の `fetch_page` を同じ引数で続けて呼んだ: mem1 の en-url 2 ターン目で 3 回 (本文は「…(本文はここで打ち切り)」)、full1 の ja-howto で 2 回、en-swift で 2 回。
- 英語の質問への回答も日本語 (システムプロンプトが「回答は日本語で書き」)。
- 日本語の回答に中国語の語が 1 か所 (`找到当てる`、mem1)。

## 4. 残り

**続きの整理は [21](21-TOOL-LOOP-TRIAGE.md)** (同じ URL の再取得は `fetch_page` の打ち切りに続きが無いこと、上限の cache 外れは server の `tool_choice: none` の扱い)。

- §0-4 の 2 つの落ち方 (文脈超過、上限でツールを外すと cache が外れる)。後者は Gemma と共有のアプリの経路 (WEB_SEARCH §2 の「宣言は毎回全部にする」に上限の分岐だけが反していた)。
- ツール結果を毎回数千トークン読む必要があるか (ページの先頭 6,000 字を中身を選ばず渡している) はユーザーから問いが出ている。未着手。
- smoke の Swapouts +7,276 (文脈 10K 台で decode の後に prefill する形) は 1 回だけで、原因は未確認。
- en-facts は未走行。DecodeService 経由 (アプリそのもの) と GUI は未確認。
