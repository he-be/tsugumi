# Mac アプリの UX 要件 — 既存チャットアプリからの借用と採否

作成 2026-09-02。Tsugumi の狙いは「夕食レシピ、時事ネタ、キャンプ飯、ジョーク」のような日常の質問を、
根拠を取りに行った上で答えるローカルのチャットです。Web の Gemini が URL も検索も断ってそれっぽく延長する、
という不満の裏返しなので、判断基準は「根拠を取りに行ったことが見えるか」「答え方を後から直せるか」の 2 つです。

要件は発明せず、Open WebUI / LibreChat / ChatGPT / Claude / Perplexity / LM Studio に既にある機能を表にして採否を決めました。
採るものは Core (`TsugumiAppCore`) の状態と意味論をテスト先行で書き、View は最小で載せます。

## 1. 要件表

| # | 機能 | 前例 | 採否 | この実装 |
|---|---|---|---|---|
| U1 | コンテキスト残量の表示 | LM Studio, Open WebUI (token counter) | 採 | HUD に「使用 / 上限」。直前ターンの prompt tokens + 生成 tokens |
| U2 | Thinking の ON/OFF をコンポーザから | LM Studio, Claude (extended thinking) | 採 | コンポーザに脳アイコンのトグル。Inspector のスイッチと同じ値 |
| U3 | 回答の下の操作行 (コピー・再生成) | 全部 | 採 | 最後の回答の下だけ。過去ターンは対象外 |
| U4 | 再生成の別回答を残して切り替え (1/2) | ChatGPT, Claude, Open WebUI | 採 | `AppChatSession.outputVariants`。選択中のものだけ履歴に折り込む |
| U5 | 文体の指定つき再生成 (短く・率直に) | ChatGPT (旧 shorter/longer), Perplexity Rewrite | 採 | 置き換え。指示はユーザーターンの末尾に 1 行足して送る |
| U6 | 反対の立場で | Claude/ChatGPT の follow-up 定型 | 採 | 追加ターン。元の回答は残す |
| U7 | 根拠バッジと「検索して答え直す」 | Perplexity, ChatGPT Search チップ, Gemini「Google で確認」 | 採 | 検索と取得を分けて数える。検索なし・取得 0・参照なしは警告色。押すと Online の方針 (必ず検索して読む) で再生成 |
| U7b | Online は必ず検索して読む | Perplexity (検索が前提のモード) | 採 | 1 ラウンド目 `function: web_search`、取得未試行なら `function: fetch_page` のみ宣言。モデルに「読むか」を判断させない (docs/WEB_SEARCH.md §2) |
| U8 | 指示層 (自認・あなたについて・答え方) | ChatGPT Custom Instructions, Claude Styles, Gemini Gems | 採 | `persona.json`。空でなければシステムプロンプトの先頭。ツール無しでも付く |
| U9 | 回答中の URL をクリックで開く | 全部 | 採 | Markdown のリンクと裸 URL に `.link` |
| U10 | 自動メモリ (会話から事実を抽出) | ChatGPT Memory, Open WebUI Memory | 否 | 小さいモデルでは誤抽出のノイズが勝つ。U8 の明示欄で代替 |
| U11 | 過去ターンの編集と枝分かれ | ChatGPT, Open WebUI | 保留 | 転写が 1 本の NSTextView で、ターン単位の操作行を持たない。U3 の後で再検討 |
| U12 | 自動タイトル | Open WebUI, ChatGPT | 保留 | decode 1 本で済むが、生成中は他チャットが読み取り専用になる制約と衝突する |
| U13 | 定型プロンプト / presets | LibreChat, Open WebUI | 否 | U8 で足りる |
| U14 | 関連質問の提案 | Perplexity, Gemini | 否 | decode を余計に使う |
| U15 | 引用番号を本文に埋める | Perplexity | 否 | モデル側の出力形式の問題。「参照:」の列挙で足りる |
| U16 | チャット内検索・ピン留め | 全部 | 保留 | 件数が増えてから |
| U17 | マシンの空きを速度計で | Activity Monitor のメモリプレッシャー、LM Studio の RAM 表示 | 採 | HUD 右端の針。借りられるメモリ ÷ モデルの重み。針の横に説明は置かず、クリックで内訳 (docs/MAC_APP.md §4e) |

## 2. 意味論 (Core)

- **再生成 (U4/U5/U7)** は「同じユーザーターンをもう一度」です。いまの回答は `outputVariants` に退避し、
  新しい回答が `outputText` に入ります。`selectVariant(_:)` で入れ替えると、次の run が折り込むのは選択中の回答です。
- **指示 (U5/U7)** はユーザーターンの末尾に足した 1 行として送ります (`AppAnswerDirective.instruction`)。
  折り込んだ履歴のユーザーターンもその文面を持ちます。モデルが見た通りに描き直すのがプロンプトキャッシュの前提です。
  表示上の `outputPromptText` は元の質問のままで、指示は `outputDirective` に別に持ちます。
- **反対の立場で (U6)** は `askFollowUp(_:)`。プロンプト欄に定型文を入れて run するのと同じで、元の回答は履歴に折り込まれます。
- **検索して答え直す (U7)** は Online 固定の再生成です。Online 自体が「検索 → 取得 → 回答」を強制するので (U7b)、
  指示行は読むことと「参照:」を求める文だけを足します。キーが無ければ従来のエラー。
- **根拠バッジ (U7)** は `outputToolTrace` を web_search / fetch_page / wikipedia_* で数えます (`AppAnswerGrounding`)。
  Wikipedia だけのターンは「Wikipedia N 件」。何も無ければ「検索なし」で、ツールが宣言できる状態なら警告色。
  検索したのに取得 0、または回答に「参照:」も URL も無い (`outputLacksCitation`) ときも警告色です。
  バッジは「検索した」と「読んだ」と「挙げた」を別々に保証します。
- **指示層 (U8)** は `AppPersona` の 3 欄 (自認・あなたについて・答え方)。空欄は出しません。
  3 欄とも空ならシステムプロンプト自体を足さず、従来の描画と同じになります。
- **コンテキスト使用量 (U1)** は直前ラウンドの `promptTokenCount + generatedTokens`。生成中は前回値に生成分を足します。
- **速度計 (U17)** は `AppMachineHeadroom`: 借りられるメモリ (物理 − アプリ − wired − 圧縮) ÷ ストリームする重み。
  decode の速さではなく機械の空き。速さとの対応は docs/MAC_APP.md §4e の表 (短い雑談ではほぼ効かず、長い文脈で半減)。
