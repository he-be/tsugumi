# 知識源 3 通り × エージェント動作の実験計画

最終更新: 2026-09-16。note 記事「ローカル LLM に道具を持たせると何が変わるか」(仮) の材料を集める実験。
Tsugumi のコンセプト（ローカル LLM ＋ オフライン Wikipedia ／ Web 検索、コーディング以外の調べ物）の検討用。

## 1. 問い

1. モデル本体の知識だけ、オフライン Wikipedia、オンライン検索で、答えはどう変わるか。どの種類の質問で差が出て、どれで出ないか。
2. 検索して読んでまた調べるエージェント動作は、検索 1 回で答える固定手順と比べてどれだけ効くか。
3. 道具を持たせても埋まらない差（質問の受け取り方、何を拾うか）は、モデルの大きさでどれだけ変わるか（Gemma-4 26B と Qwen3.8-Flash-Next）。

待ち時間は比べない。リモートの GPU で回すので Mac の時間と対応せず、論点でもないため。

## 2. 条件

2 モデル × 5 条件 × 24 問、各 1 回（Serper の節約のため繰り返さない）。thinking は全条件オフ（普段の使い方）。

| 条件 | ネットワーク | ツールのラウンド上限 | 中身 |
| --- | --- | --- | --- |
| `model` | モデルのみ | — | ツールもツール用システムプロンプトも無し |
| `offline-fixed` | Offline | 1 | 固有名詞の事前引き＋索引を 1 回引いて回答 |
| `offline` | Offline | 6（既定） | Wikipedia を何度でも引き直す |
| `online-fixed` | Online | 2 | 強制検索 1 回 → 強制ページ読み 1 回 → 回答（よくある RAG の形） |
| `online` | Online | 6（既定） | 強制検索・強制ページ読みの後、自分で検索し直す |

- 問い 1 は `model` / `offline` / `online` の比較、問い 2 は `*-fixed` と上限 6 の比較で答える。
- Online で先に検索とページ読みを強制するのはアプリと同じ方針（`AppModel.onlineToolChoice`、docs/WEB_SEARCH.md §2）。
- ラウンドの数え方: ツール呼び出しで終わった生成を 1 ラウンドと数える。上限に達すると `tool_choice: none` のラウンドで回答させる。
  アプリ自身が先に引く分（質問中の URL、固有名詞の Wikipedia）は数えない。

## 3. 環境

- 推論: llm-server 192.168.1.9 の llama-swap（`http://192.168.1.9:8080`）。GPU は排他で、モデルの切り替えに約 3 分かかる。
  → Gemma の全条件を流してから Qwen に切り替える。

  | モデル | llama-swap の ID | 中身 | スロット |
  | --- | --- | --- | --- |
  | Gemma-4 26B | `gemma4-26b-a4b-mtp-1` | Q4_0、MTP draft 1 | 3 × 32K |
  | Qwen3.8-Flash-Next | `qwen3.8-flash-next-iq3-instruct` | UD-IQ3_XXS、MTP draft 2、KV q8_0、`--jinja --reasoning off` | 1 × 98K |

- 文脈長: 全条件でアプリの既定 32,768 に揃える（`run.sh` が `--context 32768` を渡す）。Qwen はスロットが 98K あっても 32K で打ち切る。
- サンプラ: リクエストで送るアプリの値が効く（Gemma は temp 1.0 / top-k 64 / top-p 0.95、Qwen は temp 0.7 / top-k 20 / top-p 0.8）。
  アプリが送らない presence penalty はサーバの既定値になる（Qwen は 1.5、Gemma は指定なし）。
- ハーネス: このMacの `TsugumiToolLoopCheck`。システムプロンプト、ツール宣言、Online の強制、ラウンド上限、
  ツールの実行（Serper → Brave、ページ読み、`~/Library/Application Support/Tsugumi/wikipedia-ja.sqlite`（2026-08-30 版））、
  ペルソナ（`persona.json`）はアプリのものがそのまま走る。推論だけがリモートになる。
- Web の記録: `--web-store` で検索結果とページをモデル × 条件ごとに保存する。再実行では記録を再生し、Serper を叩き直さない。
  採点のときは、モデルが実際に読んだ中身をここから確認できる。

### 3.1 ハーネスの改修（2026-09-16）

- `Sources/TsugumiToolLoopCheck/RemoteInferenceClient.swift`
  - `--endpoint URL --remote-model ID` で使う。`--model` はモデル種別（プロンプト、サンプラ、呼び出しの書式）を決めるだけで、読み込まない。
  - 文脈長は、指定がなければサーバのスロットの `n_ctx` に合わせる。`--context` がスロットより大きければ警告する。
  - `live` と `progress` の判定はこの Mac のセッション用なので、リモートでは外す。
- `--max-rounds N`（保存済み設定を書き換えずにラウンド上限を変える）と `--thinking on|off` を追加。
- llama-server（b10825）は `tool_choice` の強制を無視する。`required` も関数指定も、ただの本文で返ってきた（2026-09-16 に curl で確認）。
  そのため、ラウンドの形を 3 通りにした。
  - `auto`: `/v1/chat/completions` にストリーミングで送る。テンプレートの描画と呼び出しの解析は llama-server が行う。
  - `function` / `required`（強制するラウンド）:
    1. `/upstream/<model>/apply-template` で同じメッセージを描く。
    2. 呼び出しの冒頭（Gemma なら `<|tool_call>call:web_search{`）を足し、`/v1/completions` で閉じ記号まで続けさせる。
    3. アプリのパーサ（`GemmaToolCallParser` / `QwenToolCallParser`）で読む。読めなければ `structured_output_failure` にする。
       アプリはこれを 1 回だけやり直す。
  - `none`: 呼び出しの開始トークンを `logit_bias` で禁止する。宣言はプロンプトに残す（アプリの `ForbiddenTokensConstraint` と同じ形）。
- `AppTokenEvent` に public init を追加。`Package.swift` で TsugumiToolLoopCheck が `Tsugumi` に依存するよう変更。
- `docs/experiments/knowledge-sources/run.sh`: モデルと条件を指定して流す。結果は `scratch/knowledge-sources/<model>/<condition>/`。

**アプリとの違い**（記事に一文で断る）
- テンプレートは llama.cpp の Jinja 描画。
- 自由なラウンドで呼び出しを縛る文法が無い。
- 強制ラウンドは前置きのあと、文法なしで書かせている。重複キー（`query` を 2 回書く）を一度観測した。パーサは後の値を取る。

### 3.2 動作確認（2026-09-16、gemma4-26b-a4b-mtp-1）

「城崎マリンワールドの見どころと、開業した年を教えて」で確認した（web-store は `scratch/knowledge-sources/web-store-smoke`）。
- 5 つの条件の形がすべて通った。
  - `model`: 1 ラウンド
  - `offline`: `wikipedia_page` → `wikipedia_search` × 2 → 回答
  - `online`: 強制 `web_search` → 強制 `fetch_page` → 回答
  - 上限 1: 呼び出し → `none` で回答
  - 上限 2: 検索 → ページ → `none` で回答
- `model` は「2023年3月リニューアル」「海中トンネル」をでっちあげた。
- `offline` は「開業年の記載は無い」と正直に答えた。
- `online` は 1934 年創業と 2025-07-18 のリニューアルを出典つきで答えた。
- 仮説どおりの差がすでに出ている。
- ハーネスの判定: `online-fixed` 以外は全合格。`online-fixed` はページを読まない上限 1 のとき `online` 判定で落ちる。これは仕様どおりで、上限 2 なら合格する。

### 3.3 動作確認（2026-09-16、qwen3.8-flash-next-iq3-instruct）

§3.2 と同じ質問で、5 条件を直列に流した。ロードは 2 分 9 秒。
- **形**: 全条件が合格した。
  - 呼び出しの開始 `<tool_call>` は 1 トークン（248058）。
  - 強制ラウンド（`<tool_call>\n<function=web_search>\n` を前置き）は、`QwenToolCallParser` で読めた。
  - `none` のラウンドで呼び出しは出なかった。
- **ツールの使い方**: Gemma と違いが出た。
  - `offline` は 6 ラウンドを使い切った。`wikipedia_page` と `wikipedia_search` を 5 回引き直し、それでも開業年は見つからないと答えた。推測の年は書かなかった。
  - `online` は強制の検索とページ読みのあと、自分から節の読み直し、会社の沿革ページ、リニューアルの再検索、PR TIMES まで読んだ。
    答えは「1934 年創業／1994 年に現名称」「2025-07-18 リニューアル」。
  - `model` は「1968 年開業」「2015 年リニューアル」「海中展望塔」をでっちあげた。
  - `online-fixed` は、読んでいない「ジオ・アクア」を足した。盛りに見えるが未確認。

## 4. 本番前に決める・直すこと（Phase 0）

- [x] **スロットの文脈長を 32K に。** Gemma（`gemma4-26b-a4b-mtp-1`）は `-np 3` で上げ直し、1 スロット 32,768 × 3 を確認した（2026-09-16）。
      Tsugumi の既定と同じ 32K。条件を 3 本まで並列に流せる。
- [x] **Qwen の文脈長。** `qwen3.8-flash-next-iq3-instruct` は `-c 98304 -np 1` で 1 スロット 98K。32K に足りる。
      スロットが 1 つなので条件は直列に流す。
- [x] **Qwen の thinking がオフになるか。** instruct 版はサーバが `--reasoning off`。
      `apply-template` の描画は `<think>\n\n</think>` で閉じていて、回答に思考は出なかった（2026-09-16）。
- [x] **Qwen の動作確認。** §3.3 のとおり、5 条件とも通った。
- [ ] **質問セットの確定。** `questions.json` は叩き台（§5）。
- [ ] Serper の残り枠を確認する。見込みは §8。

## 5. 質問セット（`questions.json`、叩き台）

8 カテゴリ × 3 問。半分以上は `chats.json` の実際の質問から取った（`note` に chat 番号）。各問 1 ターン。

| カテゴリ | 仮説 | 問 |
| --- | --- | --- |
| A 調べ物が要らない（創作・計算・定番） | 3 通りで差なし。検索で中央値化・悪化もありうる | 冷蔵庫レシピ、寿司ネタ俳句、三次方程式 |
| B 百科事典的なロングテール | 本体だけはでっちあげ、Wikipedia で大きく改善 | 城崎シーワールド（名前違い）、モンスーンジャイア、どんぐり共和国 |
| C 学習後の出来事 | 本体だけは不可。8/30 版 Wikipedia がどこまで拾うか | 2026年1月の日本、ベネズエラ、Granite 4.2 |
| D 刻々と変わること | Online だけが答えられる。Offline で「分からない」と言えるか | 明日の京都の天気、M6 Mac mini の SSD、今週の生成 AI ニュース |
| E 意図読み・愚痴・相談 | 知識源よりモデル差が出る | えきねっと、新横浜、Amazon 10 個パック |
| F 実例の収集 | 固定手順ではほぼ無理。エージェント動作が必須 | 書き出し小説 50 件、Maker Faire Tokyo 2026、JR 廃止路線 10 |
| G 多段の調べ物 | 上限 1〜2 と 6 の差が最大 | 芥川賞→出身地→名物、東京駅の乗り換え、最古と最新の水族館 |
| H 誤った前提 | 本体だけは話を合わせる。検索で訂正できるか | 淀城の遺構、iPhone 17 の Lightning、東西線の廃止 |

時事ものの注意:
- D、C1、F2 は実行日で答えが変わる。
- 1 モデルの全条件は同じ日に流し、2 モデルもなるべく続けて流す。
- 「明日」「今週」の基準日は `turns.jsonl` の実行時刻で記録する。

## 6. 手順

### Phase 1 正解メモ（実行前）
- 24 問それぞれについて、確認すべき事実、ありがちな誤り、誤前提の正体、「良い答え」の条件を 3〜5 行でまとめる。
  出力: `answer-notes.md`。
- Claude が Web で下調べし、あなたが確認する。
- 答えを見る前に作り、採点が回答に引っ張られないようにする。
- D（天気・ニュース）は実行日の事実なので、実行直後に記録する。

### Phase 2 Gemma 本番
```sh
cd ~/LLM/turbo-fieldfare
# スロット 3 つなので 3 本並列
docs/experiments/knowledge-sources/run.sh gemma model offline-fixed &
docs/experiments/knowledge-sources/run.sh gemma offline online-fixed &
docs/experiments/knowledge-sources/run.sh gemma online &
wait
```
- 失敗した問（error、文脈超過、空の回答）は `ONLY=… FORCE=…` で再実行する。web-store が再生するので Serper は増えない。
- 再実行の回数は記録する。「たまたま良い回答が出るまで回す」ことはしない。

### Phase 3 Qwen 本番
- llama-swap で Qwen（instruct 版）に切り替え（約 2〜3 分）、直列に流す。
```sh
docs/experiments/knowledge-sources/run.sh qwen model offline-fixed offline online-fixed online
```

### Phase 4 採点
- **盲検用に書き出す**
  - `turns.jsonl` から、問ごとに 10 回答（2 モデル × 5 条件）を並べ替えて ID を振り、条件名を伏せたシートを作る。
  - スクリプト `export_blind.py` は未作成。
- **採点の軸**（各 0〜2 点）
  1. 正確さ: 正解メモの事実と合っているか。誤りやでっちあげは減点する
  2. 根拠: 出典にない盛りが無いか。`model` 条件は出典が無いので、断定の誤りだけを見る
  3. 有用さ: 質問の意図に応えているか。誰にでも言える中央値の答えは 1 点
- **分担**: Claude が正解メモに照らして一次採点し、理由を 1 行添える。記事に載せる回答と、点の割れた回答はあなたが判断する。

### Phase 5 集計
- **得点**: カテゴリ（8）× 条件（10）の平均点ヒートマップ。記事の中心の図にする。
- **動作の指標**（`rounds.jsonl` / `turns.jsonl`、時間は使わない）
  - ラウンド数、`web_search` / `fetch_page` / `wikipedia_*` の回数、読んだ字数
  - 読んだあとに検索し直した回数（`fetch_page` の後に `web_search` が来た数）
  - 上限 6 で実際に何ラウンド使ったか（エージェント動作が必要と判断した割合）
- **エージェントの効果**: `*-fixed` と上限 6 の得点差をカテゴリ別に出す。F・G で差、A〜D で差なしが仮説。
- **失敗の類型**: 検索しても本体の知識で上書きした、誤前提に乗った、実在の例を探さず自作した、Offline で分からないと言えた／言えなかった。

### Phase 6 記事
記事は結論と図が先。数字は「どの質問に、どの道具が効くか」に答えるものだけを載せる。
1. 結論（知識源が効く質問、エージェント動作が効く質問、モデルの賢さしか効かない質問）
2. ヒートマップ 1 枚
3. カテゴリ別の代表例（同じ質問に `model` / `offline` / `online` の回答を並べる）
4. 固定手順とエージェントの比較（F・G の例）
5. Gemma と Qwen の差は道具で埋まるか
6. Tsugumi への示唆（オフライン Wikipedia の実用性、thinking オフで道具を持たせる判断）

## 7. 出力の置き場所

| もの | 場所 |
| --- | --- |
| 質問セット、実行スクリプト、この計画 | `docs/experiments/knowledge-sources/` |
| 実行結果（turns / rounds / run.log） | `scratch/knowledge-sources/<model>/<condition>/`（git 管理外） |
| Web の記録 | `scratch/knowledge-sources/web-store/<model>-<condition>/` |
| 正解メモ、採点シート、集計、図 | `docs/experiments/knowledge-sources/`（Phase 1・4・5 で作る） |

## 8. Serper の見込み

- 1 問あたりの検索回数（`online-fixed` は 1 回、`online` は強制 1 回＋自発的に 0〜3 回）で見積もると、
  24 問 × 2 モデル × (1 + 約 3) ≒ **200 クエリ前後**。
- ページ読み（自前の fetch、薄いときだけ Jina）は Serper を使わない。
- Offline と `model` は外に出ない。
- 再実行は web-store から再生するので増えない。ただしモデルが別のクエリを書けば、新しく記録される。
