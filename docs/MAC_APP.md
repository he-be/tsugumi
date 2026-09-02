# Mac アプリ — 二モデル対応の設計と運用

最終更新: 2026-09-01。対象は `TsugumiMac` (SwiftUI) と、その推論プロセス
`TsugumiDecodeService`。**このリポジトリで積んだ改善を規定値として GUI から
使えるようにした**改造の記録と、ウェイト供給の手順。

## 1. なにが入ったか

| 項目 | Gemma 4 26B-A4B QAT | Ornith-1.5 35B-A3B oQ4e-g64 |
| --- | --- | --- |
| ウェイト | `sym` repack + vision + drafter (`scratch/gemma4-qat-sym.moepack`, 15.7 GB) | q_norm bake + shisa MTP ヘッド (`scratch/ornith-oq4e-g64.moepack` 19.6 GB + sidecar 503 MB) |
| Vision | あり (画像添付 UI、最大 4 枚) | なし |
| MTP | 既定 ON、ブロック 4 | 既定 ON、幅 2 固定 ([docs/qwen35moe/36](qwen35moe/36-MTP-DECODE.md)) |
| Thinking | トグルあり、**既定 OFF** | トグルあり、**既定 ON** |
| サンプラ | 公式値を既定に編集可 (temp 1.0 / top-k 64 / top-p 0.95) | **公式値固定** temp 0.6 / top-k 20 / top-p 0.95 (S1, [docs/qwen35moe/42](qwen35moe/42-SAMPLING.md)) |
| prompt cache | LCP (サーバと同じ `ServerPromptCache`) | 厳密な延長のみ ([docs/qwen35moe/41](qwen35moe/41-PROMPT-CACHE.md)) |
| コンテキスト | 4K〜**128K** (既定 32K) | 同左。128K は wired limit を上げないと decode が半減しうる ([SERVER_RUNBOOK](SERVER_RUNBOOK.md)) |

実現方法: GUI の推論経路を、HTTP サーバが使っているファミリセッション
(`ServerModelSession` / `QwenServerSession`) にそのまま乗せ換えた。GUI 固有の
生プロンプト経路は消え、chat template・思考チャネル・投機デコード・prompt cache
はサーバと同じコードが答える。モデルの判別は `manifest.json` の `arch.family`
(無ければ Gemma) で、`AppModelKind` が能力と既定値を一元管理する。

**マルチターン** (2026-09-01): 1 プロンプト 1 回答から、単一チャットの
マルチターン会話にした。完了したターンは次の送信時に履歴
(`AppModel.conversationTurns`) へ畳まれ、要求には履歴ごと載る。assistant ターンは
`reasoning_content` 付きで描き直される (SPEC MSG-5 / INV-1) ので、**thinking の
ON/OFF に関わらず 2 ターン目以降の prompt cache は当たる** — Gemma は LCP、
Ornith は厳密な延長がそのまま成立する。答えが 1 文字も出ずに失敗したターンは
履歴に残さない (モデルが見ていない会話を描かない)。会話のリセットは従来どおり
コンテキストメニューの Clear。ターン間の履歴は `TsugumiDecodeService`
へのワイヤ (`DecodeGenerationRequest.history`) を通る。

**複数チャット** (2026-09-01): 会話ごとの状態 (下書き・履歴・ライブ出力) を
`AppChatSession` に切り出し、`AppModel` は `chats` + `selectedChatID` +
生成中チャットの参照を持つ。左サイドバーで一覧・切替・新規作成・削除
(コンテキストメニュー)。**同時生成は 1 本だけ** (バックグラウンド生成なし):
生成イベントは選択中ではなく生成を始めたチャットに書き込まれるので、生成中に
別チャットへ切り替えても流れ込みは正しいチャットに続く。その間、他のチャットは
リードオンリー (コンポーザの代わりに案内を表示、生成中チャットは削除も不可)。
transcript mailbox は 1 本のクライアント所有チャネルなので所有チャット ID で
ゲートし、他チャットの画面が生成中の本文を吸わないようにしてある。

**永続化** (2026-09-01): チャット一覧・各会話の履歴・ライブ出力・下書き・選択位置を
`~/Library/Application Support/Tsugumi/chats.json` に保存する
(`AppChatStore`)。チャットはモデルに紐付かない (別モデルで続きを打てる) ので、
モデル別設定ファイルとは別の 1 枚。構造変化 (新規・削除・切替・Clear) と
ターン完了は即時保存、下書きのタイプは 1 秒デバウンス — 終了直前 1 秒未満の
打鍵だけは落ちうる。復元時は mailbox の所有を外す (空 mailbox が復元本文を
隠すため) し、消えた画像ファイルのパスは履歴から落とす (残すと以後その
チャットの検証が永遠に失敗する)。壊れたファイルは捨てて新規開始。

思考チャネルのライブ表示は**末尾 1,500 字だけ**を描く
(`ReasoningLivePresentation.liveTail`)。全文をトークン毎にレイアウトし直すと
思考フェーズ全体で O(n²) になり、長い think でメインスレッドが飽和して
トークンイベントが滞留 → 表示が実速度から遅れて這い、Stop も効かないように
見える。完了後のディスクロージャは従来どおり全文。

**Web 検索** (2026-09-02): Gemma のチャットに `web_search` / `fetch_page` の
ツールループを足した。切り替えは Offline / Online の 1 つ (Offline はローカル
Wikipedia のみ、Online で Serper → Brave と自前 fetch → 薄ければ Jina Reader)。
質問中の URL と固有名詞はモデルの 1 ラウンド目の前にアプリが引く。設計・キーの
置き場・実機スモークは [WEB_SEARCH.md](WEB_SEARCH.md)、Wikipedia は
[LOCAL_WIKIPEDIA.md](LOCAL_WIKIPEDIA.md)。

**ローカル Wikipedia** (2026-09-02): 同じツールループに `wikipedia_search` /
`wikipedia_page` を足した。Wikimedia の週次ダンプから自前で組む SQLite (FTS5、
バイグラム) をこの Mac に置き、ネットに出ずに引く。キー無しでも動く。
[LOCAL_WIKIPEDIA.md](LOCAL_WIKIPEDIA.md)。

## 2. ウェイトの供給 — 完成品の直DL

ストリーミング repack は `sym` を作れず (staging の bias レンジが要る、
[docs/mtp/44 §7](mtp/44-W1-WEIGHT-DIET.md))、Ornith は bake と MTP ヘッドの graft が
Python パイプラインにしか無い。そこでアプリのインストーラは**完成品をそのまま
Hugging Face から落とす**方式にした:

- リポジトリ: `mh73772/turbofieldfare-gemma4-qat-sym` / `mh73772/turbofieldfare-ornith-oq4e-g64`
  (レイアウトはインストール先と同一。Ornith は `mtp-head/` にサイドカーを含む)
- 整合性: ファイル単位の SHA-256 ピン (`PrebuiltFileTables.swift`、成果物から生成)。
  revision は URL の都合であって整合性の根拠ではない
- resume: `<name>.part` に途中まで書き、Range で続きから。ハッシュは part を
  読み直して合流する。検証に通ってから最終名に rename、`manifest.json` は最後
- 完了時に `verified-install.json` (trusted-install receipt) をピンから書く
- **この MBP ではダウンロードは走らない**: `scratch/` の成果物が最終名・サイズ一致で
  そのまま完成インストールとして認識される

アップロード (ユーザー作業、書き込みトークンが要る):

```sh
Scripts/app/upload_prebuilt.sh
```

ウェイトを作り直したら `PrebuiltFileTables.swift` の再生成を忘れないこと
(ファイル冒頭のコメントに手順)。

## 3. Ornith の MTP サイドカー

ヘッドは `.moepack` に入らない ([docs/qwen35moe/30 §6](qwen35moe/30-MTP-HEAD-GRAFT.md))。
エンジンは `<モデル>/mtp-head/mtp_head.json` を先に探し、無ければ
`~/LLM/ornith-mtp-head` (`TF_QWEN_MTP_HEAD` で差し替え) に落ちる。どちらも無ければ
**MTP は静かに OFF でロードする** (答えられないより遅い方がまし)。Gemma 側も同じ:
drafter セクションの無い pack では OFF に落ちる。

## 4. ロード時拘束の変更

ファミリセッションは prefill 構成と投機ループをロード時に束ねるので、
**prefill・チャンク幅・MTP は「Reload required」になる設定**に変わった
(`AppLoadedRuntimeKey`)。thinking・サンプラ・コンテキスト内の生成長は
リクエスト時のまま。

## 4b. 画面の言語 (2026-09-02)

画面の文言は英語をキーにした `Localizable.strings` で持ち、日本語訳を同じ
ファイル名で並べる (`Sources/TsugumiApp/Core/Resources/{en,ja}.lproj` が
状態ラベルとエラー文、`Sources/TsugumiApp/Mac/Resources/{en,ja}.lproj` が
画面の文言)。Mac 側は `L("…")`、Core 側は `AppLocalization.string("…")` を
通す。どちらもキーが英語なので、訳が無い文言は英語のまま出る。

言語は OS に従う (システム設定 > 一般 > 言語と地域 > アプリケーション で
アプリごとに切り替えられる)。macOS は **main bundle が持つ言語を先に選び、
リソース bundle をそれに合わせる**ので、`make_app.sh` は
`Contents/Resources/{en,ja}.lproj` を .app に置く。これが無いと日本語環境でも
英語のままになる。`swift build` した素の実行ファイルは main bundle に lproj が
無いので常に英語になる。

同時に、細かい説明文 (スロットの解説、サンプラの解説、Last run の (i) ポップ
オーバー、プロンプトのコツ) と「Try an example」のカードは外し、Generation の
サンプリング項目は畳んだ状態を既定にした。

## 4c. 回答の Markdown 描画 (2026-09-02)

`ResponseMarkdownRenderer` は swift-markdown (cmark-gfm) の AST を 1 回歩いて
`NSAttributedString` を組む。以前は Foundation の `AttributedString(markdown:)` が
返す平らな run 列から構造を推測していて、表の有無・`<…>`・画像記法のどれか 1 つで
回答全体を素のテキストに落としていた。今は落とさない: 表は `NSTextTable`、HTML は
書かれたままの文字、閉じていないフェンスもコード、画像は alt (無ければ URL) を
リンク風に描く。パース前の書き換えは `presentationSource` の 2 つだけ
(強調の両端の句読点を `**` の外へ出す、太字だけの行を段落にする)。どちらも
CommonMark と Gemma の日本語の食い違いで、実回答から見つけたもの。

回帰は `Tests/TsugumiApp/MacPresentation/MarkdownCorpusTests.swift`。
`Fixtures/markdown-corpus/*.md` はサーバで生成した Gemma の実回答で、全件について
「素に落ちない・記法が残らない」と「モデルが書いた語がすべて画面にある」
(文字保存) を確かめる。描画がおかしい回答を見つけたら `.md` を 1 枚足す。
`TSUGUMI_MARKDOWN_CORPUS_JSON=~/Library/Application\ Support/Tsugumi/chats.json`
で手元の会話履歴にも同じ検査を掛けられ、`TSUGUMI_MARKDOWN_PNG_DIR=<dir>` を足すと
1 回答 1 枚の PNG が出るので目視できる (どちらもリポジトリには入れない)。

## 5. 設定ファイル

`mac-app-settings-<モデルdir名>.json` をモデルディレクトリの隣に 1 モデル 1 枚。
二モデルでサンプラも thinking 既定も違うため、共有ファイルをやめた
(旧 `mac-app-settings.json` は読まれなくなる。作り直しで困る値は無い)。

## 6. スモークの回し方

GUI を開かずにアプリの経路 (DecodeService の unix socket) を叩ける:

```sh
swift build -c release
python3 Scripts/app/smoke_decode.py gemma [画像パス]
python3 Scripts/app/smoke_decode.py ornith
```

見るもの: `ready` のロード時間、`finished` の tok/s、`draft=accepted/proposed`
(MTP が動いた証拠)、Ornith thinking ON で `REASONING` が別チャネルに出ること、
2 発目の `cached=` (prompt cache)。

2026-09-01 の実測 (M3 Pro、trusted-install、ctx 8K):

| 走り | ロード | 結果 |
| --- | --- | --- |
| gemma-plain | 2.3 s | `2です。` draft 3/3、15.9 tok/s |
| gemma-think | 〃 | 思考 223 字が別チャネル → `56`、cached=2、29.5 tok/s (RSN-4 の締切が立つと投機は仕様どおり素通り) |
| gemma-vision (ロゴ画像) | 2.4 s | 一文で正しく説明、draft 19/36、35.5 tok/s |
| ornith-think | 2.4 s | 思考 205 字 → `2`、**draft 32/39**、19.5 tok/s |
| ornith-plain | 〃 | `56`、36.3 tok/s |
