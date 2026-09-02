# ローカル Wikipedia — 日本語版をこの Mac に置いて引く

最終更新: 2026-09-02。対象は `TsugumiMac` のチャット (Gemma のみ、Web 検索と同じ
ツールループ)。Web 検索は「無料枠の外部 API」だが、こちらはネットに出ない。
ローカル LLM の隣にローカルの百科事典を置く、という思想の側の機能で、実用の
細部 (取り込みの手間、容量) は二の次にしている。

## 1. 何ができるか

Inspector の Web search セクションに **Local Wikipedia index** の欄がある。ここに
組んだ SQLite (§3) のパスを入れると、Web search のモード (Auto / Always) で
Gemma に 2 つのツールが増える。API キーが無くても、この 2 つだけで動く。

| ツール | 引数 | 返すもの |
| --- | --- | --- |
| `wikipedia_search` | `query` | 記事の題名と導入部の一行 (既定 8 件)。全文検索 |
| `wikipedia_page` | `title`, `from` (任意) | 記事本文。Page text limit (既定 6,000 字) で打ち切り、`from` で続きを読む |

Web 検索のキーもあれば 4 つ全部を宣言し、system prompt に使い分けの一文が入る
(百科事典にある事柄は Wikipedia から、刻々と変わることは Web から、Wikipedia で
見つからなければ Web へ)。Wikipedia だけのときは「天気や株価は無いので、その旨を
答える」に変わる。プロンプトの穴埋めは `Resources/search-tool-prompts.json`。

回答の末尾の「参照:」には Wikipedia の記事名が並ぶ。

## 2. データ源 — なぜ Kiwix (ZIM) ではないか

Wikimedia が毎週出している検索用ダンプ
(`dumps.wikimedia.org/other/cirrus_search_index/<日付>/index_name=jawiki_content/`)
を使う。1 記事が JSON 1 行で、`text` は整形済みの平文、`opening_text` は導入部、
`redirect` は転送元の名前、`incoming_links` は被リンク数。14 分割・計 9 GB (bz2)。

Kiwix の `wikipedia_ja_all_nopic` (15 GB) も検討した。落として終わりなのは
魅力だが、

- 鮮度が四半期単位 (2026-06 版) で、週次ダンプ (2026-08-30 版) より 3 か月古い。
  「最近の出来事を引かせたい」という動機に合わない。
- ZIM を読むには libzim (C++、Xapian・zstd・xz) が要り、.app に dylib を束ねて
  公証する工程が増える。SQLite なら macOS 標準で、Swift は `import SQLite3` だけ。
- 中身が HTML で、抽出を挟む。順位付けの材料 (被リンク数) も無い。
- 日本語の 2 文字語 (米国・戦争・首相) を確実に当てるには索引を自前で切りたい。

## 3. 組み方

```sh
# 1. ダンプを取る (~/Library/Caches/Tsugumi/jawiki-<日付>/ に 14 本、再開可)
python3 Scripts/wiki/build_jawiki_index.py download

# 2. 組む (4 プロセス、30〜40 分見込み。--limit N で試し組み)
python3 Scripts/wiki/build_jawiki_index.py build \
    --out ~/Library/Application\ Support/Tsugumi/wikipedia-ja.sqlite \
    ~/Library/Caches/Tsugumi/jawiki-20260830/*.json.bz2

# 3. 確かめる (アプリと同じ順位付け)
python3 Scripts/wiki/build_jawiki_index.py search ~/Library/Application\ Support/Tsugumi/wikipedia-ja.sqlite "東京タワー"
python3 Scripts/wiki/build_jawiki_index.py page   ~/Library/Application\ Support/Tsugumi/wikipedia-ja.sqlite "米国"
```

標準ライブラリだけで動く。Inspector の欄にパスを入れると、開けたかどうかと
記事数・ダンプの日付がその下に出る。環境変数 `TSUGUMI_WIKIPEDIA_INDEX` でも指定
できる (ファイルより優先)。ダンプは取り込んだら消してよい。

### 中身

| テーブル | 何が入るか |
| --- | --- |
| `meta` | スキーマ版、ダンプの日付、記事数、字句規則の版と**標本の字句列** (§4) |
| `pages` | id、題名、導入部、本文 (raw deflate)、被リンク数、更新日時 |
| `titles` | 正規化した題名と転送元 → id。`wikipedia_page` の引き当て |
| `search` | FTS5 (contentless、unicode61)。列は題名・別名・導入部・本文の頭 (既定 1,000 字) |

転送ページ (`page_type: redirect`) は記事としては入れず、名前だけ `titles` と
`search.aliases` に入る。「米国」で `アメリカ合衆国` が出るのはこのため。

2 万記事の試し組みで 0.14 GB (本文 73 MB、索引 60 MB 程度)。全体 (約 145 万記事)
は 10 GB 前後の見込み。`--body-chars` を増やすと本文の深い所まで当たるが索引が
膨らむ。

## 4. 検索の仕組み

FTS5 の標準の字句解析は日本語を切れず、内蔵の trigram は 2 文字語に当たらない。
そこで**組む側で先に切って**空白区切りにし、FTS5 には unicode61 でそのまま
食わせる。規則 (`WikipediaTokenizer` / スクリプトの `tokenize()`):

- NFKC → 小文字。
- CJK (かな・漢字・ハングル・ー・々) と数字の連なりは**文字バイグラム**
  (「東京タワー」→ 東京 京タ タワ ワー)。1 文字だけならその 1 文字。
- それ以外の英数の連なりは 1 語 (「iphone」)。英字と数字の境目は切る (「M4」→ m 4)。
- 残り (句読点、中黒「・」) は区切り。

問い合わせは空白で語に分け、各語をバイグラムの**句** (phrase) にする。全語 AND →
いずれかの語 OR → いずれかのバイグラム OR の順に緩め、最初に当たった段を使う。
題名か転送名が問い合わせ全体にそのまま一致する記事は、点数に関わらず先頭に置く
(「米国」→ アメリカ合衆国。転送名は別名の列にも入っているが、転送名が何百もある
記事は bm25 の長さ正規化で沈むので、この段が要る)。残りは bm25 (題名 10・別名 6・
導入部 2・本文 1) の上位 60 件を、被リンク数で軽く補正 (`rank − 0.5·ln(1+links)`、
加算) して並べ直す。乗算にすると、被リンク数がまだ 0 の今年の記事
(「2026年イスラエルとアメリカ合衆国によるイラン攻撃」) が、古い記事の中の一言に
負けた。

**組む側と読む側の規則が食い違うと、エラーではなく空振りになる。** それを
防ぐため、組む側は標本文 (`TOKENIZER_CHECK_SAMPLE`) を切った結果を `meta` に
書き、読む側は開くときに自分で切った結果と突き合わせる。違えば
`tokenizerMismatch` で開かない。規則を変えるときは両方を変えて版を上げる。

## 5. 実機で確かめたこと (2026-09-02, M3 Pro, Gemma 4 QAT, ctx 8K, thinking ON)

2 万記事の標本索引に対して、アプリと同じ経路 (DecodeService の unix socket) で
1 周:

```sh
TSUGUMI_WIKIPEDIA_INDEX=/path/to/sample.sqlite \
python3 Scripts/app/smoke_decode.py wiki "アンペルマンって何? 誰がデザインしたの?"
```

| ラウンド | 結果 |
| --- | --- |
| 1 (required, 思考なし) | `wikipedia_search {"query":"アンペルマン"}`、4.1 s |
| 2 (auto, 思考あり) | `wikipedia_page {"title":"アンペルマン"}`、生成 22 トークン |
| 3 (auto, 思考あり) | 本文に基づく回答 (カール・ペグラウ、1961 年…)、末尾に「参照: アンペルマン」、142 トークン |

全体索引 (1,515,165 記事、10.06 GB、構築 30 分・4 プロセス) では、学習データより
後の出来事を聞いた:

```sh
python3 Scripts/app/smoke_decode.py wiki "2026年にアメリカがイランを攻撃したって本当? 何が起きたのか教えて"
```

| ラウンド | 結果 |
| --- | --- |
| 1 (required, 思考なし) | `wikipedia_search {"query":"アメリカ イラン 攻撃 2026"}` → 8 件、1 位が「2026年イスラエルとアメリカ合衆国によるイラン攻撃」 |
| 2 (auto, 思考あり) | 結果を読んだ思考 1,091 字 (英語、内容の整理) → `wikipedia_page` で本記事を開く |
| 3 (auto, 思考あり) | 6,000 字の本文から「はい、本当です。2026年2月28日に…」と経緯を節立てで回答、945 トークン・57 s |

思考チャネルは結果の整理に使われていて、日付を疑う堂々巡り (WEB_SEARCH.md §6)
は出ていない。検索 1 回は Python 起動込みで 0.05〜0.3 秒 (「日本」のような
当たりの多い語で 0.3 秒)。
Wikipedia のツールは「この Mac に保存された {日付} 時点の複製」と説明していて、
「本物のインターネットに 2026 年は無いはず」という反射の足場が無い。

## 6. 分かっている制約

- **Gemma のみ**、Web 検索と同じ (Ornith はツールを宣言しない)。
- **本文の頭 1,000 字までしか索引に入らない。** 長い記事の後ろの節だけに書かれた
  語では出ない。`--body-chars` で広げられる。
- **見出しは残らない。** ダンプの `text` は節見出しを含まないので、`wikipedia_page`
  は平文を頭から返す。`from` で読み進めるしかない。
- **順位付けは素朴。** 語の重みも被リンク補正も手で決めた値で、評価はしていない。
- **取り込みは手作業** (スクリプト実行)。アプリからの取得・更新は未実装。
- 索引ファイルは Web 検索の設定ファイル (`web-search.json`) にパスだけ入る。
  移動したら Inspector で直す。

## 7. 主な実装箇所

| 場所 | 役割 |
| --- | --- |
| `Scripts/wiki/build_jawiki_index.py` | download / build / search / page / tokenize |
| `Sources/TsugumiApp/Core/LocalWikipedia/WikipediaTokenizer.swift` | 字句規則、MATCH 式、題名の正規化 |
| `Sources/TsugumiApp/Core/LocalWikipedia/LocalWikipediaIndex.swift` | SQLite の読み (検索・記事・meta の検査・deflate 展開) |
| `Sources/TsugumiApp/Core/LocalWikipedia/WikipediaToolExecutor.swift` | 2 つのツールの宣言と結果の文面 |
| `Sources/TsugumiApp/Core/LocalWikipedia/CompositeToolExecutor.swift` | Web と Wikipedia を 1 つの宣言にまとめる |
| `Sources/TsugumiApp/Core/Resources/search-tool-prompts.json` | system prompt の穴埋め (web / wikipedia / both) |
| `AppModel.makeToolExecutor` | キーと索引から実行器を組む。どちらも無ければエラー |
| `Tests/TsugumiApp/Core/LocalWikipedia/`, `Tests/TsugumiApp/Core/Fixtures/wikipedia-fixture.*` | 5 記事のフィクスチャ (スクリプトで組んだもの) を Swift で読む |
