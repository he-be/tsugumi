# 48. Online: 検索したページを Gemma 4 E4B が抜き出し、QFN は抜き出しだけを読む

書いた日: 2026-09-19。表記は 01 と同じ (実測 / 導出 / 未確認)。記録は `scratch/qwen38/extract48/`・`scratch/gemma4e4b/bench/` (git 管理外)。

この文書で使う言葉。

- **E4B**: Gemma 4 E4B (`google/gemma-4-E4B-it-qat-q4_0-unquantized`) と MTP 用のアシスタント (`…-unquantized-assistant`)。prefill の速い軽いモデルとして使う。Mac の llama.cpp (fe8156f78) で動かす。
- **抜き出し**: E4B が、質問と QFN の `focus` に関係する部分をページから写したもの。
- **focus**: QFN が `web_search` / `fetch_page` の引数に書く「ページから拾ってほしい点」。
- **格子**: QAT (q4_0) の重みの形。32 個ごとに d > 0 と整数 k ∈ [-8, 7] があり、原本の bf16 の値が x == bf16(d·k) になる。

## 0. 結論

1. **E4B の重みは、原本の全テンソルが格子に乗る** (実測、345 テンソル・2 億 3,320 万群、外れ 0)。群の端は -8d と ±7d の両方がある。d を fp16 で持つと 3.0% の群・1.04% の要素が bf16 で 1 ulp ずれる。16 ビットを「符号なし・テンソルごとの基準からの指数 4 ビット + 仮数 12 ビット」に割れば全群がビット一致で戻る (テンソル内の d の指数の幅は最大 9)。llama.cpp 用には fp16 の d で Q4_0 の GGUF を作った (テキスト側 4.22 GB、全テンソル 4.5 bpw)。
2. **E4B の速さは Mac の llama.cpp で天井に近い** (実測 + 導出)。prefill 605 / 596 / 563 tok/s (512 / 2048 / 8192)、decode 40.9 tok/s (MTP なし)、MTP 込みで 45〜58 tok/s。prefill は fp16 の理論値の約 7 割、decode は帯域の床の 7〜8 割 (導出) なので、E4B そのものを速くする方向は止めた。
3. **E4B は使うたびに止める** (実測、§3)。立てたままだと QFN の要求 (1,577 トークン) が 25.0〜25.6 s → 27.5〜29.9 s に遅れ、既定の設定ではスワップも出る。毎回止めれば QFN は遅れず、スワップも出ない。
4. **E4B の立ち上げは、重みの並列先読みで 3.6 → 2.85 s** (実測、QFN が重みを追い出した後、各 2 回)。重みの読みが mmap のフォールト (約 1.8 GB/s) から並列の順読み (4〜6 GB/s、4.35 GB を 0.69 s) になる。`--no-warmup` だけでは縮まない。プロセスの立ち上げに約 1.9 s 残る。
5. **Mac の 1 ターン (ja-iphone、n = 1) は 307 s (24 §3、今の形) → 420 s** (実測、§4)。立ち上げ待ちは 5 回で 18.6 s (4.4%)。増えた分の大半は E4B 自身の仕事 (prefill 78.8 s + decode 61.9 s) で、抜き出しが長い (1,024 トークンの上限で 2 回切れた)。QFN の prefill は 155.4 → 120.3 s に減ったが、`focus` を書く分ツール呼び出しの decode が 34.1 → 50.6 s に増えた。
6. **ここで止めて判断を仰いだ。** 続けるなら、E4B の出力の長さと読むページの量が次に効くところ。

## 1. 重み (格子チェックと Q4_0)

手順 (`Scripts/gemma4e4b/`):

1. 原本 (15.9 GB) を `curl -C -` で落とす。
2. `convert_text_without_ple.py`: llama.cpp の convert を、`embed_tokens_per_layer` (bf16 で 5.25 GiB) を除いて bf16 GGUF にする。除かずに流すと convert がこのテンソルを丸ごと展開し、Swapouts が約 96 万ページ (約 15 GB) 増えた (見張り無しで流した誤り。止めて作り直した)。
3. `lattice_q4_0.py IN.gguf OUT.gguf REPORT.jsonl --ple model.safetensors`: BF16 の各テンソルと、原本から直接読んだ `embed_tokens_per_layer` を群ごとに格子判定し、Q4_0 のブロック (d は原本にビット一致で戻る fp16 を近くから探す) で書く。書いた後に読み直し、bf16(d·k) と原本をビットで比べる。
4. アシスタント (23 テンソル) も同じ手順 (外れ 0、fp16 で 3% の群がずれ、仮数 12 ビットで 0)。

d の決め方: max|x|/8 と max|x|/7 の両方で k を丸めて決め、各要素の bf16 の丸め区間から d の区間 [L, H] を出す。区間が空でなければ格子に乗る。最初は「端は必ず -8d」「d は bf16」と置いて 25〜39% しか乗らなかった。

| | 値 |
| --- | --- |
| テンソル / 群 | 345 / 233,201,664 |
| 格子から外れた群 | 0 |
| fp16 の d でビット一致に戻らない群 | 7,032,163 (3.0%) |
| 同じく要素 (bf16 で 1 ulp ずれ) | 77,884,285 / 7,462,453,248 (1.04%) |
| 仮数 12 ビットの d で戻らない群 | 0 |
| テンソル内の d の 2 進指数の幅 | 最大 9 (全体で -14〜-2) |
| Q4_0 の GGUF | 4,215,695,264 バイト |

## 2. E4B の速さ (Mac、M3 Pro 18 コア GPU)

`llama-bench`、fa on、各 3 回。`-ub` 512 と 2048 で差なし。

| | 512 | 2048 | 8192 |
| --- | ---: | ---: | ---: |
| prefill (tok/s) | 605 | 596 | 563 |

decode tg128 は 40.9 tok/s。llama-server (MTP、n_max 3) で 45〜58 tok/s。ページ 3 本 (11,121 トークン) の prefill は 20.4〜22.7 s。

導出: 1 トークン約 7.6 GFLOP → 4.6 TFLOPS (理論値 約 6.4 の約 7 割)。decode は 1 トークン約 2.5 GB → 102 GB/s (この機械の帯域の床 約 135 GB/s、mtp/44)。演算ごとの内訳は取っていない。

## 3. QFN との入れ替え (実測、Chrome を閉じた状態)

`swapbench.py`: QFN は `TsugumiServer` (DS4-IQ2、32K、MTP)、要求は 1,577 トークンの日本語ページ + 64 生成 (毎回先頭を変えて prompt cache に当てない)。E4B の要求は 11,121 トークン + 512 生成。間隔 20 秒、見張り付き。

| 形 | QFN の wall | QFN の SSD 読み / 回 | Swapouts / 回 |
| --- | ---: | ---: | ---: |
| E4B を毎回止める (前 2 回・後 4 回) | 25.0〜25.6 s | 43〜46 GB | 0〜24 |
| E4B を立てたまま、既定 (常駐指定の保持 180 s・32K・`-ub 2048`) | 29.0〜29.9 s | 54〜58 GB | 2,512〜31,004 |
| E4B を立てたまま、保持 0 s・16K・`-ub 512` | 27.5〜28.9 s | 52〜54 GB | 0 (走行全体で最大 4,922) |

- QFN は要求のたびに SSD から約 45 GB を読むので、E4B に追い出されて失うものがない。E4B を立てたままにすると、E4B の重みが常駐し続ける分だけ QFN の読みが増える。
- Chrome を開いていたときも同じ順で、E4B を立てたままの既定は Swapouts +1 GiB で見張りが止めた。

立ち上げ (`loadbench.py`、QFN の要求で E4B の重みを追い出してから、各 2 回):

| 形 | 最初の要求まで |
| --- | ---: |
| 今の形 (mmap) | 3.63・3.68 s |
| `--no-warmup` | 3.56・3.65 s |
| `--no-mmap` + `--no-warmup` | 2.92・2.95 s |
| 並列 4 本の先読み (0.69 s) + `--no-warmup` | 2.85・2.86 s |

## 4. ループに入れた形と Mac の 1 ターン

`TsugumiToolLoopCheck` の `--extract-endpoint URL --extract-model ID` (`Sources/TsugumiToolLoopCheck/ExtractLoop.swift`):

- `web_search(query, focus)`・`fetch_page(url, focus)`。検索では上位 5 件を並列に取り、開けた上位 3 ページを E4B に渡す。QFN には検索結果のタイトル・URL (開かなかったものはスニペットも) と抜き出しを返す。検索の後に `fetch_page` を強制する方針は外す (`AppToolPromptFacts.searchReadsPages`)。
- E4B への頼み方: ページを先に、質問と focus を最後に。system で「ページの文のまま」「日付や条件が違っても近い記述は日付ごと」「前置き・繰り返しなし」「1,000 字以内」。サンプリングは Gemma の公式値、thinking なし、上限 1,024 トークン。最初の版 (質問を先に置き、近い記述の扱いを書かない) は、質問を写してから、日付の合わない 3 ページを「関係する記述なし」にした。
- `--extract-launch SCRIPT --extract-prefetch FILES`: ツール呼び出しごとに、重みの並列先読みと llama-server の起動をページの取得と同時に始め、抜き出しの後に止める (`ExtractorLauncher`)。
- アプリ側に足したのは差し替え口だけ: `AppToolPromptFacts.webReading` / `searchReadsPages`、`WebSearchToolExecutor.searchResponse` / `page` / `urlsInPrompt`、`HTMLTextExtractor.collapseWhitespace` の公開。既定の動きは変えていない (Web 検索まわりの単体テスト 53 件緑)。

Mac の 1 ターン (`mac-launch1`、ja-iphone、ローカルの QFN、検索は固定、ページは記録から返すので取得はほぼ 0 s):

| | 今の形 (24 §3、2026-09-15) | 抜き出し |
| --- | ---: | ---: |
| 1 ターン | 307 s | 420 s |
| ラウンド | 6 | 6 |
| QFN の prefill | 155.4 s | 120.3 s |
| QFN のツール呼び出しの decode | 34.1 s (36〜69 トークン/回) | 50.6 s (66〜102) |
| QFN の回答の decode | 114.1 s (904) | 87.3 s (779) |
| E4B の立ち上げ待ち | — | 18.6 s (2.87〜4.11 s × 5) |
| E4B の prefill | — | 78.8 s (42,643 トークン) |
| E4B の decode | — | 61.9 s (3,488 トークン、1,024 の上限で 2 回切れた) |

- 会話 1 つ・n = 1。今の形の側は 4 日前の走行で、同じ日に並べて流していない。
- QFN は同じページを focus を変えて 2 回 `fetch_page` し、E4B はそのたびにページを全部読み直した (約 5,200 トークン × 2)。
- 立ち上げ待ちはベンチ単体 (2.85 s) より約 1 s 長い。QFN が同じプロセスで動いているためとみているが確かめていない。

PC の QFN (Q4_K_XL) で 15 ターンを流した記録 (`base-a`・`base-b` は今の形、`ext-a` は抜き出し) もある。`ext-a` は PC を空けるために 13 ターンで止めたので、今の形とは並べていない。E4B の 52 回の抜き出しは、prompt の中央値 4,254、生成の中央値 144、上限で切れた 5 回。

## 5. 残ること

- 続けるなら次に効くのは、E4B の出力の長さ (上限 1,024 で切れる回がある) と読むページの量 (3 ページ × 8,000 字、同じページの読み直し)。どちらもまだ試していない。
- 自前ランタイムで E4B を動かすなら、scale を 16 ビット (指数 4 + 仮数 12) にすれば原本とビット一致で 4.5 bpw になる。カーネルは書いていない。
- 実際のネットワークでのページの取得時間は測っていない。取得が長ければ、E4B の立ち上げはその裏に隠れる。

## 出典

- 24 §3 (今の形の Mac の記録 `qwen32k-b`)、44 §2 (2 つのモデルは同時に載らない)、47 §9 (Online の確かめ方)、mtp/44 (帯域の床)
- コード: `Sources/TsugumiToolLoopCheck/ExtractLoop.swift`、`Scripts/gemma4e4b/lattice_q4_0.py`・`convert_text_without_ple.py`
- 記録: `scratch/gemma4e4b/bench/` (`b1.md`、`nochrome/*.out`、`swapbench.py`、`loadbench.py`、`prefetch.sh`)、`scratch/qwen38/extract48/` (`base-a`・`base-b`・`ext-a`・`mac-launch1`、`pc.sh`、`table.py`)、`~/LLM/gemma-4-E4B-qat/e4b-lattice.jsonl` (テンソルごとの格子チェック)
