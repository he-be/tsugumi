# 36. Q2_0 は日本語が壊れている (35 W0 の結果、打ち切り)

書いた日: 2026-09-18。表記は 01 と同じ (実測 / 導出 / 未確認)。

> **状態 (2026-09-18)**: 記録として残す。Q2_0 は以後使わない。仕切り直しは 37。§1 の Q4_K_XL の各 1 セットは 37 §3-1 の試走として数える。

35 の W0 (土俵を立てる) を回した結果。**道具の経路は無傷だったが、日本語の生成が落ちていたので、この土俵での R は進めない。**

## 0. 土俵

- llama-swap のエントリ `qwen3.8-flash-next-q2_0-32k-instruct` (ユーザーが 09-18 に用意。最初 8K、32K で上げ直し)。
  本体は ISTA-DASLab `Qwen3.8-Flash-Next-GSQ-RCO-GGUF` の Q2_0、PLE は IQ4_NL、MTP なし (35 §1)。
- 比べ先は同じサーバの `qwen3.8-flash-next-q4kxl-32k-instruct-bf16ple` (unsloth UD-Q4_K_XL、MTP なし)。GPU 排他なので入れ替えて流した。
- **接続は Tailscale の `100.121.61.11:8080`。** `192.168.0.199` は macOS のローカルネットワーク許可で Swift / Python から繋がらない
  (`Local network prohibited`、curl だけ通る)。`TsugumiToolLoopCheck --endpoint` を使うなら Tailscale 側を書く。
- PC 側の速さ・サイズ・カーネルの実測は PC の `C:\LLM\ARTICLE-q2_0-gsq-rco.md` にある (記事用のメモ)。ここでは繰り返さない。

## 1. 経路は無傷 (実測、35 §3 の W0-1〜4)

- `n_ctx` 32768。サンプラは公式値のまま (temp 0.7 / top_p 0.8 / top_k 20 / min_p 0 / presence 1.5)、thinking off。
- `<tool_call>` は 1 トークン (id 248058)、`/apply-template` が使える。→ 強制ラウンドと `none` の作りがそのまま通る。
- アプリのループ (`TsugumiToolLoopCheck`) を 2 条件で 1 セットずつ。**どちらも全問 ok、検査 (live / error / answer / online) 全通過。**

| 条件 | 問 | Q2_0 | Q4_K_XL |
| --- | ---: | --- | --- |
| Offline (ローカル Wikipedia、`questions-v2-stage2.json` の N1〜N5) | 5 | 25 ラウンド / 131 s | 23 ラウンド / 184 s |
| Online (`first-fetch/questions.json` の p01・p04・c02・h04・w01、固定検索、Serper 0 回) | 5 | 26 ラウンド / 115 s | 23 ラウンド / 175 s |

- Q2_0 のラウンドの内訳: 強制 `web_search` 5・強制 `fetch_page` 5 (**structured_output_failure 0**)、auto 31 (うち回答で終了 7)、`none` 3。
  `none` は 3 つとも回答で終わり、`<tool_call>` の書き出しは無い (27 の拘束がそのまま効いている)。
- 呼び出しの中身も壊れていない: `wikipedia_search` 10 / `wikipedia_page` 12 / `web_search` 7 / `fetch_page` 14。
- 速さ (参考値): decode は Q2_0 43〜44 tok/s、Q4_K_XL 16〜32 tok/s。prefill の中央値は Q2_0 で 433 (Offline) / 510 (Online) tok/s。
- **手数の差 (25 対 23、26 対 23) は読まない。** 1 セットずつなので、32 §5-4 の揺れと区別できない。

## 2. 日本語 (実測)

### 2-1. ツールを使わない生成 — 5 問すべてで崩れた

公式サンプラで 1 回ずつ引いた。Q4_K_XL は同じ 5 問で混入ゼロ。

| 問 | Q2_0 |
| --- | --- |
| 参勤交代を 3 見出し × 3 文 | 読点がすべて `，`。「随从」「潜在 threat」「构造形成了」 |
| 猛暑の文章を 1 文に要約 | 「电力需求が逼迫**，**…防止されました」 |
| 光合成を小学生に 3 文 | 「地球上的生命を支える」 |
| 年齢の推論 (正解 19 歳) | 答えは正解。途中に「花子**是**太郎より 3 歳年下」 |
| りんご・みかん・ぶどうの旬を 3 行 (各 20 字以内) | **果物名が消えた。**「夏から秋に収穫される品種が主流」など一般論 3 行。Q4_K_XL は「りんご：10月〜12月が旬です」 |

- 最後の 1 つは混入ではなく**指示の取りこぼし**で、字数の制約を優先して問いの対象を落としている。

### 2-2. ループの中 — 軽くなるが残る

上の 1 セット (10 回答) を数えた。URL の中は除く。

| | 中国語の語・簡体字 | 中国語コンマ | 地の文の英単語 | 出た回答 |
| --- | ---: | ---: | --- | ---: |
| Q2_0 | 6 (参观・模型・能源・因素・时段・有ります) | 1 | 5 (taken / temporary / finance / Insurance / published) | 10 中 6 |
| Q4_K_XL | 5 (进入 1・成年人 4) | 0 | 0 | 10 中 2 |

- **Q4_K_XL でも出る。**「Q4 なら皆無」ではない。差は頻度と、地の文に英単語が入るかどうか。
- 事実の正しさは両方とも概ね合っている (N1 高市早苗、N3 2025-12-31 廃止・本則 28.7 円/L、N4 2026-03-22 閉館、N5 18 歳以上は 10 年のみ。
  `answer-notes-v2.md` と照合)。
- 分かれた 1 件: h04 (経団連の会長) で Q2_0 が前任を「奥田碩」と書いた (正しくは十倉雅和)。Q4_K_XL は正しい。**1 回ずつなので、これは数えない。**

## 3. 読み (導出)

- **道具の土俵としては立っている。** 強制ラウンド・`none` の拘束・呼び出しの記法が全部素直に動き、decode は Q4_K_XL より速い。
- **回答文を見る用途には使えない。** 2-1 の崩れは腕 (ツールの形) の差ではなく土俵の差として入る。
  35 §7 の「受け入れる」欄に、本体の型・PLE 表と並べて**生成日本語の質**を足す必要がある。足したうえで R を回しても、
  Mac (2-1 のような崩れが無い側) への転移は今より弱くなる。
- よって **35 の R をこの土俵で進めるのはやめる。** W1 以降は回していない。

## 4. 35 への影響

- 35 §2 の段取り (W0 → W1 → R1 → M) は W0 で止める。W1 の 15 問 (33 §4 の Offline 追加分) は作っていない。
- R (往復を減らす) を測り直す土俵は、`qwen3.8-flash-next-q4kxl-32k-instruct-bf16ple` に戻す (32 §4-1・§5 と同じ土俵)。
  こちらは本体が Mac と違う点は変わらないが、少なくとも日本語の崩れは持ち込まない。
- K (33 §5) は元から Q2_0 を使わない計画なので影響なし。

## 5. 記録の在処

- 走行の記録 (`rounds.jsonl` / `turns.jsonl` / `run.log`) はセッションの一時ディレクトリに置き、リポジトリには入れていない。
- 再現の形:

```
.build/release/TsugumiToolLoopCheck --out DIR --model ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2 \
  --conversations docs/experiments/knowledge-sources/questions-v2-stage2.json \
  --network offline --max-rounds 6 --thinking off --context 32768 \
  --endpoint http://100.121.61.11:8080 --remote-model qwen3.8-flash-next-q2_0-32k-instruct

.build/release/TsugumiToolLoopCheck --out DIR --model ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2 \
  --conversations docs/experiments/first-fetch/questions.json --only p01,p04,c02,h04,w01 \
  --network online --max-rounds 6 --thinking off --context 32768 \
  --web-store STORE --pin-search --search-budget 0 \
  --endpoint http://100.121.61.11:8080 --remote-model qwen3.8-flash-next-q2_0-32k-instruct
```

- 固定した検索結果 (40 問ぶん) は 32 §5 のものをそのまま使った。Serper は 0 回 (実測)。

## 6. 未確認

- 2-1 の崩れが Q2_0 の型のせいか、PLE 表が IQ4_NL であることのせいか (Mac の既定は BF16、31)。分けていない。
- MTP を足したとき (PC 側は unsloth の MTP と組める) に崩れが動くか。受理率が低いので組んでいない。
- h04 の事実誤りが Q2_0 の性質か、1 回の引きの揺れか。反復していない。
