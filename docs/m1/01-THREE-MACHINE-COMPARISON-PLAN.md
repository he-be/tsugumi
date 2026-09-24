# M1 / M3 Pro / M6 で同じ Gemma 4 26B-A4B を走らせて比べる — 記事用の測定設計

2026-09-24 起草。Apple は新チップを「M1 比で xx% 高速」と言う。手元に M1 mini、M3 Pro の MBP、M6 mini の 3 台があるので、
**同じモデル・同じバイナリの作り方・同じプロンプト**で、記事に載せられる比較を取る。この文書は測定の設計だけを持つ。
数字はまだ 1 つも無い。数字は §6 の証拠ディレクトリに置き、表は [02](02-THREE-MACHINE-RESULTS.md) (未作成) に書く。

**[TWO_MACHINE_DEV](../TWO_MACHINE_DEV.md) との関係。** 性能の記録値 (台帳) は M6 でしか取らない決まりだが、
この測定は「3 台を同じ条件で比べる」ことが目的そのものなので、ユーザーの依頼で 3 台すべてで取る。
ここで取る値は**記事用の比較値**であって、[m6-prefill/06](../m6-prefill/06-LEDGER.md) の台帳には入れない
(コマンドも既定の `--max-new 1` ではない)。証拠ディレクトリ無しの数字を書かない規則はそのまま守る (§6)。

## 1. 立場 — チップの比較ではなく、機械まるごとの比較

Apple の「xx%」は CPU / GPU の話である。ここで測るのは **SSD と RAM と OS を含めた 1 台の機械が、この 26B の MoE を
どれだけの速さで動かすか**である。3 台は RAM も SSD も OS も揃わない (§2)。揃えられないので、揃えずに「機械の値」として出す。
記事では M1 を 1.0 とした倍率を書いてよいが、**倍率はこの文書群の表には書かない**。表は絶対値 (秒、tok/s) と反復数と幅だけを持ち、
倍率は記事の側で中央値から計算する。

## 2. 機械

| | M1 mini | MBP (M3 Pro) | M6 mini |
| --- | --- | --- | --- |
| 機種 | Macmini9,1 / Apple M1 / **16 GB** | Mac15,6 / Apple M3 Pro / **18 GB** | Mac18,5 / Apple M6 / **16 GB** |
| OS | macOS 27.0 (26A428)、ヘッドレス | macOS 15.7.5 (24G624)、GUI | macOS 27.0 (26A428)、ヘッドレス |
| ツールチェーン | CLT / Swift 6.4 | Xcode 26.6 / Swift 6.3.3 | CLT / Swift 6.4 |
| SSD | AP0256Q (251 GB) | AP1024Z (読み 6.7 GB/s) | AP0256Z (読み 3.3 GB/s) |
| 接続 | `ssh mini` / remote `m1` | 母艦 | `ssh m6` / remote `m6` |
| 既知の癖 | audiomxd のループは止めてある ([README §2](README.md)) | 熱で止まることがある | — |

**揃わないもの**: RAM (MBP だけ 2 GB 多い)、SSD (MBP が 2 倍速い)、Swift (MBP だけ 6.3.3)。
これらは記事にそのまま書く。「チップの差」と言い切らない。

## 3. 固定する条件

| 項目 | 値 | 理由 |
| --- | --- | --- |
| commit | 3 台とも同じ 1 つ (実行時に `bench/hostinfo.sh` が記録) | `git push m1 HEAD` / `git push m6 HEAD` で揃える |
| バイナリ | 各機で `swift build -c release --product TsugumiCLI` | M1 と M6 の Swift 6.4 では Server が通らないが CLI は通る ([README §3](README.md)) |
| モデル | `scratch/gemma4-qat-sym.gturbo` (vision と draft を含む) | 3 台の `manifest.json` の `sourceSnapshotHash` が一致することを先に見る |
| 実行形 | CLI、run ごとに新しいプロセス | footer が load / prefill / ttft / decode / peak / MTP / vision を全部出す。サーバの HTTP 層を挟まない |
| 生成 | `--thinking on --draft-block-size 4 --max-new 1024` | thinking と MTP は記事の条件。1024 は CLI の既定で、3 台とも同じ長さを吐かせて壁時計を比べる |
| それ以外 | 既定のまま (32 スロット、prefill on、文脈 4096、温度 1.0 / Top-K 64 / Top-P 0.95、検証は既定) | サンプリングは公式値固定。`TF_` 環境変数は 1 つも置かない |
| 画像 | `~/sample_imgs/NO_FUSION_0408_001.png` (リポジトリ外、`--image-tokens` 既定 280) | sha256 `968bb14b…192ec`。3 台に同じファイルを置く。ロゴ類 4 枚は使わない |
| ページキャッシュ | 暖機 1 本の後、前の run が残した状態のまま | 3 台で同じ扱い。キャッシュの purge はしない |

thinking on では 1024 トークンの大半が思考になり、答えが途中で切れることがある。**それでよい**。
測るのは速さで、答えの中身は記事に使わない。`stop=maxTokens` でない run は new トークン数をそのまま書く。

## 4. 課題 — 4 本だけ

「派手で意味がある」ものだけ。文脈長の刻み、MTP の on/off 表、スロット数などは網羅しない (§8)。

| # | 名前 | プロンプト | 見せたい数字 | 何が派手か |
| --- | --- | --- | --- | --- |
| A | 短い日本語の質問 | `bench/compare/prompts/chat-ja.json` (梅雨のしくみ、約 40 トークン) | tok/s、ttft、1024 トークンの壁時計 | 「16 GB の M1 で 26B が喋る」。日常の体感そのもの |
| B | 長い英文を読んで要約 | `docs/benchmark-prompts/real-generation-v1/long-synthesis.json` (README を読ませる、約 3,000 トークン) | **ttft** (prefill)、tok/s | 長文を読ませたときの待ち時間。M1 と M6 で一番差が開くはずの数字 |
| C | 写真の説明 | `bench/compare/prompts/photo-ja.json` + `--image` | **tower 秒**、ttft、tok/s、MTP 受理長 | vision tower の GPU 時間が機械でどれだけ違うか。画像の説明は MTP が効く (約 1.4 倍、[CLI.md](../CLI.md)) |
| D | コード生成 | `bench/compare/prompts/code-en.json` (Python の CLI とテスト) | tok/s、**MTP 受理長**、壁時計 | MTP の一番いい場面。ドラフターの効きが遅い GPU でどうなるか |

A と D の prompt はプロンプトの長さが小さいので decode の値、B は prefill の値、C は tower の値が主役になる。
**B の prompt トークン数は暖機の footer で確かめる** (`prefill=Ntok`)。N + 1024 が 4096 を超えるなら、
3 台とも `--max-context 8192` を付けて、その旨を command.txt に残す (§7 の確認項目)。

## 5. 手順

各機で同じ順に回す。1 台あたり 1 + 12 run。M1 は 1 run に 2 分前後かかる見込み (10〜12 tok/s、[README §4](README.md))。

1. **実行前の確認** ([AGENTS.md](../../AGENTS.md) の規則): モデルを使うプロセスが無い、`memory_pressure -Q`、空き容量、
   `vm_stat` の Swapouts を控える。M1 は `ps` で audiomxd / configd の CPU も見る。MBP は他のアプリを閉じて `caffeinate -i` を挟む。
2. **暖機 1 本** (課題 A、記録するが表には入れない)。シェーダのコンパイルと SSD からの初回読みをここで払う。
3. **本番 12 run を A B C D の順で 3 周**。同じ課題を続けて回さない (熱の偏りを課題間で均す)。run の間は 20 秒空ける
   (`COOL_S`、下げるのはユーザーの判断)。
4. **熱の検定**: 1 周目の A と 3 周目の A の tok/s を比べ、5% 超ずれていたらその機械の 12 run を捨てて回し直す
   (MBP のベンチで使ってきた決まりと同じ)。
5. **反復は 3**。同じセルの 3 本で (最大 − 最小) / 中央値が 5% を超えたら 2 本足して 5 にする。3 本未満のセルには解釈を書かない。
6. run ごとに `vm_stat` を前後で取り、Swapouts が動いた run は印を付ける (捨てるかは表で判断する)。

コマンドの形 (課題 C の例。A / B / D は `--image` を外してファイルを替えるだけ):

```
.build/release/TsugumiCLI --model scratch/gemma4-qat-sym.gturbo \
  --messages-file bench/compare/prompts/photo-ja.json --image ~/sample_imgs/NO_FUSION_0408_001.png \
  --thinking on --draft-block-size 4 --max-new 1024 > run-N.out 2> run-N.log
```

## 6. 証拠の置き場と記録する列

`bench/compare/results/<日付>-<hw.model>-<課題>/` に、[bench/m6/prefill_bench.sh](../../bench/m6/prefill_bench.sh) と同じ構成で置く:
`hostinfo.txt` (先頭に必ず)、`command.txt`、`env.txt` (`TF_` が空であること)、`image.sha256` (C のみ)、
`run-N.log` (footer)、`run-N.out` (生成文)、`run-N.pre.txt` / `run-N.post.txt` (`vm_stat`)、`summary.tsv`。
M6 の分もここに置く。**M6 の台帳 (`bench/m6/results/`) には入れない**。

`summary.tsv` の列 (footer から機械的に取る):

| 列 | 出所 |
| --- | --- |
| run, stop, prefill_tok, new_tok, decode_s, tok_s | `[stop=… prefill=…tok new=…tok decode=…s tok/s=…]` |
| load_s, layerVerify_s, prefill_s, ttft_s, peak_gb | `[load=… layerVerify=… prefill=… ttft=… peak=…]` |
| mtp_accept, mtp_rounds | `[mtp bs=4 rounds=… accept=…/3 …]` |
| tower_s, soft_tok | `[vision images=1 soft=… tower=…s …]` (C のみ) |
| wall_s | スクリプトがプロセスの前後で取る壁時計 (load と検証を含む) |
| swapouts_delta | pre / post の `vm_stat` の差 |
| exit | 終了コード |

これを吐くスクリプト `bench/compare/compare_bench.sh` は未作成。`prefill_bench.sh` を写して、固定引数と footer の awk を上の列に替える。

## 7. 先に確かめること (測る前に潰す)

| 確認 | やり方 | 落ちたら |
| --- | --- | --- |
| M1 で MTP と vision が動く | 暖機の代わりに C を `--max-new 16` で 1 本 | 動かない機能は 3 台とも外して、外した旨を表の上に書く |
| 3 台の pack が同一 | `manifest.json` の `sourceSnapshotHash` と `verified-install.json` を比べる | rsync し直す |
| 3 台の commit が同一 | `hostinfo.txt` の commit と `+dirty` の有無 | push し直す |
| B の prompt が文脈に収まる | 暖機の footer の `prefill=Ntok` | §4 の通り `--max-context 8192` を 3 台に |
| 16 GB で thinking + vision + draft が収まる | 課題 C の `peak=` と Swapouts | 収まらない機械があれば、それ自体が記事の事実 |
| M1 の CLI が今の commit で通る | `swift build -c release --product TsugumiCLI` | ソースは直さない。落ちた commit を書いて止める |

## 8. やらないこと

- サーバ経由の測定、Mac アプリ、prompt cache。CLI の新規プロセスだけ。
- MTP off の表。D だけ MTP off を 3 本足すのは「後回し」に置く (機械ごとの MTP の効きが読めるが、記事の主役ではない)。
- 文脈長を変えた表、スロット数を変えた表、`TF_` の候補経路、サンプリングの変更、キャッシュの purge。
- Ornith / Qwen。記事は Gemma 4 26B-A4B の 1 本。
- 温度 0 (greedy)。運用点ではないし、Gemma は同一文の反復に落ちる。
