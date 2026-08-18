# RESULTS_MTP — MTP (投機デコード) の受入結果

実施: 2026-08-17〜18 / M3 Pro 18GB / macOS 15.7.5 (24G624) / `macos15-support` ブランチ
commit `a533171` + M6 の作業ツリー (Server の `--draft-block-size`)
Apple Swift 6.3.3 / target arm64-apple-macosx15.0
モデル: `scratch/gemma4-qat.gturbo` (QAT 15 GB + vision tower 1.15 GB + ドラフター 236 MB、
`--add-draft` で追記)
プロトコル: PLAN §6 / [docs/mtp/22-GOAL-RESET.md](docs/mtp/22-GOAL-RESET.md) §4-5
(`.build/release` 直叩き、同一セッションの on/off 交互 A/B、絶対 t/s は採点に使わない)

表記は PLAN と同じ: **実測** / **導出** / **未確認**

---

## 0. 結論 — **受入。既定はオフのまま、運用は `--draft-block-size 4`** (**実測**)

| # | ゲート | 判定 | 実測 |
| --- | --- | --- | --- |
| — | **採点基準 v1** | **達成** | ゴールタスク 8 枚の e2e 壁時計比の**中央値 1.336 倍** (目標 1.33 / 中止 1.10)。8 枚すべて 1.28 以上、長さ比 1.000 (§1) |
| 1 | 出力一致 | **分岐を記録した** | code 2,213 文字と 2 ターン対話は全一致、日本語散文は 247 文字目で分岐。**提案ゼロ (`TF_MTP_DRAFTS=0`) でも 90 文字目で分岐する**ので、原因は投機ではなく経路差 (§3) |
| 2 | 正解到達時間 | **読み替えた** | 22 §1 が「導出を成績表に載せた」誤りを特定したため、採点は §1 の 1 本に一本化した。tok/s は載せない (§1) |
| 3 | テキスト回帰 (MTP 無効) | **合格** | 同一セッション交互 A/B で **+0.75%** (±4% 以内)。M5.5 以降の変更は `speculativeBlock` かつ `t > 1` の中だけ (§4) |
| 4 | メモリ | **合格** | peak **10.32〜10.36 GB** < 12 GB (80 スロット)。MTP の新規確保はブロック用 scratch の 4 MB 1 本のみ、オフなら確保しない (§4) |
| 5 | ロード時間 | **合格** | `load=0.763 s` で追記前と同じ桁。236 MB のドラフターは遅延ロードで、オフの実行では読まない (§5) |
| 6 | 旧ランタイム拒否 | **合格** | pre-M1 ビルドが `unknown key "mtpDraft"` で exit 1、対照 (フラグなし) は exit 0 (§5) |
| 7 | 追記 | **合格** | `--add-draft` が **236,114,440 B ちょうど**を取得、テキスト側 inode 不変、再検証 exit 0 (§5) |
| 8 | KV 巻き戻しの検出力 | **合格** | わざと壊した巻き戻し 3 通りが全部 FAIL、正常な `rewind/*` 4 本は PASS (§6) |
| 9 | Server | **合格** | `cached_tokens` が MTP on/off で一致 (0 → 209)、キャッシュがヒットしたターンの本文も一致 (§2) |
| 10 | 退行なし | **合格** | `Scripts/test.sh` **860 テスト / 既知の 11 issues のみ**、新規失敗ゼロ (§7) |

**残した傷は 2 つあり、通ったことにしていない**: `--verify-block` の 3 FAIL (§6) と、
**日本語散文では速くならないこと** (受理長 1.058、§2)。

---

## 1. 採点基準 v1 — ゴールタスク 8 枚 (**実測**、[25-M5.6 §3](docs/mtp/25-M5.6-RESULTS.md))

```
$ ./bench/mtp_goal_ab.py --block-size 4
```

`sample_imgs/` 8 枚 / 固定プロンプト / Reasoning ON / temp 0 / 80 スロット /
画像ごとに別プロセス / 画像ごとに on/off の順序を入れ替え / 生成長に上限なし。
16 本すべて `stop=endOfTurn`:

| | 値 |
| --- | --- |
| **e2e 壁時計比の中央値** | **1.336** (画像ごとに 1.282 / 1.309 / 1.327 / 1.325 / 1.344 / 1.364 / 1.384 / 1.401) |
| 生成長の比 (中央値) | **1.000** — 速い側が短く答えたのではない |
| 受理長 a(4) | **1.885** |
| decode t/s の比 (中央値) | 1.419 (**参考**。採点には使わない) |
| off と on で文字まで一致 | 4/8 |

到達までの内訳は `docs/mtp/README.md` の読む順に全部ある。要約すると
**費用側だけを 3 段で削った**: k 行カーネル (M4.5)、expert I/O の先行発行と
スロット 80 (M4.6-4.7)、活性化側の k 倍の除去 (M4.8)、ブロックの attention を
decode の split-KV へ (M5.5)、k 行を 1 回の split-KV に畳む (M5.6)。
**受理率側は M3.5 以降 1 行も触っていない。**

## 2. Server (**実測**、[26-M6 §2](docs/mtp/26-M6-RESULTS.md))

`--expert-cache-slots 80 --verification trusted-install --max-context 16384`、temp 0。
on と off は別プロセス (サーバーの設定はプロセス固定)、1 回捨ててから 3 回の中央値。

**(a) pi 形のコーディング要求** (tools 4 本を宣言 + `stream: true` + `max_tokens 600`):

| | off | on (bs=4) | 比 |
| --- | ---: | ---: | ---: |
| 壁時計 | 25.329 s | **18.280 s** | **1.386** |
| 出力 | 2,213 文字 | 2,213 文字 | **1 バイト差なし** |
| 受理長 / ラウンド | — | 2.346 / 179 | |

**(b) 2 ターンの対話** (2 ターン目はプロンプトキャッシュがヒットする):

| ターン | off | on | 比 | `cached_tokens` (off/on) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8.257 s | **5.453 s** | 1.514 | 0 / 0 |
| 2 | 7.351 s | **5.004 s** | 1.469 | **209 / 209** |

本文は両ターンとも一致。**ゲート 9 はこの行で合格**である。

**(c) 日本語の散文** (600 字程度の説明):

| | off | on (bs=4) |
| --- | ---: | ---: |
| 壁時計 | 11.214 s | 11.136 s (**1.007**) |
| 受理長 | — | **1.058** |

> **受理長はタスクで 2 倍動く** (**実測**): vis 1.885 / code 2.346 / 日本語散文 1.058。
> 費用側は同じなので、**同じバイナリで 1.39 倍のタスクと 1.0 倍のタスクがある**。
> 13-M3 が測った「日本語長文 0.91〜1.12 倍」がそのまま残っている、と読むのが正しい。

その他 (**実測**): tool call は on/off で同一 (`finish_reason=tool_calls`、引数まで一致)。
`repetition_penalty != 1.0` の要求は**その要求だけ**素の decode 経路で答える
(D5 の受理規則が未確定の履歴に依存できないため)。ドラフター無しのモデルや
`--prefill off` との併用は**ポートを開く前に** exit 2 で落ちる。

## 3. ゲート 1 — 出力一致 (**実測**)

| 比較 | 結果 |
| --- | --- |
| `bench/m.json` (96 tok / thinking off / temp 0) の MTP off × on | **215 / 215 全一致** |
| bs=4 × bs=2 / bs=4 × 投機ゼロ | 全一致 |
| Server: code (600 tok、tools + streaming) | **2,213 文字 全一致** |
| Server: 2 ターン対話 (164 + 161 tok) | **全一致** |
| Server: 日本語散文 (294〜319 tok) | **247 文字目で分岐** |
| Server: 日本語散文、`TF_MTP_DRAFTS=0` (提案ゼロ) | **90 文字目で分岐** |

**投機を切っても分岐する。**したがって分岐は投機ではなく、15-M4 §2 が特定した
「ブロック経路 対 スカラー decode 経路」の FP16 の丸め差である。競り合った位置
(top-1 マージンが logits 誤差の帯 3.8e-2 より小さい位置) がどちらに倒れるかを
丸めが決め、長い日本語生成では**数百トークンに 1 回**それが起きる。
**どちらが「正しい」かはこの検査では決まらない** — 両方ともターゲットの argmax である。
各条件の生成はプロセスを跨いで決定的 (同条件 2〜3 回で 1 バイト差なし)。

## 4. ゲート 3・4 — テキスト回帰とメモリ (**実測**)

| | |
| --- | --- |
| MTP 無効のテキスト経路 | 同一セッション交互 A/B で **+0.75%** (23-M5 §0)。M5.5 以降の変更は `speculativeBlock` かつ `t > 1` の中だけで、decode も prompt prefill も 1 命令も変わらない (**導出**) |
| peak メモリ | **10.32〜10.36 GB** (80 スロット / vision + Reasoning ON、25-M5.6 §4)。12 GB 予算の内側 |
| MTP の新規確保 | ブロック用 partial scratch **4 MB** 1 本 + ブロックの logits/hidden 行。**`--draft-block-size 0` では確保しない** |

## 5. ゲート 5・6・7 — ロード・旧ランタイム・追記 (**実測**、[11-M1](docs/mtp/11-M1-RESULTS.md))

```
$ .build/release/TurboFieldfareRepack --add-draft scratch/gemma4-qat.gturbo
Drafter: 48 tensors, 236114440 bytes
Downloaded 236114440 bytes                                            (exit 0)
```

| 項目 | 実測 |
| --- | --- |
| 取得バイト | **236,114,440 B ちょうど** (payload と同じ。ギャップの余分な取得はゼロ) |
| テキスト側 | inode / mtime / サイズとも不変。再検証 exit 0 |
| ロード | `load=0.763 s` (追記前 0.708 s と同じ桁)。`draft/` は load 経路に現れない |
| 旧ランタイム | pre-M1 ビルドが `manifest.flags contains unknown key "mtpDraft"` で **exit 1**、対照の `gemma4.gturbo` は exit 0 |
| 重みの一致 | HF から Range 取得した 3 テンソルが**バイト一致** |

## 6. ゲート 8 — 巻き戻しと verify の検査 (**実測**)

```
$ .build/release/TurboFieldfareKernelCheck --verify-block scratch/gemma4-qat.gturbo \
    --verify-block-size 4 --verify-rounds 8
12 中 9 PASS / 3 FAIL
```

| ケース | 結果 |
| --- | --- |
| `rewind/*` 4 本 (正常な巻き戻し、わざと壊した 3 通り) | **PASS** (壊した 3 通りは意図どおり FAIL する) |
| `verify/rows/logits k=4` (tol 8e-2) | PASS (worst 1.9e-2) |
| `verify/rows/logits k=1` / `argmax k=1` | PASS |
| `verify/rows/wide-vs-width1` | **FAIL** (3.2、tol 3) |
| `verify/rows/argmax k=4` / `verify/rows/greedy` | **FAIL** — どちらも同じ pos 25 (top-1 マージン 1.0e-3) の argmax 反転 |

**3 FAIL は残したまま受け入れている。**誤差そのものは M5.5 から縮んでいる
(worst 4.0e-2 → 1.9e-2) が、ガードの意図 (「幅を広げたことで増える誤差を検出する」) は
まだ満たしていない。§3 の分岐と同じ現象の、閾値側から見た姿である。

## 7. 退行・逸脱・残っているもの

**ゲート 10 (退行なし)** (**実測**):

```
$ ./Scripts/test.sh
Test run with 860 tests in 145 suites failed after 673.016 seconds with 11 issues.
```

11 issues は **4 スイート** (`AppContextLengthOptionTests` / `AppModelInstallTests` /
`AppRuntimeOptionsTests` / `RuntimeConfigurationTests`) に閉じている。**すべて
QAT・M2 以前の既定値をピン留めしたままの陳腐化した監査**で、MTP 以前から同じ 11 件である
(24-M5.5 / 25-M5.6 と同一)。**新規失敗ゼロ。**テスト数は M6 の新規 2 本で 858 → 860。

**逸脱**:

- **採点基準を途中で作り直した** (22-GOAL-RESET)。M4 系の 6 本が互いの速度比を
  差し替え続けた原因は、別条件で測った 2 つの proxy の積 (**導出**) を成績表に
  載せたことだった。**倒れたのは水準だけで、同一セッション A/B の差分は 1 件も倒れていない。**
- **ゲート 2 (正解到達時間) を採点に使っていない。**上と同じ理由で、採点は
  ゴールタスク 8 枚の壁時計 1 本に絞った。
- **サーバーは常に logits ヘッド経路で走る** (`forceLogitsHead: true`)。§1 の採点は
  temp 0 = 融合 greedy ヘッドなので、§2 のサーバーの数字は不利な側で測られている
  (**導出**、A/B は取っていない)。

**残っているもの** (着手しない。[25-M5.6 §5](docs/mtp/25-M5.6-RESULTS.md) / [26-M6 §4](docs/mtp/26-M6-RESULTS.md)):

| | いま | 床 (**導出**) |
| --- | ---: | ---: |
| `moe` | 27 ms | 約 16 ms |
| ホスト往復 (ブロックの GPU idle) | 14 ms | — |
| `post` (router の行ループ) | 5 ms | 約 1 ms |
| 受理長の低いタスク | a=1.06 (日本語散文) | ドラフター側の話。M3.5 以降未着手 |

`bench/mtp_budget.py` の入力 (F=1.33 / M=0.31) は 23-M5 のもので、
**M5.5 / M5.6 の 2 回ぶん古い** (**未確認**)。再開するならまず引き直すこと。

## 8. 運用点

```bash
.build/release/TurboFieldfareServer --model scratch/gemma4-qat.gturbo \
  --port 8091 --expert-cache-slots 80 --verification trusted-install \
  --draft-block-size 4
```

```bash
.build/release/TurboFieldfareCLI --model scratch/gemma4-qat.gturbo \
  --messages-file bench/mtp_goal_prompt.json --image sample_imgs/IMG_2112.JPG \
  --thinking on --temperature 0 --expert-cache-slots 80 --draft-block-size 4
```

- **既定はオフ** (`0`)。オフの実行は MTP 以前と 1 命令も変わらない。
- **幅は 4。**k=1 は運用点ではなく (21 §5)、費用の伸びは M4.8 以降緩んで
  bs=4 が最良に戻っている (20-M4.8 §5)。
- **効くのは受理長が高いタスク** — コード・画像の説明。日本語の散文は等速である。

### 8-b. メモリの少ない機械 (16GB) の運用点 (**実測**、[27-M7](docs/mtp/27-M7-RESULTS.md))

上の 80 スロットは 18GB 機の設定である (peak 10.3 GB)。16GB 機に載せるなら:

| スロット | peak | 画像キャプション decode t/s (bs=4) | 備考 |
| ---: | ---: | ---: | --- |
| 32 | 5.09 GB | **23.6** | `Scripts/demo/serve.py` の既定。**expert I/O 律速** (ブロック 1 回で SSD から 469 MB) |
| **48** | **6.83 GB** | **28.9〜29.9** | I/O の壁が消える運用点。64 にしても速くならない |
| 64 | 8.67 GB | 28.5 | 48 と同じ (もう I/O では律速していない) |

**採点基準 v1 は 32 スロットで 1.394** (80 スロットの 1.336 より高い)。8 枚とも
`endOfTurn`、長さ比 1.000、文字まで一致 4/8。スロットが少ないほど投機が効く —
off 側も expert I/O で律速していて、ブロックは k トークンの和集合 1 件で読むので
**産出トークンあたりの読み出しが decode より少ない** (27 §2c)。

27-M7 の 3 つの改修 (層 union の 1 プラン / 常駐タイル先頭 / router の k 行畳み込み) で
**32 スロットは 20.8 → 23.6 t/s、48 スロットは 24.7 → 28.9 t/s**。
**出力は 1 バイトも変わらない** (27 §8)。
