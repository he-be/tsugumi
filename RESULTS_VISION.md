# RESULTS_VISION — 画像入力 (vision) の受入結果

実施: 2026-08-17 / M3 Pro 18GB / macOS 15.7.5 / `macos15-support` ブランチ
commit `e4be236` + V6 の作業ツリー (Server の `image_url` 入口)
Apple Swift 6.3.3 / target arm64-apple-macosx15.0
モデル: `scratch/gemma4-qat.gturbo` (QAT / 15 GB + tower 1.15 GB、`--add-vision` で追記)
プロトコル: PLAN §6 / PLAN_VISION §7 (`.build/release` 直叩き、footer 全文を記録)

表記は PLAN と同じ: **実測** / **導出** / **未確認**

---

## 0. 結論 — **受入。既定 soft token は 280 のまま** (**実測**)

| # | ゲート | 判定 | 実測 |
| --- | --- | --- | --- |
| 1 | 正しさ (層 B) | **合格** | `KernelCheck --vision-tower` 121 ケース PASS、exit 0 (§1) |
| 2 | 目視 | **合格** | 実写 3 枚 + 図表 1 枚の日本語質問で、色・数・位置が画像と一致 (§2) |
| 3 | テキスト回帰 | **合格** | `bench.sh ja` 中央値 32.2 / 28.8 / 31.4 tok/s。V6 はテキスト経路に 1 行も触れていない (§3) |
| 4 | TTFT | **記録した** | S=280 で **4.13〜4.16 s** (trusted-install)。10 s 線に触れないので既定 280 を維持 (§4) |
| 5 | メモリ | **合格** | peak **6.97〜7.01 GB** < 12 GB、48 スロットのまま (§4) |
| 6 | Server | **合格** | `data:` 200 / `http(s)` 400 / 画像ありでキャッシュ publish なし (§5) |
| 7 | 起動 | **合格** | `trusted-install` / `full-sha256` 両方で exit 0 (CLI・Server とも) (§6) |
| 8 | 退行なし | **合格** | 凍結フィクスチャ (manifest / layout / resident index の SHA-256) がそのまま通る (§7) |
| 9 | 旧ランタイム拒否 | **合格** | `7b625f6` のビルドが `unknown key "visionTower"` で exit 1、対照は exit 0 (§8) |
| 10 | 追記 | **合格** | 1,145,588,832 B のみ取得、テキスト側 inode 不変、再検証 exit 0 (§9) |

---

## 1. ゲート 1 — 参照実装との一致 (**実測**)

```
$ ./.build/release/TurboFieldfareKernelCheck --vision-tower scratch/gemma4-qat.gturbo
PASS  121 cases (group sizes [64, 32] + vision)          exit 0   (11.9 s)
```

閾値は PLAN_VISION §0-G-2 で**測って**決めた 2 本立て (max / rms)。
6 fixture × 段階別 + 検出力 3 件。soft token の最悪ケース:

| fixture | max (許容 8e-2) | rms (許容 2e-3) |
| --- | ---: | ---: |
| wide-1024x768-s280 (P=2394, S=266) | 1.186e-2 | 5.017e-4 |
| wide-1024x768-s70 (P=567, S=63) | 5.684e-3 | 3.805e-4 |

検出力 (わざと壊した参照との差が閾値を**超えること**が PASS 条件):
格子の転置 max 1.269 / 標準化の省略 0.605 / 全層が層 0 の重み 1.329 — 通常時の 20〜100 倍。

---

## 2. ゲート 2 — 目視 (**実測**、Server 経由 / temp 0)

実写 3 枚 + 図表 1 枚に、**色・数・位置関係を問う日本語の質問**を投げた
(`sample_imgs/`、`data:` URI で `image_url` content part)。
回答は画像を実際に開いて突き合わせた。

| 画像 | 質問 | 回答 | 判定 |
| --- | --- | --- | --- |
| IMG_2113.JPG (実写・室内) | ラグの色 / クッションの数 / スマホの位置 | ベージュ（薄いグレー）/ **2 つ** / 画面右側、人物の頭のすぐ近く | **一致**。緑のクッションは 2 個、赤ケースのスマホは頭の右横 |
| IMG_2115.JPG (実写・車内) | 首輪の色 / 犬の向き / 屋内か屋外か | 青色 / 左（車の窓の外）/ 屋内（車内） | **一致** |
| IMG_2118.JPG (実写・店内) | 中央の看板の内容と色 / 看板上の文字 | 黄色い立て看板にマグロ、魚は青みがかった灰色 / 「170、160、150cm…」の目盛りと「本まぐろ」 | **一致**。目盛りの数値まで読めている |
| 2026-07-12_225404.png (図表・UI) | パネルの数 / 左から何を表示 / テーマ | **3 つ** / Model Loader・Image To 3D・Export GLB + 3D プレビュー / ダークテーマ | **一致**。ノード名も表示順も画面どおり |

**画像を見ていないと書けない内容** (クッション 2 個、身長目盛りの数値、3 パネルの
左右の並び) が 4 枚すべてで出ている。

> **一つだけ完全一致ではない例も記録する。**IMG_2114 (チューリップ畑) に
> 「花の色を手前から順に」と聞くと「赤 → 黄色 → 白 → ピンク」と答えた。
> 手前が赤、次が白、その次が黄色である (画面左半分)。ただし右半分では
> 白と黄の帯の前後が逆に見えるので、**色の集合は正しく、中間 2 帯の順序だけが
> 画像から一意に読めない**。誤りとは言い切れないが、一致とも言わない。

V5 の 8 枚キャプション比較 (別マシンの llama-swap / 同一チェックポイント) は
PLAN_VISION §0-H-1 に記録済みで、本ゲートはそれとは別に「色・数・位置」を
名指しで問う形で取り直したものである。

---

## 3. ゲート 3 — テキストのみの回帰 (**実測**)

```
$ TEMP=0 MAXNEW=384 ./bench.sh ja        # 64 スロット / chunk 128 / trusted-install
```

3 回インターリーブ、中央値:

| prompt | V6 (今回) | V5 (§0-H-2) | `RESULTS_QAT.md` §2-1 | 生成長 |
| --- | ---: | ---: | ---: | ---: |
| haiku | **32.226** | 23.366 | 21.286 | 384 (maxTokens) |
| math | **28.849** | 24.406 | 19.602 | 384 |
| story | **31.371** | 28.576 | 23.447 | 384 |

peak 7.02〜7.12 GB、decode hit 97.3〜99.2%、io/tok 1.7〜6.8 ms。

**判定は合格だが、この表を「V6 で速くなった」と読んではいけない。**
V6 の差分はサーバ側 (`TurboFieldfareServerCore`) と、テキスト経路が
**一度も呼ばない**新しい静的関数 1 本 (`VisionImagePreprocessor.pixelSize`、+22 行) だけである
(`git diff --stat`: 変更 9 ファイル中、ランタイムはこの 1 ファイルのみ)。
CLI の prefill / decode 経路には 1 行も触れていないので、
**速くなる経路も遅くなる経路も存在しない。**

上の 3 つの列がそれぞれ 10〜50% 違うのは**測定セッションの状態**の差である
(page cache の温度、熱、メモリ圧)。V5 が同一機の A/B を採った理由がこれで、
セッションを跨いだ比較は ±4% ゲートの分解能を持たない。
今回は A/B を採らず、**差分がテキスト経路に存在しないこと**を根拠にした。
そのうえで実測値は記録されたどのベースラインよりも速く、退行がここに隠れる余地はない。

prefill 側 (`PREFILL_THROUGHPUT.md` §7-9 の 231 pp / peak 6.64 GB) は
V5 で A/B 済み (231.8 対 230.1)。V6 では prefill 経路にも触れていないため再測していない
(**未確認**)。

---

## 4. ゲート 4 / 5 — TTFT とメモリ (**実測**)

S=280 になる形を作るため、`sample_imgs/IMG_2114.JPG` を **1000×700** に縮小した
(960×672 に丸められ 60×42 = 2520 patch = **280 soft token**)。実写の元画像は
アスペクト比の関係で 260〜266 soft token にしかならない (PLAN_VISION §0-B-2)。

```
$ ./.build/release/TurboFieldfareCLI --model scratch/gemma4-qat.gturbo \
    --messages-file msg-image.json --image s280-1000x700.jpg \
    --temperature 0 --max-new 64 --max-context 16384 --verification trusted-install
[stop=maxTokens prefill=309tok new=64tok decode=3.35s tok/s=19.103]
[load=0.708s layerVerify=0.000s/30layers prefill=4.161s ttft=4.161s peak=7.01GB rss=4.08GB]
[expert prefill hit=0.0% 0/2445 io=1.565s | decode hit=93.0% 14068/15120 io=1.281s]
[vision images=1 soft=280 tower=1.904s towerLoad=0.002s]
[decode/tok io=20.06ms cb1=0.69ms cb2=0.29ms head=3.59ms]
exit 0
```

| 構成 | TTFT | 塔 | towerLoad | peak | exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `trusted-install`、S=280 | **4.161 s** | 1.904 s | 0.002 s | 7.01 GB | 0 |
| `trusted-install`、S=280 (2 回目、max-new 32) | **4.129 s** | 1.900 s | 0.002 s | 6.97 GB | 0 |
| `full-sha256`、S=280 | **9.172 s** | 1.509 s | 0.488 s | 6.98 GB | 0 |

- **ゲート 4 は発火しない。**10 s を超えたら既定 soft token を 140 に落とす、という
  条件付きの判断は**行わない**。S=280 の TTFT は 4.13〜4.16 s である。
- `full-sha256` の 9.17 s は塔ではなく**初回の層検証** (`layerVerify=6.088s`) と
  tower の SHA-256 (`towerLoad=0.488s`) が占める。どちらもプロセスにつき 1 回で、
  2 通目以降の画像には乗らない。塔そのものは 1.5〜1.9 s。
- **ゲート 5 合格**: peak 6.97〜7.01 GB (テキストのみは 6.64 GB)。12 GB 予算に対し
  5 GB の余白があり、既定 48 スロットのまま通る。
- `load=` は 0.708〜0.759 s で、テキストのみの実行 (0.70〜0.75 s) と変わらない。
  vision を別ファイルにした狙い (§4-1) は V6 でも守れている。

---

## 5. ゲート 6 — Server (**実測**)

`.build/release/TurboFieldfareServer --model scratch/gemma4-qat.gturbo --port 8080
--max-context 16384 --verification full-sha256` に対して:

| 検査 | 結果 |
| --- | --- |
| `data:image/jpeg;base64,…` (IMG_2112、190,513 B) | **200**。prompt_tokens=283 (soft 260 + テキスト)、10.51 s (初回、tower 検証込み)、以降 4.36 s |
| `https://example.invalid/cat.png` | **400** `unsupported_image_url` — 「this server accepts only data: URIs and never fetches image URLs」 |
| `http://127.0.0.1:8080/health` (自分自身を指す URL) | **400** 同上。**取りに行かない** |
| テキストのみの 2 ターン目 | `cached_tokens=25` — **プロンプト再利用は生きている** |
| 画像リクエスト (1 回目 / 2 回目) | `cached_tokens=0` / `0` — 同じ画像・同じ文でも**当てない** |
| 画像の次のテキストターン | `cached_tokens=0` — 画像ターンは **publish もしていない** |

最後の 3 行が §4-6 の要求そのものである。「テキスト再利用が生きている」ことを
同じ実行の中で確認しているので、`cached_tokens=0` が「キャッシュが死んでいるから 0」
ではなく「画像だから 0」であることが言える。

413 側 (`image_too_large` / `too_many_images`) と本文サイズの上限は
`Tests/TurboFieldfareServer/ServerImageRequestTests.swift` が HTTP 経由で固定している
(実サーバでの再確認は行っていない — **未確認**)。

---

## 6. ゲート 7 — 起動 (**実測**)

| バイナリ | `--verification` | 結果 |
| --- | --- | --- |
| CLI (画像 1 枚) | `trusted-install` | exit 0 |
| CLI (画像 1 枚) | `full-sha256` | exit 0 |
| Server (画像 1 枚を処理) | `trusted-install` | 起動・応答 200、SIGINT で **exit 0** |
| Server (画像 6 リクエスト) | `full-sha256` | 起動・応答 200 |

> Server の終了コードは一度 1 に見えたが、**測り方の誤り**だった:
> `pkill -f 'TurboFieldfareServer --model'` は起動用のシェル自身にも一致するので、
> シェルを殺していた。PID を捕まえて `kill -INT` した測定では exit 0 である。

---

## 7. ゲート 8 — 退行なし (**実測**)

`Scripts/test.sh`: **803 テスト / 138 スイート、11 issue**。
issue の内訳は `PREFILL_THROUGHPUT.md` §7-7 の陳腐化 4 スイート
(QAT ピン / prefill 2048 / 48 スロット / causalQBlock) と**完全に同じで、
新しい失敗はない** (V5 の 784 → 803 は今回の +19)。

`Tests/TurboFieldfareFormatCompatibility` の凍結フィクスチャ (manifest / layout /
resident index の SHA-256) が通っている = `--include-vision` なしの `.gturbo` は
現行とバイト一致する。`vision` は optional なので text-only の JSON にキー自体が現れない。

---

## 8. ゲート 9 — 旧ランタイムが拒否すること (**実測**)

PLAN_VISION §0-D-6 が「このリポジトリ内では直接試験できない」として V6 に送った項目。
**vision 以前の commit `7b625f6`** (「routed MoE を expert 単位 GEMM に (231pp)」= 
vision の作業が 1 行も入っていない最後のコミット) を一時 worktree に建て、
そのビルドに vision 付きの `.gturbo` を食わせた。

```
$ .../prevision/.build/release/TurboFieldfareCLI \
    --model scratch/gemma4-qat.gturbo --prompt "The capital of France is" --max-new 8
error: manifest.flags contains unknown key "visionTower"
exit 1
```

**対照 (この検査の検出力そのもの):** 同じバイナリで、`visionTower` フラグを持たない
`scratch/gemma4.gturbo` (旧 4bit インストール) は**正常に生成して exit 0**。
拒否はフラグに対するものであって、「古いバイナリが何にでも失敗する」のではない。

| バイナリ | モデル | flags | 結果 |
| --- | --- | --- | --- |
| `7b625f6` (vision 以前) | `gemma4-qat.gturbo` (vision あり) | `visionTower: true` | **exit 1**、`unknown key "visionTower"` |
| `7b625f6` (vision 以前) | `gemma4.gturbo` (vision なし) | フラグなし | exit 0、生成成功 |

> メッセージの実文言は PLAN が予想した `unknown v1 flag` ではなく
> `manifest.flags contains unknown key "visionTower"` である。意味は同じで、
> **どのフラグが未知なのかを名指ししている**ぶん実物のほうが良い。

worktree は検査後に `git worktree remove` で撤去した (作業ツリーには残っていない)。

---

## 9. ゲート 10 — `--add-vision` (**実測**、PLAN_VISION §0-E-5 の再掲)

```
$ ./.build/release/TurboFieldfareRepack --add-vision --input-gturbo scratch/gemma4-qat.gturbo
Added the vision tower to …/scratch/gemma4-qat.gturbo
Tower: 356 tensors, 1145588832 bytes
Source: google/gemma-4-26B-A4B-it-qat-q4_0-unquantized @ f1e06dc520982d9b9edd76859fdb7ab209449949
Downloaded 1145588832 bytes
Re-verified 38 files (16980804090 bytes)
      183.15 real        11.58 user        10.84 sys     (exit 0)
```

ダウンロードは tower の 1,145,588,832 B のみ (テキスト側 0 B)。
`model_weights.bin` / `packed_experts/layout.json` の (inode, mtime) は前後で不変。
本受入で使ったインストールはこの追記で作ったものであり、
**15 GB のコピーは 1 度も発生していない**。

---

## 10. プロトコルからの逸脱

- ゲート 2 の 4 枚目は「図表」として **UI のスクリーンショット**を使った (グラフではない)。
  §7 の「図表 1 枚」の趣旨 (自然画像でない、文字と構造のあるもの) は満たしている。
- ゲート 6 の 413 系は実サーバではなくテスト経由での確認 (§5)。
- ゲート 3 の decode ベースラインは `RESULTS_QAT.md` の値、prefill は
  `PREFILL_THROUGHPUT.md` §7-9 の値を使った (PLAN_VISION §0-A-5 のとおり)。
