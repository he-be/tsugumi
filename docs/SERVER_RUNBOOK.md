# サーバーの建て方 (M3 Pro 18GB 用の手順書)

`TurboFieldfareServer` を pi / OpenCode などの OpenAI 互換クライアントに
つなぐための、この機体専用の手順。一般向けの説明は
[OPENAI_SERVER.md](OPENAI_SERVER.md)、フラグの意味は
[RUNTIME_CONTROLS.md](RUNTIME_CONTROLS.md) にある。

実測は 2026-08-18 / M3 Pro 18GB / macOS 15.7.5 / `scratch/gemma4-qat.gturbo`
(QAT + vision tower + ドラフター)。

---

## 0. 建てる前に 1 つだけ確認する

**TurboFieldfare のプロセスは同時に 1 つだけ。**テストも Mac アプリも数えるので、
先に必ずこれを見る:

```bash
pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

何か出たら、それを止めてから建てる。

```bash
swift build -c release --product TurboFieldfareServer   # 初回とコード変更のあと
```

## 1. コピペする 3 つの構成

**コンテキスト長で決まるのは「スロットの上限」である** (理由は §2)。
長いコンテキストを取ると KV キャッシュがメモリを食い、その分だけ expert
キャッシュのスロットを削る必要がある。**スロットが減ると生成も遅くなる** (§4)。

上限まで取るかどうかは別の判断で、**常用は 3 構成とも 32 スロット**にしている
(2026-08-19 に変更。16K は 80、64K は 64、128K は 48 だった)。上限まで取ると
推奨作業セット 12.88 GB をほぼ使い切り、**同じ機体で他の作業を並行させる余地が
残らない**。32 なら peak 約 5.1 GB で、速度は落ちるが常時起動に耐える。
上限は §2 の表に残してあるので、速度が要る単発の測定だけそちらを明示して付ける。

> **2026-08-20 追記** — 既定の expert 経路が mmap になった
> (`docs/mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md` §8)。私有スロットのコピーが無いので、
> **上の 5.1 GB は旧経路の数字である**。同じ 32 スロットの実測 peak は **約 1.3 GB**
> (CLI、`math` / 256 tok。**サーバー経路では測っていない**)。「他の作業と分け合う」の
> 判断はそのぶん緩む。**同じ日に前面の上限も 32 スロット / 128K に絞った**
> (52 §9a) — 48 以上はもう受け付けない。起動時のガードは**確保するバイトだけ**を
> 数えるようになったので、**128K も 32 スロットのまま素の 8192 で通る**
> (実測 peak 3.88 GB)。`TF_EXPERT_MMAP=0` で旧経路に戻る (そのときの 128K は
> 私有スロットを実際に確保するので正しく弾かれる)。

つまり **3 つの構成の違いはコンテキスト長だけ**になった。

### (a) 16K — pi の常用。他の作業とメモリを分け合う設定

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4-qat.gturbo \
  --port 8091 \
  --max-context 16384 \
  --expert-cache-slots 32 \
  --verification trusted-install \
  --draft-block-size 4
```

### (b) 64K — 長い会話やログを丸ごと貼るとき

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4-qat.gturbo \
  --port 8091 \
  --max-context 65536 \
  --expert-cache-slots 32 \
  --verification trusted-install \
  --draft-block-size 4
```

### (c) 128K — 最長

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4-qat.gturbo \
  --port 8091 \
  --max-context 131072 \
  --expert-cache-slots 32 \
  --verification trusted-install \
  --draft-block-size 4
```

(c) が (a) より 15〜20% 遅いという §4 の数字は、128K/48 と 16K/80 を比べた
ときのものである。スロットを 3 構成とも 32 に揃えた今、両者の差は KV の大きさ
だけになった — **この条件での再測定は未了**。

**ポートは 8091。**`~/.pi/agent/models.json` の `local-turbofieldfare` が
`http://127.0.0.1:8091/v1` を向いている。

バックグラウンドに置いてログを残す形:

```bash
nohup .build/release/TurboFieldfareServer --model scratch/gemma4-qat.gturbo \
  --port 8091 --max-context 65536 --expert-cache-slots 32 \
  --verification trusted-install --draft-block-size 4 > /tmp/tf-server.log 2>&1 &
```

## 2. なぜコンテキストごとにスロット数が変わるのか (**実測**)

起動時のガードが「常駐させる合計」をこの機体の Metal 推奨作業セット
**12.88 GB** と比べ、超えていたら落ちる。**このガードはロードの中にあり、ポートは
もう開いている**ので、クライアントからは 503 `model_loading` のあと接続断に見える
(`exit 1`)。フラグの値そのものが許可リストの外なら、ポートを開く前に `exit 2`。
内訳は実測でこう出る:

この表の「使えるスロット」は**載る上限**であって、常用の設定ではない
(常用は §1 のとおり 32)。

| コンテキスト | KV | resident + vision + scratch | 使えるスロット (上限) |
| ---: | ---: | ---: | --- |
| 16K | 0.96 GB (**導出**) | 1.51 + 1.15 + 0.29 GB | **80** (合計 12.84 GB、上限 12.88 GB にぎりぎり) |
| 64K | **1.97 GB** | 同上 | **64** (80 は 13.85 GB で拒否) |
| 128K | **3.31 GB** | 同上 | **48** (64 は 13.40 GB で拒否) |

スロットは 1 つ約 0.11 GB。**KV が伸びる分をスロットで返す**、それだけの話である。
コンテキストを伸ばしても KV が線形に増えないのは、30 層のうち
**全注意層 (5 層) だけが最大長ぶんを持ち、残り 25 層の SWA はリング長 (窓 1024 +
チャンク 2048 = 3072 行) で頭打ち**だから。SWA 側は常に 0.63 GB で、
伸びるのは全注意層の 5 × (最大長 × 2048 B × 2) だけである。

拒否されたときのメッセージは足し算をそのまま出すので、読んでスロットを 1 段下げればよい:

```
error: expert cache configuration exceeds this device's recommended Metal working set —
resident 1.51 GB + vision 1.15 GB + experts 8.93 GB (80 slots) + kv 3.31 GB
+ prefill scratch 0.29 GB = 15.19 GB; device recommends at most 12.88 GB.
Lower --expert-cache-slots or --max-context.
```

## 3. 建ったことの確認

起動には 20〜40 秒かかる (モデルの検証と mmap)。**ポートは先に開く**ので、
ロード中も接続はできて、全エンドポイントが 503 `model_loading` を返す。
`/v1/models` が 200 で返れば準備完了である (`/health` も同じ)。

```bash
curl -s http://127.0.0.1:8091/health          # {"status":"ok"}
curl -s http://127.0.0.1:8091/v1/models       # モデル ID が 1 つ
curl -s http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-26b-a4b-it",
       "messages":[{"role":"user","content":"Reply with exactly READY."}],
       "temperature":0,"max_tokens":16}'
```

**止め方**: フォアグラウンドなら Control-C。バックグラウンドなら

```bash
pkill -f TurboFieldfareServer
```

## 4. 出る速度 (**実測**、同じ要求を 3 回の中央値)

要求は「tools 4 本を宣言 + streaming + 600 トークン生成」の pi 形:

| 構成 | 壁時計 | MTP オフなら |
| --- | ---: | ---: |
| 16K / 80 スロット | **18.3 s** | 25.3 s (**1.39 倍**) |
| 128K / 48 スロット | 20.6〜22.4 s | (未測定) |

**この表は 80 スロット時の記録である。**§1(a) を 32 スロットに変えたぶんの
再測定はまだ取っていない (32 スロットの生成速度は
[docs/mtp/27-M7-RESULTS.md](mtp/27-M7-RESULTS.md) §2 が別タスクで 23.6 t/s と
出している)。

- **MTP (`--draft-block-size 4`) の効きはタスク次第**: コードと画像の説明で約 1.4 倍、
  日本語の散文はほぼ等速 (受理長 1.06)。詳細は [RESULTS_MTP.md](../RESULTS_MTP.md)。
- **長いプロンプトの prefill は約 193 tok/s** (**実測**、128K 構成で 23,893 トークンの
  プロンプトに 124 秒)。24k トークン貼ったら最初の 1 文字まで 2 分かかる、という感覚で
  よい。答えは合っていた (プロンプト中央に埋めた型番を正しく返した)。
- 完了ログの読み方:

  ```
  request chatcmpl-… completed in 18.264s prompt=208 cached=0 completion=600 finish=length mtp=4 rounds=179 accept=2.346
  ```

  `cached=` はプロンプトキャッシュが再利用したトークン数、`accept=` は MTP が
  1 ラウンドあたり受理したドラフト数 (高いほど MTP が効いている)。

## 5. pi につなぐ

`~/.pi/agent/models.json` は設定済み (`local-turbofieldfare` → 8091)。
サーバーを建ててから pi を起動するだけでよい。

**2026-08-19 に制限が 2 つとも消えた**: 対話モード (tools 有効) で画像が使え、
Reasoning も使える ([docs/serving/SPEC.md](serving/SPEC.md) MSG-6)。

### (a) 思考を pi 側から切り替えたい場合の設定

思考は既定で効く (`--reasoning-budget` の既定は -1 = 無制限)。pi の Shift-Tab で
切り替えたいなら `~/.pi/agent/models.json` の `local-turbofieldfare` に 2 行足す:

```jsonc
"compat": {
  "supportsDeveloperRole": false,
  "supportsReasoningEffort": false,
  "thinkingFormat": "qwen-chat-template"   // ← 追加
},
"models": [{
  "id": "gemma-4-26b-a4b-it",
  "reasoning": true,                        // ← false から変更
  "input": ["text", "image"],
  "contextWindow": 16384,
  "maxTokens": 8192
}]
```

`thinkingFormat` が `qwen-chat-template` のとき、pi は
`chat_template_kwargs: {enable_thinking: …, preserve_thinking: true}` を送る。
サーバーはこれを読む (`reasoning_effort` でも同じことができる)。
応答の `reasoning_content` は pi 側がそのまま思考として表示する。

### (b) 通しの確認手順

```bash
# 1. サーバー (§1(a) のまま。思考は既定で効く)
.build/release/TurboFieldfareServer --model scratch/gemma4-qat.gturbo \
  --port 8091 --max-context 16384 --expert-cache-slots 32 \
  --verification trusted-install --draft-block-size 4

# 2. 別ターミナルで pi (対話モード = tools 有効のまま)
pi --provider local-turbofieldfare --model gemma-4-26b-a4b-it
```

セッション内で見るもの:

| やること | 期待 |
| --- | --- |
| Shift-Tab を押す | `Thinking level: …` と出る。`Current model does not support thinking` なら (a) の `reasoning: true` が入っていない |
| `@sample_imgs/IMG_2113.JPG この写真を説明して` | 画像の説明が返る。以前の 400 (`images cannot be combined with tools`) は出ない。クリップボードからは Ctrl-V でも貼れる |
| `/etc/hosts の 1 行目を教えて` | `bash` などの tool call が走る。思考 ON でも tool 呼び出しは出る |

サーバーの stderr 側で見るもの:

```
request chatcmpl-… accepted streaming=true thinking=on     ← 要求が思考を頼んだ
request chatcmpl-… completed in … finish=stop reasoning=1246B
request chatcmpl-… completed in … finish=tool_calls        ← tool 呼び出し
```

**思考 ON のときの注意 2 つ:**

- **生成予算を食う。**画像 1 枚の説明で思考が 1,200 字・生成 479 トークン
  ほど要る (思考 OFF は 29 トークン)。`maxTokens` を絞りすぎると思考だけで
  尽きて本文が出ない (`finish=length`)。
- **`cached=0` が続くのは異常。**思考 ON でも画像込みでもプロンプトキャッシュは
  効く ([docs/serving/SPEC.md](serving/SPEC.md) §7)。再利用は**トークン列の
  最長共通接頭辞だけ**で決まるので (CACHE-1)、理由の分類は無い —
  完了行の `cached=` の数字が唯一の観測値である (CACHE-6)。
  2 ターン目以降で `cached=0` が続くときに疑うのは 3 つだけ:

  | 疑う先 | 見分け方 |
  | --- | --- |
  | クライアントが履歴を書き換えている (圧縮・要約) | `cached` が 0 ではなく**途中の値**なら、書き換えた地点まで再利用できている。0 なら先頭から違う |
  | 要求が `cache_prompt: false` を送っている | 要求本文を見る (CACHE-5) |
  | 巻き戻し深さを超えた | 分岐点が KV の末尾から 2048 トークン以上前 (SPEC §12 DEV-13)。長い会話の先頭付近を書き換えるとこれになる |

## 6. よくある失敗

| 症状 | 原因と対処 |
| --- | --- |
| `exit 2` + usage が出る | フラグの値が許可リストの外。`--max-context` は 4096 / 8192 / 16384 / 32768 / 65536 / 131072、`--expert-cache-slots` は 8/16/24/32/48/64/80/96/112、`--draft-block-size` は 0 または 2...8 |
| `exceeds this device's recommended Metal working set` | §2。スロットを 1 段下げる |
| `--draft-block-size 4 requires --prefill on` | MTP はチャンク prefill の経路を通る。`--prefill off` とは併用できない |
| `needs a model installed with the drafter section` | そのモデルにドラフターが入っていない。`TurboFieldfareRepack --add-draft <model>` で 236 MB を追記する |
| 500 `prompt contains the special token <\|image\|>` | 貼り付けた本文にテンプレートの特殊トークンが入っている。プロンプト側で取り除く |
| 503 `model_loading` が返る | まだロード中。`/v1/models` が 200 になるまで待つ (20〜40 秒) |
| ポートが開かない / `connection refused` | プロセスが立っていないか、ロードに失敗して落ちた (§2 のガードなど)。stderr の `error:` 行を読む |
| 起動はするのに極端に遅い | 別の TurboFieldfare プロセスが同時に走っている (§0) |

## 7. 画像デモから自動で建てる

§1(a) の 16K 構成を自分で打たずに、起動・ウォームアップ・終了までまとめて任せる
方法もある。`sample_imgs/` の画像を選ぶと、プレフィルの進捗、ストリーミング、
生成速度がそのままブラウザに出る:

```bash
python3 Scripts/demo/serve.py     # http://127.0.0.1:8799/
```

- 建てるのは §1(a) と同じコマンド。Control-C で**このコマンドが建てたサーバーだけ**
  止める。
- 起動後にテキストと画像を 1 回ずつ通す (ウォームアップ)。vision タワーの初回は
  **実測 5〜11 秒**かかるので、これを最初の 1 枚に払わせないためである。
- §0 のプロセスが既にいる場合は**建てず**、8091 が答えるならそれに接続して、
  終了時にも止めない。
- 進捗の出どころと「どれが実測でどれが推定か」は
  [Scripts/demo/README.md](../Scripts/demo/README.md) にある。
