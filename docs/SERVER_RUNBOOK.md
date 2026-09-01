# サーバーの建て方 (M3 Pro 18GB 用の手順書)

`TsugumiServer` を pi / OpenCode などの OpenAI 互換クライアントに
つなぐための、この機体専用の手順。一般向けの説明は
[OPENAI_SERVER.md](OPENAI_SERVER.md)、フラグの意味は
[RUNTIME_CONTROLS.md](RUNTIME_CONTROLS.md) にある。

実測は 2026-08-18 / M3 Pro 18GB / macOS 15.7.5 / `scratch/gemma4-qat.moepack`
(QAT + vision tower + ドラフター)。

---

## 0. 建てる前に 1 つだけ確認する

**Tsugumi のプロセスは同時に 1 つだけ。**テストも Mac アプリも数えるので、
先に必ずこれを見る:

```bash
pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

何か出たら、それを止めてから建てる。

```bash
swift build -c release --product TsugumiServer   # 初回とコード変更のあと
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
.build/release/TsugumiServer \
  --model scratch/gemma4-qat.moepack \
  --port 8091 \
  --ctx-size 16384 \
  --expert-cache-slots 32 \
  --verification trusted-install \
  --draft-block-size 4
```

### (b) 64K — 長い会話やログを丸ごと貼るとき

```bash
.build/release/TsugumiServer \
  --model scratch/gemma4-qat.moepack \
  --port 8091 \
  --ctx-size 65536 \
  --expert-cache-slots 32 \
  --verification trusted-install \
  --draft-block-size 4
```

### (c) 128K — 最長

```bash
.build/release/TsugumiServer \
  --model scratch/gemma4-qat.moepack \
  --port 8091 \
  --ctx-size 131072 \
  --expert-cache-slots 32 \
  --verification trusted-install \
  --draft-block-size 4
```

(c) が (a) より 15〜20% 遅いという §4 の数字は、128K/48 と 16K/80 を比べた
ときのものである。スロットを 3 構成とも 32 に揃えた今、両者の差は KV の大きさ
だけになった — **この条件での再測定は未了**。

**ポートは 8091。**`~/.pi/agent/models.json` の `local-tsugumi` が
`http://127.0.0.1:8091/v1` を向いている。

バックグラウンドに置いてログを残す形:

```bash
nohup .build/release/TsugumiServer --model scratch/gemma4-qat.moepack \
  --port 8091 --ctx-size 65536 --expert-cache-slots 32 \
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
Lower the expert-cache slots or the context size.
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
pkill -f TsugumiServer
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
  日本語の散文はほぼ等速 (受理長 1.06)。
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

`~/.pi/agent/models.json` は設定済み (`local-tsugumi` → 8091)。
サーバーを建ててから pi を起動するだけでよい。

**2026-08-19 に制限が 2 つとも消えた**: 対話モード (tools 有効) で画像が使え、
Reasoning も使える ([docs/serving/SPEC.md](serving/SPEC.md) MSG-6)。

### (a) 思考を pi 側から切り替えたい場合の設定

思考は既定で効く (`--reasoning-budget` の既定は -1 = 無制限)。pi の Shift-Tab で
切り替えたいなら `~/.pi/agent/models.json` の `local-tsugumi` に 2 行足す:

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
.build/release/TsugumiServer --model scratch/gemma4-qat.moepack \
  --port 8091 --ctx-size 16384 --expert-cache-slots 32 \
  --verification trusted-install --draft-block-size 4

# 2. 別ターミナルで pi (対話モード = tools 有効のまま)
pi --provider local-tsugumi --model gemma-4-26b-a4b-it
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

**「詰まってないか」は stderr で見るのをやめてよい。**`--metrics` を付けて
建てれば `GET /metrics` が累積のトークン数と秒数を返し、`GET /slots` が
生成スロットが塞がっているかを返す (`--no-slots` で切れる)。

```bash
curl -s http://127.0.0.1:8091/slots     # [{"id":0,"is_processing":false}]
curl -s http://127.0.0.1:8091/metrics   # Prometheus 形式
```

**思考 ON のときの注意 2 つ:**

- **生成予算を食う。**画像 1 枚の説明で思考が 1,200 字・生成 479 トークン
  ほど要る (思考 OFF は 29 トークン)。`maxTokens` を絞ると思考の締切が早まり、
  サーバーが終了タグを差し込んで本文へ移らせる (SPEC RSN-4) — **本文 0 字で
  `finish=length` にはならない**。答えのために `max_tokens` の 1/4 を
  取り置く (DEV-21)。
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
| `/slots` が 501 | `--no-slots` で建てた。`/metrics` が 501 なら `--metrics` を付けずに建てた (既定で切ってある) |
| `exit 2` + usage が出る | フラグの値が受け付けられない。**`-c/--ctx-size` と `--expert-cache-slots` は列挙で断らず、この機体が確保できる値へ下に丸める** (SPEC FLAG-2 / DEV-2) — 断られるのは 0 と負のときだけ。実効値は `/props` の `n_ctx` で読める。列挙で断るのは `--draft-block-size` (0 または 2...8) と `--prefill-chunk-tokens` |
| `exceeds this device's recommended Metal working set` | §2。スロットを 1 段下げる |
| `--draft-block-size 4 requires --prefill on` | MTP はチャンク prefill の経路を通る。`--prefill off` とは併用できない |
| `needs a model installed with the drafter section` | そのモデルにドラフターが入っていない。`TsugumiRepack --add-draft <model>` で 236 MB を追記する |
| 500 `prompt contains the special token <\|image\|>` | 貼り付けた本文にテンプレートの特殊トークンが入っている。プロンプト側で取り除く |
| 503 `model_loading` が返る | まだロード中。`/v1/models` が 200 になるまで待つ (20〜40 秒) |
| ポートが開かない / `connection refused` | プロセスが立っていないか、ロードに失敗して落ちた (§2 のガードなど)。stderr の `error:` 行を読む |
| 起動はするのに極端に遅い | 別の Tsugumi プロセスが同時に走っている (§0) |

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

## 8. Ornith (Qwen3.5-MoE) を pi につなぐ — MTP つき

§1〜§7 は Gemma 4 の話である。同じサーバーの実体で
`Ornith-1.5-35B-A3B` (`qwen3_5_moe`) も建つ ([docs/qwen35moe/26](qwen35moe/26-PHASE8-SERVER.md))。
**家族を見分けるのはバックエンドだけ**なので、HTTP から見える形は同じ。
違うのはフラグ 3 つと、できないこと 2 つである。

```bash
.build/release/TsugumiServer \
  --model scratch/ornith-oq4e-g64.moepack \
  --model-id ornith-1.5-35b-a3b \
  --port 8092 \
  --ctx-size 131072 \
  --expert-cache-slots 32 \
  --verification trusted-install \
  --draft-block-size 2 \
  --metrics
→ … family=qwen3_5_moe context=32768 slots=32 expert_io=mmap mtp=2 …
```

| 違い | 中身 |
| --- | --- |
| `--draft-block-size` | **0 か 2 のみ。**この家族のヘッドは 1 パスに 1 本しかドラフトしないので、幅は 2 で固定である ([docs/qwen35moe/40](qwen35moe/40-MTP-GRAMMAR.md) §4)。**tools と併用できる** |
| MTP ヘッドの在処 | `~/LLM/ornith-mtp-head/` の **480 MB sidecar** (`.moepack` の中ではない)。`TF_QWEN_MTP_HEAD` で差し替える。**無ければ起動しない** |
| 画像 | **400 `unsupported_image`。**Phase 9 で、`/props` の `modalities.vision` も false を返す |
| prompt cache | **ある。ただし「厳密な延長」だけ** ([docs/qwen35moe/41](qwen35moe/41-PROMPT-CACHE.md))。再帰状態は巻き戻せないので**部分再利用が無い** — 新しいプロンプトが前回の続きでなければ `cached=0` になる (Gemma のような最長共通接頭辞ではない)。ミスの理由は stderr の `prompt cache miss diverged_at=… held=…` が名指す |
| サンプラ | **公式推奨の 0.6 / 0.95 / 20 で必ず走る** ([docs/qwen35moe/42](qwen35moe/42-SAMPLING.md))。要求が別の値を送っても**上書きして実行**し、完了行に `approx="sampling/official-override: temperature=1.0→0.6 …"` と出る。公式値どおり送った要求には何も出ない。`repeat_penalty` と `seed` は使わない (これも上書きとして名前が出る) |

### コンテキスト長 — 64K までは無料、128K は decode が半分になる (**実測(手元)**)

Gemma の §2 は「コンテキストを伸ばしたらスロットを削る」だったが、この家族は
**KV が 10 層ぶんしか無い** (20,480 B/token、[docs/qwen35moe/34](qwen35moe/34-PROMPT-CACHE-ESTIMATE.md) §2-1)
ので、128K でも 32 スロットのまま**起動する**。問題は起動ではなく速度である:

| `--ctx-size` | KV | Metal peak | decode (エージェント形の会話) |
| ---: | ---: | ---: | ---: |
| 32,768 | 0.67 GB | — | 15.3 / 15.4 / 17.1 tok/s |
| 65,536 | 1.34 GB | 1.94 GB | 14.4 / 14.1 / 15.7 tok/s |
| 131,072 | 2.68 GB | 3.37 GB | **6.7 / 7.0 / 7.3 tok/s** |

同じ仕事をしている。CLI で同じプロンプトを 64K と 128K で流すと、**答えも
受理率も常駐の出入りも 1 つも変わらないのに tok/s だけ 16.70 → 8.24 になる**
(`--max-context`、a1、96 トークン)。増えているのは確保した KV だけで、
`commit` は 24.4 → 29.0 ms/tok しか動かない — 残りはカーネルの中で待っている。

**この機体の `iogpu.wired_limit_mb` は既定 8192 (= 8 GiB) で、再起動で戻る**
([[wired-limit-for-48-slots]] と同じ話)。128K の内訳は
常駐 2.18 + KV 2.50 + MTP 0.45 + 再帰 0.06 + scratch 0.25 + 常駐要求 2.11 =
**約 7.6 GiB** で上限の 94%、64K は 6.3 GiB (79%) である。

**上げると崖は消える** (2026-08-22、**実測(手元)**):

```bash
sudo sysctl iogpu.wired_limit_mb=14336     # 再起動で 8192 に戻る
```

これを立てたあと、pi の実タスク 10 ターン (文脈 2,724 → 11,712 トークン) を
128K のサーバーで流した集計が **decode 14.81 tok/s / prefill 166.3 tok/s**
(`/metrics` の累計、1,477 トークン / 99.7 秒)。**上限 8192 のときの同じ
サーバー設定は 7.5 tok/s** だったので、**倍近く戻っている** — しかも今回の方が
文脈は長い (文脈が長いほど decode は遅い、[27 §4](qwen35moe/27-PHASE6-THROUGHPUT.md))。
**ただし同一タスクの A/B は取っていない** (この 2 つは別の仕事である)。

したがって:

- **`sudo sysctl iogpu.wired_limit_mb=14336` を立てるなら 128K でよい。**
  立てないなら 32K か 64K — 8192 のままの 128K は decode が半分になる
- **再起動すると 8192 に戻る。**戻ったことに気付く方法は tok/s しかない
  (起動は通るし、`peak` も出力も変わらない)

pi 側は `~/.pi/agent/models.json` に provider を 1 つ足してある
(`local-tsugumi-ornith` → 8092、`reasoning: true`、`input: ["text"]`):

```bash
pi --provider local-tsugumi-ornith --model ornith-1.5-35b-a3b
```

**MTP はサンプリング下でも切れない** (要件 S2)。受理規則は
`u ≤ p(d)/q(d)`、棄却時は残差 `(p−q)+` からの再抽出で、**出力分布は
サンプラそのものと一致する** (42 §2-2)。受理率はほとんど落ちない
(**実測(手元)**、192 トークン、interleaved A/B 3 反復の中央値):

| タスク | 腕 | P1 (受理率) | a (トークン/パス) | tok/s |
| --- | --- | ---: | ---: | ---: |
| a1-agent-edit | greedy | 89.1% | 1.901 | 18.89 |
| a1-agent-edit | 公式サンプラ | 84.6% | 1.846 | 16.46 |
| t2-code | greedy | 81.1% | 1.811 | 17.74 |
| t2-code | 公式サンプラ | 82.9% | 1.829 | 19.34 |

**tok/s の符号は 2 本のタスクで逆になっている**ので、この n では速度差の向きを
言えない — 2 つの腕は**違うトークンを生成する**ので、同じ仕事ではないためである
(エキスパートのヒット率も 49.9% 対 51.7% と違う)。言えるのは**受理率がほぼ動かない**
ことだけで、これは温度 0.6 / top_p 0.95 の分布が十分に尖っていて
`p(d)/q(d)` が 1 に近いため。

完了行に MTP の効きと prompt cache の当たりが出る:

```
completed in 21.256s prompt=2935 cached=0 completion=160 finish=length \
  reasoning=642B mtp=2 rounds=89 accept=0.787      ← 会話の 1 ターン目
completed in  8.104s prompt=3092 cached=3056 completion=120 finish=stop \
  reasoning=331B mtp=2 rounds=63 accept=0.905      ← 続きのターン
```

**2 ターン目以降で `cached=0` が続くならクライアントが履歴を書き換えている。**
このモデルは巻き戻せないので、1 トークンでも違えば全ミスになる。よくある原因は
3 つ: assistant の思考 (`reasoning_content`) を送り返していない、文脈を切り詰めて
いる、system プロンプトに時刻や乱数 ID が入っている
([docs/qwen35moe/41](qwen35moe/41-PROMPT-CACHE.md) §4-2)。pi は思考を送り返すので
当たる。

**速度の期待値** (プロンプト約 2,936 トークン + tools + thinking、**実測(手元)**、
[docs/qwen35moe/40](qwen35moe/40-MTP-GRAMMAR.md) §5 と
[41](qwen35moe/41-PROMPT-CACHE.md) §5-2):

| 腕 | 1 ターン目の TTFT | 続きのターンの TTFT | decode |
| --- | ---: | ---: | ---: |
| `--draft-block-size 0` | 10.3 s | **0.72 s** | 13.9〜14.3 tok/s |
| `--draft-block-size 2` | 10.4 s | **0.78 s** | **16.0〜17.5 tok/s** |

短いエージェント形の会話 (275 トークンから始まる 3 ターン) は
prefill **3.19 → 1.14 → 0.58 秒**。**文脈を全部計算するのは会話の 1 回目だけ**で、
続きは追加ぶんだけを計算する。

- `TF_EXPERT_MMAP_RESIDENCY_ASYNC=1` を付けると素の decode も MTP も速くなる
  ([docs/qwen35moe/39](qwen35moe/39-RESIDENCY-COMMIT.md))。**既定 off、
  Gemma と共有の経路**なので付けるかはユーザー判断。
- Gemma のサーバーと**同時には建てられない** (§0)。ポートを分けてあるのは
  設定を残しておくためで、プロセスは 1 つずつ。
