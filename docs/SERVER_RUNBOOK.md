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

**どれを選ぶかは「コンテキスト長」だけで決まる。**長いコンテキストを取ると
KV キャッシュがメモリを食い、その分だけ expert キャッシュのスロットを削る必要がある
(理由は §2)。**スロットが減ると生成も遅くなる** (§4)。

### (a) 16K — 既定。いちばん速い。pi の常用はこれ

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4-qat.gturbo \
  --port 8091 \
  --max-context 16384 \
  --expert-cache-slots 80 \
  --verification trusted-install \
  --draft-block-size 4
```

### (b) 64K — 長い会話やログを丸ごと貼るとき

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4-qat.gturbo \
  --port 8091 \
  --max-context 65536 \
  --expert-cache-slots 64 \
  --verification trusted-install \
  --draft-block-size 4
```

### (c) 128K — 最長。生成は (a) より 15〜20% 遅い

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4-qat.gturbo \
  --port 8091 \
  --max-context 131072 \
  --expert-cache-slots 48 \
  --verification trusted-install \
  --draft-block-size 4
```

**ポートは 8091。**`~/.pi/agent/models.json` の `local-turbofieldfare` が
`http://127.0.0.1:8091/v1` を向いている。

バックグラウンドに置いてログを残す形:

```bash
nohup .build/release/TurboFieldfareServer --model scratch/gemma4-qat.gturbo \
  --port 8091 --max-context 65536 --expert-cache-slots 64 \
  --verification trusted-install --draft-block-size 4 > /tmp/tf-server.log 2>&1 &
```

## 2. なぜコンテキストごとにスロット数が変わるのか (**実測**)

起動時のガードが「常駐させる合計」をこの機体の Metal 推奨作業セット
**12.88 GB** と比べ、超えていたら**ポートを開く前に exit 2** で落ちる。
内訳は実測でこう出る:

| コンテキスト | KV | resident + vision + scratch | 使えるスロット |
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

起動には 20〜40 秒かかる (モデルの検証と mmap)。**ポートが開くのはロード後**なので、
`/v1/models` が返れば準備完了である。

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
サーバーを建ててから pi を起動するだけでよい。**既知の制限が 2 つある**:

| 制限 | 中身 | 回避 |
| --- | --- | --- |
| 対話モードで画像が使えない | pi は毎リクエストに built-in tools を宣言し、サーバーは「画像 + tools」を 400 で拒否する | 画像を見せたいセッションだけ `pi -nt` (tools 無効) で起動する |
| Reasoning が使えない | サーバーに `--thinking` 相当が無く、tools 宣言時はテンプレートが `enable_thinking: false` を固定する | CLI (`--messages-file` + `--thinking on`) を使う |

どちらも [TODO.md](../TODO.md) に調査済みの項目として残してある。

## 6. よくある失敗

| 症状 | 原因と対処 |
| --- | --- |
| `exit 2` + usage が出る | フラグの値が許可リストの外。`--max-context` は 4096 / 8192 / 16384 / 32768 / 65536 / 131072、`--expert-cache-slots` は 8/16/24/32/48/64/80/96/112、`--draft-block-size` は 0 または 2...8 |
| `exceeds this device's recommended Metal working set` | §2。スロットを 1 段下げる |
| `--draft-block-size 4 requires --prefill on` | MTP はチャンク prefill の経路を通る。`--prefill off` とは併用できない |
| `needs a model installed with the drafter section` | そのモデルにドラフターが入っていない。`TurboFieldfareRepack --add-draft <model>` で 236 MB を追記する |
| 500 `prompt contains the special token <\|image\|>` | 貼り付けた本文にテンプレートの特殊トークンが入っている。プロンプト側で取り除く |
| ポートが開かない / `connection refused` | まだロード中。`/v1/models` が返るまで待つ (20〜40 秒) |
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
