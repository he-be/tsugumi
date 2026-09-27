# 16. MBP の Tsugumi GUI から、M6 単体の ishizuki を使う — 検索などのエージェント動作の計画 (2026-09-27)

**状態: 計画。新しい実測は無い。**[15](15-M6-PREFILL-PLAN.md) §7-5 で ishizuki (MLX Swift) が M6 で Bonsai 2 27B の prefill を 3 経路中最速で回したのを受け、
ユーザーが 2026-09-27 に「MBP で Tsugumi の GUI を出し、中身は M6 単体の ishizuki で検索等のエージェント動作をさせる」と決めた。この文書はその形と、作るもの・段・合否・決めてもらうことを書く。
表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ (実測 / 導出 / 未確認)。コードを読んだ結果は「読んだ」と書き、動かしていない。

言葉: **UI 側** = MBP の Tsugumi アプリ (会話・ツールの実行・Web の取得・表示)。**モデル側** = M6 で常駐する ishizuki のサーバ (Bonsai 2 27B の MLX 2-bit パック、text だけ)。
**ラウンド** = アプリが 1 ターンの中でモデルに投げる 1 回の要求 ([qwen38/20](../qwen38/20-WEB-TOOLS-WIRING.md) のツールループの単位)。

## 0. 結論

1. **形は「子サーバ経路」([11](11-GUI-WIRING.md)) のサーバを M6 の ishizuki に置き換えたもの。**アプリのツールループ・システムプロンプト・Online の方針・ラウンド予算は今のまま UI 側に残り、
   モデル側は OpenAI 互換の HTTP だけを喋る。モデルのプロセスは M6 に 1 つ、MBP には無い (§2)。
2. **ishizuki の API は Tsugumi の遠隔クライアントが頼っている口の半分を持たない** (§1-2、読んだ)。無いのは `/v1/completions`・`/apply-template`・`/props`・`/tokenize`・`grammar`・`logit_bias`・`tool_choice`・要求ごとの `top_k` / `top_p` / `min_p` / `presence_penalty`。
   ある のは `/v1/chat/completions` (stream、`tools`、`response_format` の json_schema、`chat_template_kwargs.enable_thinking`)・`/v1/messages`・`/health`・`/v1/models`。
   したがって **`auto` のラウンドはそのまま通り、`function` / `required` (強制呼び出し) と `none` (呼び出し禁止) のラウンドは通らない。**ここを埋めるのが仕事の中心 (§3 S3)。
3. **埋め方は 2 つあり、推奨は ishizuki 側に小さく足す** (§6-1)。`none` は [qwen38/26](../qwen38/26-NONE-IS-A-CONSTRAINT.md)・[27](../qwen38/27-NONE-CONSTRAINT-DONE.md) の通りサンプラの拘束で、プロンプトの言い回しで代えない。
   ishizuki には `OutputConstraint` (読んだ) があるので、`logit_bias` の −100 を禁止トークンとして受ける口と、強制呼び出しのための素の prompt 継続 (`/v1/completions` 相当)、
   要求ごとのサンプリング 4 項目を受ける口を、`bench/m6/ishizuki-driver/` にパッチとして置き、`ishizuki_ship.sh` が当てる。アプリのラウンドの形は変えない ([11 §6](11-GUI-WIRING.md) の 14 ターン × 2 セットで検証済みの形を保つ)。
4. **M6 はヘッドレスなので ishizuki のメニューバーアプリは使えない。**`APIServer(directory:)` + `listen(port:)` を呼ぶ 30 行の実行ファイル (`ishizuki-serve`) を、[15 §3-3](15-M6-PREFILL-PLAN.md) の driver と同じ道で MBP で組んで運ぶ (§3 S1)。
5. **経路は ssh のポート転送で、M6 のポートを LAN に出さない** (§2-2)。ishizuki の `HTTPServer` は `NWListener` を全インタフェースに bind し、認証も TLS も無い (読んだ)。
   MBP から `ssh -N -L 8128:127.0.0.1:8128 m6` を張り、アプリは `http://127.0.0.1:8128` を叩く。[OPENAI_SERVER](../OPENAI_SERVER.md) の「127.0.0.1 から出さない」と同じ扱いになる。
6. **メモリは text だけなら収まる見込み、画像は入れない** (§4)。2.4K トークンの prefill で peak 9.43 GB (15 §7-5、実測)。KV fp16 64 KB/トークン (ishizuki README) で 32K なら 2 GB (導出)。
   vision の塔を読むと R2 で 14.44 GB になった (15 §7-4) ので、この計画では画像を送らない。
7. **合否はアプリの検査で取る。**`TsugumiToolLoopCheck --endpoint` の Offline 5 問 × 2 セット (answer 28/28 が [11 §6](11-GUI-WIRING.md) の基準) → Online 6 会話 × 3 ターン → GUI。速さの記録値は M6 の証拠ディレクトリの規則で取る (§3 S6)。

## 1. 前提と事実

### 1-1. 手元にあるもの (実測、2026-09-27)

| もの | 場所 | 状態 |
| --- | --- | --- |
| ishizuki の prefill driver (MBP で組んだ) | `m6:~/LLM/ref/ishizuki-bin/` | 動く。同じ本文 2,427 トークンで 9.205 s、peak 9.43 GB、Swapouts 0 ([15 §7-5](15-M6-PREFILL-PLAN.md)) |
| MLX 2-bit パック | `m6:~/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit/` (8.6 GB) | `config.json` schema 2、`components.vision = true`、MTP 無し |
| 組んで運ぶ道 | [`bench/m6/ishizuki_ship.sh`](../../bench/m6/ishizuki_ship.sh) | tools-version 6.4 → 6.3、FoundationModels の 1 ファイル除外、Swift Build `--arch arm64`、MBP の Metal Toolchain。実行ターゲットを足す形はこのスクリプトが持っている |
| M6 の内蔵の空き | — | 37 GB (PQ2_0・prism フォーク・venv・ishizuki の checkout は 2026-09-27 に消した) |
| アプリの子サーバ経路 | [11](11-GUI-WIRING.md) | 種別 `.bonsai27b` (`AppModelKind`) が `LlamaServerInferenceClient` (子プロセスの起動・停止) → `RemoteInferenceClient` (OpenAI 互換) で動く。14 ターン × 2 セット完走 |
| 検査の口 | `TsugumiToolLoopCheck --endpoint URL --remote-model ID [--remote-direct]` | 既存。`--remote-direct` は llama-swap でなく素の llama-server を指す印 |

### 1-2. ishizuki のサーバの口 (読んだ、`Sources/IshizukiKit/Serve/APIServer.swift`、459ee00)

| 口 | ある / 無い | 備考 |
| --- | --- | --- |
| `POST /v1/chat/completions` | ある | `stream`、`tools` (function の中身を取り出す)、`tool` ロールのメッセージ、`response_format` の json_schema (このとき thinking は強制 off)、`max_tokens` (既定 1024)、`temperature` |
| `POST /v1/messages`、`/v1/messages/count_tokens` | ある | Anthropic 形。この計画では使わない |
| `GET /health`、`GET /v1/models` | ある | 生存確認と ID |
| `grammar` (GBNF) | 無い | 送ると `GBNF grammars are not supported; send response_format with a json_schema instead` で拒否 |
| `logit_bias`、`tool_choice` | 無い | 文字列が出てこない |
| `POST /v1/completions`、`/apply-template`、`/props`、`/tokenize` | 無い | ルートは上の 5 つだけ |
| 要求ごとの `top_k` / `top_p` / `min_p` / `presence_penalty` | 無い | 読むのは `temperature` だけ。他はサーバ起動時の `samplingOptions` (`SamplingOptions` に temperature / topP / topK / minP / repetitionPenalty / presencePenalty / seed はある) |
| thinking の切り替え | ある | `thinking: Bool`、`chat_template_kwargs.enable_thinking`、`reasoning_effort` のどれか。既定はサーバ起動時の `thinking:` |
| 会話の cache | ある | `SessionCache` (スロット数はメモリ予算の段で決まる) と `PrefixStore` (ディスク、既定 8 GB、2,048 トークン以上の接頭辞を保存)。要求の `messages` の共通接頭辞で当てる (`prefixTokens`、読んだ)。当たり方の実測は無い |
| bind | 全インタフェース | `NWListener(using:on:)` にホストの指定が無い。認証は companion (iPhone) の PSK/TLS だけで、HTTP の API には無い |

### 1-3. Tsugumi の遠隔クライアントが頼っている口 (読んだ、`RemoteInferenceClient.swift`)

| ラウンド (`toolChoice`) | 使う口 | ishizuki で |
| --- | --- | --- |
| `auto` | `/v1/chat/completions` (stream、`tools`、`chat_template_kwargs.enable_thinking`、`top_k` 等 4 項目) | **通る。**ただし 4 項目は無視され、サーバ既定のサンプリングになる |
| `function` / `required` (強制呼び出し) | `/apply-template` で描いた prompt に `<tool_call>\n<function=名前>\n` を足し、`/v1/completions` で閉じ印まで継続、アプリの `QwenToolCallParser` で読む | **通らない** (`/apply-template` も `/v1/completions` も無い) |
| `none` (呼び出し禁止) | chat completions に `logit_bias` で呼び出しの開始トークンを禁止 | **通らない** (`logit_bias` が無い)。プロンプトの言い回しで代えない ([qwen38/26](../qwen38/26-NONE-IS-A-CONSTRAINT.md)) |
| 拘束付き生成 (`constrained`) | `/v1/completions` + `grammar` | **通らない。**json_schema で書けるものは `response_format` に移せる (未確認: 今の grammar が全部 schema に落ちるか) |
| 読み込み時 | `/props` の `default_generation_settings.n_ctx` | **通らない。**文脈長は種別の固定値で持つ |

サンプリングは種別の公式値で固定 ([AppModelKind](../../Sources/TsugumiApp/Core/Configuration/AppModelKind.swift) の `.bonsai27b`: thinking off で temperature 0.7 / top_k 20 / top_p 0.8 / min_p 0.0 / presence 1.5、
thinking on で 1.0 / 20 / 0.95 / 0.0 / 0.0)。パックの `generation_config.json` は 1.0 / 20 / 0.95 (thinking on の行と同じ)。**値を変えない・疑わない。**

## 2. 形

### 2-1. 部品

| | 置き場 | 役割 | 状態 |
| --- | --- | --- | --- |
| Tsugumi アプリ | MBP | 会話・ツールループ・Web の取得と要約・表示・設定。種別 `.bonsaiIshizuki` (仮) を足す | 種別と client の追加が要る (S3) |
| `ssh -N -L 8128:127.0.0.1:8128 m6` | MBP | 経路。アプリが張るか、ユーザーが張る | 未定 (§6-4) |
| `ishizuki-serve` | M6 `~/LLM/ref/ishizuki-bin/` | `APIServer(directory:…)` を起動して `listen(port: 8128)`。パック・サンプリング・thinking の既定・KV・PrefixStore の置き場を引数で受ける | 未作成 (S1) |
| launchd (M6) | M6 | `ishizuki-serve` の常駐 (KeepAlive)。再起動後も待ち受ける | 未作成 (S1) |
| MLX パック | M6 | 重み。M6 に置いたまま (測定対象でなく常用なので [TWO_MACHINE_DEV §5](../TWO_MACHINE_DEV.md) の「置かない」の例外にする。決めてもらう、§6-5) | ある |

### 2-2. 経路の決まり

- **ポートは M6 の 127.0.0.1 の外に出さない。**ishizuki は全インタフェースに bind するので、M6 側の `pf` で 8128 を loopback 以外から落とすか、
  Tailscale の ACL で塞ぐ。最も単純なのは MBP からの ssh 転送だけを通し、M6 の 8128 は外から見えない前提を `pf` で担保すること (未実施、S1 の合否に入れる)。
- アプリから見た endpoint は `http://127.0.0.1:8128`。[OPENAI_SERVER](../OPENAI_SERVER.md) の Tsugumi 自身のサーバと同じ場所で、混同しないようポートを分ける (Tsugumi のサーバは別ポート)。
- モデルのプロセスは M6 の 1 つだけ。アプリはこの種別のとき decode service を起動しない (子サーバ経路と同じ、[11 §1](11-GUI-WIRING.md))。

## 3. 作るもの (段)

| 段 | 作るもの | 合否 |
| --- | --- | --- |
| **S1** M6 の常駐 | `ishizuki-serve` (30 行、`bench/m6/ishizuki-driver/serve.swift`)。`ishizuki_ship.sh` に実行ターゲットを 1 つ足す。launchd の plist (`~/Library/LaunchAgents`、ヘッドレスなので LaunchDaemon か、ssh ログインのユーザーで `launchctl bootstrap gui/…` が無い点を確かめる)。`pf` の規則 | MBP から転送越しに `GET /health` と `GET /v1/models` が返り、M6 の再起動後も返る。M6 の LAN の IP に直接叩くと繋がらない |
| **S2** API の照合 | curl だけの検査 `bench/m6/ishizuki_api_check.sh`: chat completions (stream あり/なし)、`tools` + `tool` ロールの往復、`response_format` json_schema、`enable_thinking` on/off、`max_tokens`。各応答の形を残す | §1-2 の「ある」が全部動く。thinking off のとき `<think>` の空ブロックが出るか (本文に混ざらないか) を確かめる |
| **S3** 埋める | (a) ishizuki への足し込み: `logit_bias` (−100 → 禁止トークン、`OutputConstraint` へ)、素の prompt 継続 (`/v1/completions` の `prompt` + `stop`)、要求ごとの `top_k` / `top_p` / `min_p` / `presence_penalty`。パッチは `bench/m6/ishizuki-driver/*.patch` に置き `ishizuki_ship.sh` が当てる。(b) Tsugumi 側: 種別 `.bonsaiIshizuki`、`RemoteInferenceClient.Routing` に `.ishizuki` を足し、`/apply-template` の代わりに **アプリが持つ chat template で描く** (`QwenToolCallParser` と同じ側にテンプレートが要る。未確認: アプリが Qwen のテンプレートを持っているか。無ければ `/v1/completions` に `messages` を描かせる口を (a) に足す)、`/props` を読まず固定の文脈長 | 3 つのラウンドの形が全部通り、`TsugumiToolLoopCheck --endpoint http://127.0.0.1:8128 --remote-model ternary-bonsai-2-27b` の Offline 5 問が 1 セット通る |
| **S4** 検査 | Offline 5 問 × 2 セット、Online 6 会話 × 3 ターン (`docs/qwen38/48` の 6 会話、Serper 0 回)。記録は `scratch/bonsai27b/runs/` の形 ([11 §6](11-GUI-WIRING.md)) | answer が 11 §6 と同じ 28/28。ターンの秒は同じ形で並べる (機械が違うので比は書かない) |
| **S5** GUI | 種別を選ぶ欄に出す。有効条件は転送越しの `/health` が返ること (M6 ボタンの「居れば使う」[qfn-prefill/33](../qfn-prefill/33-OPERATING-FORM-PLAN.md) と同じ考え)。HUD は client 側の計測 (TTFT・tok/s) で、モデル側のメモリは `/health` に無ければ出さない | GUI から 1 会話 3 ターンが通る |
| **S6** 記録値 | 1 ターンの秒・TTFT・decode tok/s を、M6 の証拠の規則 ([m6-prefill/02](../m6-prefill/02-EVIDENCE.md)) で `bench/m6/results/` に残す。本文はアプリの会話そのもの | n ≥ 3。解釈は中央値とレンジで |

順は S1 → S2 → S3(a) → S3(b) → S4 → S5 → S6。S2 で「ある」が動かなければ S3 の前に止めて報告する。

## 4. M6 のメモリと常駐

| 項目 | 値 | 札 |
| --- | --- | --- |
| 重み (text) + 2.4K の prefill の peak | 9.43 GB | 実測 ([15 §7-5](15-M6-PREFILL-PLAN.md)) |
| KV fp16 | 64 KB/トークン → 32K で 2.0 GB | 公称 (ishizuki README の表) → 導出 |
| `iogpu.wired_limit_mb` 既定 0 | 約 12 GB (RAM の 75%) | 公称 (同 README)。M6 は今 14336 だが再起動で 0 に戻る |
| vision の塔ごと読んだ peak (R2) | 14.44 GB | 実測 ([15 §7-4](15-M6-PREFILL-PLAN.md)、mlx-vlm 側の値。ishizuki は塔を最初の画像要求で遅延読み込みする、読んだ) |
| PrefixStore | 既定 8 GB、内蔵に置く | 読んだ。空きは 37 GB |

- 32K を常用点にすると 9.4 + 2.0 = 11.4 GB (導出) で、既定の 12 GB の中に入る見込み。**入るかどうかは S1 で `/health` の後に 32K の会話を 1 本流して Swapouts で確かめる。**
- 画像は送らない。塔を読むと 16 GB を超える (R2 の 14.44 GB + KV) 見込みなので、この種別ではアプリの画像添付を無効にする。
- `SessionCache` のスロット数はメモリ予算の段 (`MemoryBudget.tier`) が決める (読んだ)。会話を切り替えたときに prefix が当たるかは S4 の記録 (`cachedPromptTokens` 相当が API に無いので、TTFT で見る)。

## 5. やらないこと

- ishizuki を Tsugumi に移植しない。ishizuki のメニューバーアプリを M6 で動かさない (ヘッドレス)。
- M6 の 8128 を LAN や Tailscale に直接出さない。
- サンプリングの値を変えない。ishizuki の既定 (0.7、warper 全部 off) は使わず、種別の公式値を起動時と要求時に渡す。
- M6 で 2 つ目のモデルプロセスを立てない。Gemma の参照ランタイム (`~/LLM/ref/llama.cpp`、`mlx-venv`) はディスクに残るだけで起動しない。
- MTP・投機は無し (パックに MTP が無い。ishizuki の prompt lookup 投機は既定 on で、S2 で効いているかだけ見る。切らない)。

## 6. 決めてもらうこと

1. **S3 の埋め方**: ishizuki 側に足す (推奨、§0-3) か、アプリのラウンドの形を ishizuki の口に合わせて変える (`none` を `tools` 無しで送る形は [qwen38/26](../qwen38/26-NONE-IS-A-CONSTRAINT.md) で退けた形なので勧めない) か。
2. **thinking の既定**: 子サーバ経路は thinking off ([11](11-GUI-WIRING.md))。同じにするか。
3. **画像**: この種別では無効 (§4)。
4. **ssh 転送を誰が張るか**: アプリが起動時に張る (鍵・ホスト名の設定が要る) か、ユーザーがターミナルで張る (最初はこちら)。
5. **パックを M6 に置いたままにすること** ([TWO_MACHINE_DEV §5](../TWO_MACHINE_DEV.md) の例外)。
6. **文脈長の常用点**: 32K から始める (§4)。

## 7. この計画で使わない・参照だけのもの

- [10](10-GUI-PLAN.md)・[11](11-GUI-WIRING.md): 子サーバ経路の設計と結線。種別・client・検査の形はここから引き継ぐ。
- [12](12-NATIVE-PORT-GAIN.md) §1: 1 ターンの 53〜56% が prefill で新規トークンが 98%。ishizuki の prefill が llama.cpp より短い (15 §7) のが、この計画の動機。decode は測っていない (S6 で出る)。
- [15](15-M6-PREFILL-PLAN.md): prefill の 3 経路の実測と、組んで運ぶ道。
