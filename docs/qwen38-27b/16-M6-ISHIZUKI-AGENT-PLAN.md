# 16. MBP の Tsugumi GUI から、M6 単体の ishizuki を使う — 検索などのエージェント動作の計画 (2026-09-27)

**状態: S1 は再起動後の常駐を除いて通過 (§9)。GUI から手で使える (§12)。MTP は未着手で、別セッションで扱う (§13)。S2 は 12 件中 10 件 (§9)。S3 は通過: アプリの形のラウンドが全部通り、Offline 5 問 × 2 セットが 10/10・Swapouts +0 (§10)。S4 の Online は、1 セット目を 8 ターン目で見張りが止めた (§11) が、M6 の再起動後に流し直した 1 セットは 14 ターン完走・Swapouts +0 (2026-09-27、§11-2)。**[15](15-M6-PREFILL-PLAN.md) §7-5 で ishizuki (MLX Swift) が M6 で Bonsai 2 27B の prefill を 3 経路中最速で回したのを受け、
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
| MLX 2-bit パック | `m6:/Volumes/SSD256/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit/` (8.0 GB、外付け。`~/LLM/ref/` の同名はシンボリックリンク、2026-09-27 に移動) | `config.json` schema 2、`components.vision = true`、MTP 無し |
| 組んで運ぶ道 | [`bench/m6/ishizuki_ship.sh`](../../bench/m6/ishizuki_ship.sh) | tools-version 6.4 → 6.3、FoundationModels の 1 ファイル除外、Swift Build `--arch arm64`、MBP の Metal Toolchain。実行ターゲットを足す形はこのスクリプトが持っている |
| M6 の内蔵の空き | — | 45 GB (PQ2_0・prism フォーク・venv・ishizuki の checkout を消し、パックを外付けへ移した後。外付け SSD256 の空きは 87 GB) |
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
| MLX パック | M6 の外付け (`/Volumes/SSD256/LLM/ref/`) | 重み。起動時に 1 回読むだけなので外付けに置く (ユーザー決定、§6-5)。内蔵には置かない | ある (移動済み) |

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
- MTP・投機は無し (パックに MTP が無い)。**訂正 (§13)**: 「prompt lookup 投機は既定 on」は誤りで、公式値の presence 1.5 のもとでは prompt lookup も含めて投機は動いていない。この「無し」は測って決めたものではない。

## 6. 決めてもらうこと

1. **S3 の埋め方**: ishizuki 側に足す (推奨、§0-3) か、アプリのラウンドの形を ishizuki の口に合わせて変える (`none` を `tools` 無しで送る形は [qwen38/26](../qwen38/26-NONE-IS-A-CONSTRAINT.md) で退けた形なので勧めない) か。
2. **thinking の既定**: 子サーバ経路は thinking off ([11](11-GUI-WIRING.md))。同じにするか。
3. **画像**: この種別では無効 (§4)。
4. **ssh 転送を誰が張るか**: アプリが起動時に張る (鍵・ホスト名の設定が要る) か、ユーザーがターミナルで張る (最初はこちら)。
5. **パックを M6 に置いたままにすること** ([TWO_MACHINE_DEV §5](../TWO_MACHINE_DEV.md) の例外)。**決まった (2026-09-27)**: 置いたままにするが内蔵ではなく外付け SSD256 に置く。「初回ロードするだけ」なので内蔵の空きを使わない。移動済み (§8)。
6. **文脈長の常用点**: 32K から始める (§4)。

1・2・3・4・6 はユーザーから「何を決めるの？」と返ったので、推奨側 (ishizuki 側に足す・thinking off・画像なし・ssh 転送は最初は手で・32K) で進める。異論が出た段で戻す。

## 7. この計画で使わない・参照だけのもの

- [10](10-GUI-PLAN.md)・[11](11-GUI-WIRING.md): 子サーバ経路の設計と結線。種別・client・検査の形はここから引き継ぐ。
- [12](12-NATIVE-PORT-GAIN.md) §1: 1 ターンの 53〜56% が prefill で新規トークンが 98%。ishizuki の prefill が llama.cpp より短い (15 §7) のが、この計画の動機。decode は測っていない (S6 で出る)。
- [15](15-M6-PREFILL-PLAN.md): prefill の 3 経路の実測と、組んで運ぶ道。

## 8. 進捗 (2026-09-27、S1 着手)

やったこと (実測 / 読んだ):

- **パックを外付けへ**: `m6:~/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit` (8.0 GB) を `/Volumes/SSD256/LLM/ref/` に `mv` し、旧パスにシンボリックリンクを置いた。
  [`ref_ishizuki_bonsai.sh`](../../bench/m6/ref_ishizuki_bonsai.sh) の既定 `MLX_PACK` はリンク越しにそのまま通る (未確認: 移動後にまだ 1 本も流していない)。内蔵の空き 37 → 45 GB、SSD256 の空き 95 → 87 GB。
- **M6 の前提を確かめた** (ssh `m6-lan`、実測): macOS 27.0、RAM 16 GB、`iogpu.wired_limit_mb` = 14336 (再起動で 0 に戻る)、モデルのプロセス無し。
  `~/Library/LaunchAgents` はあり (Google の 1 本)、`/Library/LaunchDaemons` は tailscaled だけ。**ログインユーザー `mh` は admin だが `sudo -n` はパスワード要求** → `pf` の規則と LaunchDaemon はこのセッションからは置けない。
  S1 の「M6 の LAN の IP に直接叩くと繋がらない」は、`pf` を使うならユーザーが `sudo` で入れるか、`ishizuki-serve` 側で bind を 127.0.0.1 に限る (下の読んだ結果の通り `HTTPServer` はホストを渡せないので、`NWListener` の `requiredLocalEndpoint` を足すパッチが要る) かの二択。**推奨は後者** (ssh 転送だけを通し、`pf` に頼らない)。
- **ishizuki の `APIServer` を読んだ** (`Sources/IshizukiKit/Serve/APIServer.swift`、459ee00)。`ishizuki-serve` の中身になる口:
  - `APIServer(directory:modelName:thinking:samplingOptions:kvConfig:residency:politeness:ropeScaling:budget:prefixStore:catalog:catalogRoots:preload:hot:)` → `listen(port:)`。`preload: true` が既定で init の中で重みを読む。`stop()` が `sessions.persistAll()` を呼ぶので、launchd で止めるときは SIGTERM を受けて `stop()` を呼ぶ形にする。
  - `log: (@Sendable (String) -> Void)?` にログが来る (cache の当たり `cache: reused N of M prompt tokens`、モデル読み込み秒、budget の段)。**cache の当たりは API の応答には無い** (`usage` は prompt_tokens / completion_tokens だけ、stream には `usage` が無い) → S4 で当たりを見るのはこのログか TTFT。
  - `/health` は `{"status":"ok","model":名前}` だけ。メモリは無い (S5 の HUD の前提通り)。
  - 要求ごとに読むサンプリングは `temperature` のみ (§1-2 の通り)。`Request` には `tag` (会話の札) があるが `parseOpenAI` は埋めない → S3(a) で `logit_bias` 等と一緒に読む候補。
  - `SamplingOptions(temperature: 0.7, topP: 1.0, topK: 0, minP: 0.0, repetitionPenalty: 1.0, repetitionContext: 64, presencePenalty: 0.0, seed:)`。`.bonsai27b` の thinking off の公式値は 0.7 / 20 / 0.8 / 0.0 / presence 1.5 なので、起動時の `samplingOptions` にこれを渡す。
  - `HTTPServer(port:psk:advertise:handler:)` は `NWParameters.tcp` で `NWListener(using:on:)`、ホスト指定無し (§0-5 の通り)。
- **ishizuki の checkout は MBP の `scratch/ishizuki` にある** (459ee00、ship スクリプトが置いたもの)。S3(a) のパッチはここで作って `bench/m6/ishizuki-driver/*.patch` に出す。

## 9. S1 と S2 の結果 (2026-09-27、実測)

### 9-1. S1: `ishizuki-serve` を組んで運び、手で起動した

- **作ったもの**: [`bench/m6/ishizuki-driver/serve.swift`](../../bench/m6/ishizuki-driver/serve.swift) (`APIServer` を立てて `listen` する実行ファイル)、
  [`bind-host.patch`](../../bench/m6/ishizuki-driver/bind-host.patch) (`HTTPServer` に `host:` を足し、`requiredLocalEndpoint` でその番地だけで待ち受ける。`APIServer.listen(port:host:)`)。
  [`ishizuki_ship.sh`](../../bench/m6/ishizuki_ship.sh) がパッチを当て、`ishizuki-prefill` と `ishizuki-serve` の 2 本を組んで運ぶ形にした。
- **既定値**: ishizuki アプリの `ServerSettings.init` と同じ (KV 3.5 bit / 窓 128、idle 120 s、アンロードしない、politeness normal、PrefixStore 8 GB を `~/Library/Application Support/Ishizuki/cache/prefixes`、preload あり、vision の塔は遅延読み込み)。
  サンプリングは `.bonsai27b` の公式値で、`--thinking off` (既定) なら 0.7 / 20 / 0.8 / 0.0 / presence 1.5、`on` なら 1.0 / 20 / 0.95 / 0.0 / 0.0。要求で上書きできるのは temperature だけ (§1-2)。
  **§4 の KV fp16 の見積もりは当たらない**: ishizuki の既定は 3.5 bit なので、32K の KV は fp16 の 2.0 GB より小さい (導出。実測はまだ)。
- **起動** (M6、手で `nohup`、ログは `m6:~/LLM/ref/ishizuki-logs/serve.log`): `ready ... in 0.5 s`、`listening on 127.0.0.1:8128`。起動直後の RSS は 6.2 GB、Swapouts は起動前から 113 MiB のまま。
- **合否** (MBP から): `ssh -f -N -L 8128:127.0.0.1:8128 m6-lan` 越しに `/health` → `{"status":"ok","model":"ternary-bonsai-2-27b"}`、`/v1/models` も返る。
  M6 の LAN の IP (192.168.0.64:8128) を直接叩くと接続拒否 (curl exit 7)。M6 の `lsof` は `TCP 127.0.0.1:8128 (LISTEN)` だけ。`pf` は使わない。
- **再起動後の常駐は未了**: M6 には GUI セッションも自動ログインも無い (`launchctl print gui/501` が失敗、`autoLoginUser` なし) ので、LaunchAgent では再起動後に起動しない。
  `UserName mh` の LaunchDaemon (`/Library/LaunchDaemons`、sudo が要る) が必要。このセッションからは plist も置けなかった (auto mode が常駐化として止めた)。ユーザーの判断待ち。
- **32K の会話を 1 本流して Swapouts を見る**のはまだ。

### 9-2. S2: API の照合 ([`bench/m6/ishizuki_api_check.sh`](../../bench/m6/ishizuki_api_check.sh))

2 回流して同じ結果 (`scratch/bonsai27b/ishizuki-api/20260927-120937`・`-121020`)。12 件中 10 件通過。

| 件 | 結果 | 中身 |
| --- | --- | --- |
| `/health`・`/v1/models` | 通る | |
| chat (stream なし / あり) | 通る | 本文に `<think>` が混ざらない |
| `max_tokens` 8 | **落ちる** | 8 トークンで止まるが、`finish_reason` が `length` にならず `stop` のまま (非 stream の応答は tool_calls の有無しか見ていない、`APIServer.swift` 637 行、読んだ)。アプリは `finish == "length"` を `.maxTokens` として扱う (`RemoteInferenceClient.swift` 219 行) ので、打ち切りを終端と読み違える |
| `tools` の呼び出し・`tool` ロールの往復・stream の `delta.tool_calls` | 通る | |
| `response_format` json_schema (制約なしの string) | **落ちる** | 400 `a string without an enum admits unboundedly many values; constrained decoding needs an enum, a bounded integer range, or a boolean` |
| 同 (enum の string) | 通る | `{"capital":"パリ"}` |
| `enable_thinking` false / true | 通る | off は `reasoning_content` なし。on は `reasoning_content` に分かれ、本文に 391 |

- **S3(a) に足すもの** (§3 の 3 つに加えて): `finish_reason` の `length` (非 stream と stream の両方)。
- **§1-3 の「拘束付き生成は json_schema に移せる」は、ishizuki では有限の値の集合 (enum・範囲付き整数・真偽) だけ**。自由な文字列を含む grammar は移せない。アプリは今 `response_format` を使っていない (grep で 0 件) ので、`constrained` のラウンドは S3 で `/v1/completions` + 何らかの拘束として扱う必要がある (未確認: `constrained` のラウンドがこの種別で出るか)。
- thinking off で空の `<think>` ブロックは本文に出ない (§3 S2 の確認項目)。
- cache: 同じ接頭辞の 2 回目で `cache: reused 303 of 340 prompt tokens` がログに出る (tools の往復)。応答には当たりの数は出ない (§8 の通り)。

いま M6 の `ishizuki-serve` (pid 9744、手で起動) と MBP の ssh 転送は動いたまま。

## 10. S3 の結果 (2026-09-27、実測)

### 10-1. 埋め方: ishizuki に llama-server と同じ形の口を足した

§3 S3 の (a) を広げ、アプリの `RemoteInferenceClient` が `.direct` で叩く口を ishizuki 側に全部そろえた。**アプリ側 (S3(b)) は変えていない**。
`Routing.direct` と `Dialect.qwen` のまま、`/apply-template`・`/v1/completions`・`/tokenize`・`/props` がそのまま通る。§3 S3(b) の「アプリが持つ chat template で描く」は要らなくなった。
種別の追加 (`.bonsaiIshizuki`) は GUI の段 (S5) でやる。

[`bench/m6/ishizuki-driver/api.patch`](../../bench/m6/ishizuki-driver/api.patch) (ishizuki 459ee00 に当てる。`bind-host.patch` とはどちらの順でも当たる):

| 足したもの | 中身 |
| --- | --- |
| `SamplingOptions.bannedTokens` | Sampler が logits を −inf にしてから warper に渡す。素の経路・パイプライン・prompt lookup の検証の全部に効く。enum 拘束の経路では許す集合から外す |
| 要求ごとのサンプリング | `top_k`・`top_p`・`min_p`・`presence_penalty`・`seed` を chat と completions の両方で読む。無ければサーバの値 (起動時の公式値) |
| `logit_bias` | `[[id, false]]`・`[[id, -100]]`・`{"id": -100}` を禁止として読む。禁止以外の値は 400 で断る (黙って無視しない) |
| `finish_reason` の `length` | 生成が `max_tokens` (または文脈の上限) で尽きたときだけ。終端トークン・stop 文字列・拘束の完了は `stop` |
| `timings` | llama-server の形 (`cache_n`・`prompt_n`・`prompt_ms`・`predicted_n`・`predicted_ms`・`predicted_per_second`)。非 stream の応答と stream の最後の chunk に付ける。アプリの診断 (cached・prefill 秒・tok/s) がこれで埋まる |
| `generation_settings` | 実際に使ったサンプリングを返す (llama-server の `/completion` と同じ名前)。検査用 |
| `POST /v1/completions` | `prompt` をテンプレートを通さずそのまま続ける。`stop` (文字列か配列) で切り、stop 以降は返さない。stream は断る |
| `POST /apply-template` | chat の要求を描いた prompt を返す (`tools`・`enable_thinking` も効く) |
| `POST /tokenize` | `parse_special` で追加トークンをそのまま読む (`<tool_call>` は 1 トークン) |
| `GET /props` | `default_generation_settings.n_ctx` |
| 文脈の上限 (`contextLimit`) | prompt が上限以上なら、stream を始める前に 400 `exceed_context_size_error` (`n_prompt_tokens`・`n_ctx` 付き、llama-server の形) を返す。返答は上限で `length` になる |

`serve.swift` に足した引数: `--ctx` (既定 32768、§6-6)・`--ceiling-gb`・`--slots` (§10-3)。

### 10-2. API の照合 ([`ishizuki_api_check.sh`](../../bench/m6/ishizuki_api_check.sh)、`scratch/bonsai27b/ishizuki-api/20260927-173152`)

26 件中 25 件が通過。落ちたのは §9-2 の「制約なしの string の json_schema」1 件だけで、ishizuki の拘束の作りによるもの。アプリはこれを使っていない。
S3 で足した件: `max_tokens` の `length`・`timings` (非 stream / stream)・`/props`・`/tokenize`・`/apply-template`・強制呼び出し
(`/apply-template` の prompt に `<tool_call>\n<function=get_weather>\n` を足して `/v1/completions`、stop `</tool_call>` → `<parameter=city>\n大阪\n</parameter>\n</function>\n`、`stop`)・
completions の `length`・`none` の禁止 (呼び出しなしで答える)・禁止以外の bias は 400・要求ごとのサンプリング 6 項目・無指定なら公式値・文脈超えは 400。

- **同じ要求を 2 回送っても出力は一致しない。**`top_k 1`・temperature 1.0 の 2 回が 50 トークンほどで分かれた。
  1 回目は `cache_n 0`、2 回目は prefix cache に当たって `cache_n 16` で、prefill の数値がずれる。なので出力の一致では要求ごとのサンプリングが効いているかを確かめられず、`generation_settings` を読む形にした。

### 10-3. Offline 5 問 (`TsugumiToolLoopCheck --endpoint http://127.0.0.1:8128 --remote-model ternary-bonsai-2-27b --remote-direct`)

引数は [05 §6](05-BONSAI-PQ2-0.md) と同じ (`--conversations docs/experiments/knowledge-sources/questions-v2-stage2.json --network offline --max-rounds 6 --thinking off --context 32768 --repeats 1`)。
`--model` は種別を選ぶだけで `~/LLM/Ternary-Bonsai-2-27B-MTP` (`.bonsai27b`: 公式値 0.7 / 20 / 0.8 / 0.0 / 1.5)。MBP では何も読み込まない。
M6 の Swapouts を 10 秒ごとに取り、60 秒で +512 MiB か合計 +1 GiB で止めた (`scratch/bonsai27b/runs/*/watch.log`)。

| run | サーバの設定 | N1 / N2 / N3 / N4 / N5 (ラウンド・秒) | checks | M6 の Swapouts | サーバのメモリ (top の MEM) |
| --- | --- | --- | --- | ---: | --- |
| `ishizuki-offline-set1` | 予算の上限 = `iogpu.wired_limit_mb` 14336、スロットは 8 まで育つ | 3·24 / 4·31 / 4·55 / 4·40 / 6·59 | 5/5 | **+166,512 ページ (2.54 GiB)**、swap 使用 105 MB → 1.8 GB | 走行後 13 GB |
| `ishizuki-offline-c12-set1` | 上限 12 GB、スロットは育つ | 3·38 / 4·32 / 3·40 / 7·66 / 7·59 | 5/5 | +0 | 8.8〜11 GB |
| `ishizuki-offline-c12-set2` (同じサーバで続けて) | 同上 | N1 3·21 のあと N2 の 5 ラウンド目で見張りが止めた | — | **+41,208 ページ (644 MiB) を 20 秒で**。スロットが 4 → 8 になった直後 | 11〜12 GB |
| `ishizuki-offline-c12s2-set1` | 上限 12 GB、**スロット 2 に固定** | 3·26 / 4·35 / 3·36 / 7·72 / 3·27 | 5/5 | +0 | 9.1〜11 GB |
| `ishizuki-offline-c12s2-set2` (同じサーバで続けて) | 同上 | 3·22 / 4·34 / 4·42 / 4·44 / 6·50 | 5/5 | +0 | 10〜11 GB |

- **ishizuki の予算は、M6 の `iogpu.wired_limit_mb` (14336、参照ランタイムの計測用に上げてある) を上限に使う。**16 GB 機で 13 GB まで育ってスワップした。
  §4 は ishizuki の既定 (RAM の 75% = 12 GB) を前提にしていたので、`serve.swift` で 75% を明示して渡す形にした (`--ceiling-gb`、既定は RAM の 75%)。
- **上限を 12 GB にしても、スロットが 8 まで育つとスワップした。**予算の見積もり (重み + 1 GB + KV の約束 + buffer pool) が、16 GB 機で OS に残す分と合っていない。
  スロットを 2 に固定すると、同じサーバで 2 セット続けて Swapouts +0 だった。**`serve.swift` の既定を `--slots 2` にした** (0 で ishizuki の育て方に戻る)。
  代わりに失うのは、3 つ以上の会話を行き来したときの prefix cache の当たり (PrefixStore のディスク保存は残る)。アプリで同時に使う会話は 1 つなので、目的に反する代償ではないと判断した。
- 5 問の中で、アプリのラウンドは `auto` 86 と `none` 3 (4 run 合計)。**強制呼び出し (`function` / `required`) は Offline では 1 回も出ていない**ので、アプリのループの中で `/v1/completions` を通ったのは curl の照合 (§10-2) だけ。Online (S4) で出る。
- checks は `answer` と `error` だけ見ている。答えの中身が正しいかは、05 §6 の「正解 note との一致」のように照らしていない (未確認)。
- 秒は機械も実装も違うので、05 の llama.cpp の値とは比べない。

いま M6 の `ishizuki-serve` (手で起動、`--slots 2` を明示) と MBP の ssh 転送は動いたまま。LaunchDaemon は §9-1 の通り未設置。

## 11. S4 の途中結果: Online 1 セット目を見張りが止めた (2026-09-27、実測)

流し方: [`bench/m6/ishizuki_toolloop_run.sh`](../../bench/m6/ishizuki_toolloop_run.sh) `online` (引数は [11 §6](11-GUI-WIRING.md) と同じ。
`Scripts/qwen38/tool_loop_conversations.json` の 6 会話 14 ターン、`--web-store scratch/bonsai27b/web --pin-search --search-budget 0`、`--max-rounds 6 --thinking off --context 32768`)。
サーバは §10 の最後の設定のまま (上限 12 GB・`--slots 2`・n_ctx 32768)。出力は `scratch/bonsai27b/runs/ishizuki-online-set1`。

| 会話 | ターン | ラウンド | 秒 | うち prefill | decode | 最大 prompt | checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ja-news | 1 | 5 | 88 | 25 s | 952 tok / 15.3 tok/s | 6,233 | ok |
| ja-news | 2 | 7 | 105 | 27 s | 1,098 tok / 14.2 tok/s | 12,618 | ok |
| ja-news | 3 | 4 | 57 | 19 s | 520 tok / 13.7 tok/s | 16,909 | ok |
| en-swift | 1 | 7 | 91 | 31 s | 888 tok / 15.0 tok/s | 7,069 | ok |
| en-swift | 2 | 3 | 31 | 7 s | 341 tok / 14.6 tok/s | 9,161 | `online` 不成立 |
| en-swift | 3 | 7 | 66 | 25 s | 470 tok / 14.1 tok/s | 13,226 | ok |
| ja-howto | 1 | 7 | 129 | 33 s | 1,418 tok / 14.9 tok/s | 7,368 | ok |
| ja-howto | 2 | (5 ラウンド目の途中で停止) | — | — | — | 約 13K | — |

- **ラウンドの形は全部通った。**40 ラウンドのうち `auto` 22・強制呼び出し (`function`) 14・`none` 4。強制呼び出しは `/v1/completions` を通り、アプリの `QwenToolCallParser` で読めた。
  `structured_output_failure` もエラーも 0 回。アプリの記録に `cached` が出る (例: ja-news の 1 ターン目で 0 → 1,821 → 2,933 → 4,452 → 5,425)。
- **スワップ**: M6 の Swapouts は t = 260 s まで +0。そこから段々に増えた: +8,187 (t = 270)、+21,563 (t = 350)、+49,567 (t = 550)、+57,399 ページ (t = 590、897 MiB)。
  最後の 60 秒で +35,836 ページ (560 MiB) になり、見張り (60 秒で +512 MiB) が止めた。**この run は無効。**
  サーバのメモリ (top の MEM) は 10〜12 GB。サーバのログの MLX の値は、走行後で active 9.29 GB・peak 11.24 GB (Offline の後は peak 10.60 GB)。
  走行中に予算の段 (`budget:`) は動いていない (文脈の予約 32K・buffer pool 1 GB・スロット 2 は Offline の間に決まったまま)。
- 増えたのは文脈が 12K〜17K に伸びたターン (ja-news 2・3、en-swift 3、ja-howto 2) の辺り。長い prompt の prefill で一時的に持つ分が peak を押し上げている見込みだが、どの部分かは切り分けていない (未確認)。
- 秒は 11 §6 の llama.cpp の子サーバ (MBP、18 GB) と機械が違うので比べない。
- **別件**: ishizuki は待機中に `idle: released buffer pool` を 5 秒ごとにログへ書き続ける (serve.log 2,952 行のうち 2,741 行)。害は無いが、LaunchDaemon で常駐させるとログが育つ。

### 11-1. 判断してもらうこと

M6 (16 GB) でこのサーバが 12K トークンを超える会話をスワップなしで回せていない。手当ての候補 (どれも未測定):

1. **ishizuki の一時的な持ち分を下げる**: `Memory.cacheLimit` (buffer pool) を 0.5 GB に固定する、prefill の chunk を小さくする (`politeness`)。prefill の秒が延びるかもしれない。
2. **予算の上限をさらに下げる** (`--ceiling-gb 10` など)。予算が抑えるのは KV と buffer pool の約束で、prefill の一時的な持ち分には効かない見込み。
3. **文脈の常用点を 16K に下げる** (§6-6 を変える)。11 §6 の Online は最大 prompt が 25K まで行くので、ターンが途中で切れる。

推奨は 1 (目的の往復・待ち時間を変えずにメモリだけ下げる見込みがある)。1 で Online 1 セットを見張り付きで流し、Swapouts と prefill 秒を並べてから決める。

### 11-2. M6 の再起動後に流し直した (実測、n = 1)

ユーザーが M6 を再起動した後 (2026-09-27 22:37 JST)、同じバイナリ・同じ引数 (上限 12 GB・`--slots 2`・n_ctx 32768) でサーバを立て、同じ Online セットを同じ見張り付きで流した。
出力は `scratch/bonsai27b/runs/ishizuki-online-reboot-set1`。

- 前提: `iogpu.wired_limit_mb` は再起動で 0 に戻っていた (前の run は 14336)。swap 0 MB・Swapouts 0 から始めた。SSD256 は再起動でマウントが外れていたので `diskutil mount SSD256` した。
  mh のプロセスは ssh・cfprefsd・distnoted だけで、止めるべき不要なプロセスは無かった (メモリの上位はログイン画面の WindowServer 110 MB などシステムのもの)。
  マウント直後に Spotlight (`mds_stores`) が CPU 80% を使ったので、load が 1.5 を下回るまで 3 分待ってから流した。
- **14 ターン完走、Swapouts +0 (最初から最後まで)、swap 使用 0 MB のまま。**サーバのメモリ (top の MEM) は 8.3〜11 GB。最大 prompt は 17,839 トークン (ja-news 3)。
- `answer`・`error` 14 / 14、`online` 11 / 14 (ja-news 3・en-url 1・ja-price 2 が不成立)。検査の exit 1 は `online` の不成立による。

| 会話 | ターン | ラウンド | 秒 | うち prefill | decode | 最大 prompt | checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ja-news | 1 | 7 | 89 | 37 s | 763 tok / 15.1 tok/s | 8,197 | ok |
| ja-news | 2 | 7 | 100 | 30 s | 975 tok / 14.0 tok/s | 14,516 | ok |
| ja-news | 3 | 3 | 39 | 19 s | 265 tok / 13.6 tok/s | 17,839 | `online` 不成立 |
| en-swift | 1 | 7 | 74 | 25 s | 736 tok / 15.2 tok/s | 5,925 | ok |
| en-swift | 2 | 4 | 41 | 11 s | 432 tok / 14.7 tok/s | 8,840 | ok |
| en-swift | 3 | 7 | 82 | 23 s | 514 tok / 14.0 tok/s | 13,666 | ok |
| ja-howto | 1 | 7 | 101 | 27 s | 1,111 tok / 15.1 tok/s | 6,577 | ok |
| ja-howto | 2 | 7 | 98 | 29 s | 909 tok / 14.3 tok/s | 10,959 | ok |
| en-url | 1 | 1 | 25 | 10 s | 229 tok / 15.7 tok/s | 2,298 | `online` 不成立 |
| en-url | 2 | 7 | 43 | 13 s | 450 tok / 15.2 tok/s | 5,228 | ok |
| ja-price | 1 | 7 | 91 | 40 s | 750 tok / 14.8 tok/s | 8,715 | ok |
| ja-price | 2 | 7 | 81 | 27 s | 728 tok / 14.0 tok/s | 14,387 | `online` 不成立 |
| en-facts | 1 | 7 | 69 | 29 s | 544 tok / 15.1 tok/s | 6,672 | ok |
| en-facts | 2 | 5 | 63 | 26 s | 514 tok / 14.3 tok/s | 12,060 | ok |

合計 994 秒 (1 ターン 25〜101 秒)、prefill 346 秒、decode 8,920 トークン / 14.6 tok/s。MTP は無い。

- **§11 のスワップとの違いが何によるかは切り分けていない。**違いは少なくとも 2 つある: `iogpu.wired_limit_mb` (14336 → 0) と、前の run の前に M6 が長く動いていて swap に 700 MB 残っていたこと。
  予算の上限はどちらも `--ceiling-gb` で 12 GB に固定してあるので、予算の見積もりは同じ。n = 1 なので、§11-1 の手当ての要否は 2 セット目で決める。
- M6 を再起動すると wired limit は 0 に戻るので、参照ランタイム (llama.cpp・mlx_lm) を測るときは 14336 に上げ直す必要がある ([m6-prefill/29](../m6-prefill/29-REFERENCE-RUNTIMES-ON-M6.md))。ishizuki のサーバを動かしている間は、2 セット目で確かめるまで 0 のままにしておく (14336 のときにスワップした run が 1 本あるだけで、原因とは決まっていない)。

## 12. S5 の準備: GUI から手で試す形 (2026-09-27)

ユーザーが GUI で手で試したいと言ったので、§2-1 の「種別 `.bonsaiIshizuki` を足す」より小さい形にした。**新しい種別は足していない。**

- **manifest に `endpoint` を書くと、子の llama-server を起動せずそのサーバを使う** (`RemoteServerModelDirectory`、`LlamaServerInferenceClient.swift`)。
  `{"arch":{"family":"qwen3_8_dense_llamacpp"},"endpoint":"http://127.0.0.1:8128","remote_model":"ternary-bonsai-2-27b"}`。
  種別は `arch.family` で `.bonsai27b` のまま (公式値・Qwen の呼び出し形・文脈 4K〜32K)。127.0.0.1 / localhost 以外の endpoint は読み込み時に断る。アプリはこのサーバを起動も停止もしない。
  インストール判定 (`AppModelInstallationProbe`) は endpoint があれば完了とみなし、サーバに届くかは読み込み時 (`/props`・`/tokenize`) に確かめる。届かなければ「ssh 転送は張ってあるか」を添えて失敗する。
  テスト 3 本を `LlamaServerModelDirectoryTests` に足した (`Scripts/test.sh --filter "LlamaServerModelDirectoryTests|AppModelInstallationProbeTests"` で 14 本通過)。
- **切り替えはシンボリックリンク**: `~/LLM/Ternary-Bonsai-2-27B-ishizuki/manifest.json` (上の 1 ファイルだけ) を作り、アプリが読む `scratch/Ternary-Bonsai-2-27B-MTP` の向きを
  `~/LLM/Ternary-Bonsai-2-27B-MTP` (llama.cpp の子サーバ) からこちらへ替えた。戻すのは `ln -sfn /Users/mh/LLM/Ternary-Bonsai-2-27B-MTP scratch/Ternary-Bonsai-2-27B-MTP`。
  設定ファイル (`scratch/mac-app-settings-Ternary-Bonsai-2-27B-MTP.json`、文脈 32768・thinking off) はディレクトリ名で引くのでそのまま使われる。
- HUD のメモリは、子サーバのときと同じく MBP 全体の値で、M6 のサーバの値ではない。MTP の表示は出ても投機はしない (ishizuki の側に MTP が無い)。

## 13. ishizuki で MTP が効いていない理由 (2026-09-27、コードを読んだ。動かしていない)

ユーザーの問い「ishizuki はそもそも MTP に対応しないのか、損だからやらないのか」への答え。**損得を測って決めたのではない。**ishizuki は MTP の投機を実装している (`Text/MTPHead.swift`・`MTPDrafter`) が、今の形では次の 3 つが全部当てはまって使われない。

| 条件 | どこ (ishizuki 459ee00) | 今の形 |
| --- | --- | --- |
| パックに MTP ヘッドがある | `BonsaiModel`: `config.components?.mtp == true` のときだけ `mtp.fc` 以下を読む | M6 の MLX 2-bit パックは `components = {text: true, vision: true, mtp: false}`。llama.cpp で使った `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` にはヘッドがある |
| greedy (temperature 0) | `SpeculativeGeneration.drafts`: `model.mtp == nil \|\| options.temperature > 0 ? nil : MTPDrafter(…)`。README も「Pack MTP heads draft greedy turns」 | 公式値は thinking off で 0.7、on で 1.0 |
| 罰則が無い | 同じ `drafts` の入口: `options.repetitionPenalty == 1, options.presencePenalty == 0` でなければ投機を丸ごと返さない (prompt lookup の n-gram も) | thinking off の公式値は presence 1.5。**§5 の「prompt lookup 投機は既定 on」は誤りで、今の要求では prompt lookup も動いていない** |

取り代の手がかり (実測、別の土台):
- llama.cpp の子サーバでは、公式値のまま MTP (n_max 1) の受理が 79.9〜81.5% ([11 §6](11-GUI-WIRING.md))、mul_mv_ext を外して decode 1.30 倍 ([07](07-PQ2-0-SMALL-BATCH-KERNEL.md))。
- §11-2 の Online 1 セット (ishizuki) は、合計 994 s のうち decode が 8,920 トークン / 14.6 tok/s ≒ 610 s。1 ターンの半分以上が decode。
- ishizuki で縮む秒は測っていない。見積もりは置かない。

ishizuki で効かせるのに要るもの (どれも ishizuki に無い): (1) GGUF の MTP ヘッドを MLX パックへ移す、(2) temperature > 0 で分布を保つ検証 (llama.cpp が公式値で受理を出している経路)、(3) presence_penalty がある場合の扱い。
最小の段は「ヘッドをパックに載せ、greedy で受理率と decode tok/s を 1 回測る」。**別セッションで扱う** (ユーザー、2026-09-27)。

### 13-1. 次のセッションへの引き継ぎ (2026-09-27 時点の状態)

- M6: `ishizuki-serve` を手で起動したまま (pid はログ `~/LLM/ref/ishizuki-logs/serve.log`、`--slots 2`・上限 12 GB・n_ctx 32768・thinking off)。`iogpu.wired_limit_mb` 0 (再起動後の既定)。SSD256 はマウント済み。LaunchDaemon は未設置 (§9-1、sudo が要る)。
- MBP: `ssh -f -N -L 8128:127.0.0.1:8128 m6-lan` が張ってある。`scratch/Ternary-Bonsai-2-27B-MTP` は `~/LLM/Ternary-Bonsai-2-27B-ishizuki` (endpoint の manifest) を指している (§12。llama.cpp に戻すにはリンクを戻す)。
- ishizuki の checkout は `scratch/ishizuki` (459ee00 + `api.patch` + `bind-host.patch` + 2 つの実行ターゲット)。組んで運ぶのは `bench/m6/ishizuki_ship.sh`。
- 未了: S4 の Online 2 セット目 (§11-2 は n = 1)、§11 のスワップの原因の切り分け、S5 の GUI での確認 (ユーザーが手で試す)、S6 の記録値。


## 14. MTP を公式サンプリングのまま有効にした (2026-09-27〜28、実測)

ユーザー指示 (2026-09-27): 「今の実装に無いから使えない」で止めず、decode を速くするために MTP を有効にする。サンプリングは公式値 (thinking off: 0.7 / 20 / 0.8 / 0.0 / presence 1.5) のまま。
§13 の 3 条件を全部外した。

### 14-1. 変えたもの

| 何 | どこ |
| --- | --- |
| MTP ヘッドをパックに足す: MTP 同梱 GGUF の blk.64 (15 本、Q8_0 と F32) を 8-bit / group 32 の affine に量子化し、元のパックをリンクで指す `m6:~/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit-mtp` を作った (足したのは `mtp.safetensors` 478 MB と config・index だけ)。行列は Hadamard 回転なし (block 0)。norm は GGUF とパックで同じ形 (本体の層どうしで値が一致) なのでそのまま | [`bonsai_mtp_pack.py`](../../bench/m6/bonsai_mtp_pack.py) |
| 温度 > 0 と presence_penalty でも MTP で下書きする。検証は元からある「下書き 1 トークンを目標の確率で受理、外れたら残りから引く」(分布を変えない) をそのまま使い、罰則を行ごとに、その行より前の下書きトークンまで含めた窓で当てる (`Sampler.truncatedScores(rows:windows:)`)。最初のトークンにも窓を渡す。ヘッドの無いパックで罰則がある要求は、今まで通り下書きしない | `mtp.patch` (SpeculativeGeneration / Sampler) |
| 外れた下書きの扱い: 元はキャッシュをブロックごと巻き戻して、確定した分を次の検証に持ち越していた (外れの次の回が 3 トークンの forward)。既にある `ModelCache.keep` (attention は offset を戻し、delta-net は受理した位置まで規則だけ回し直す) を使うようにした。`ISHIZUKI_SPEC_KEEP=0` で元に戻る | `mtp.patch` (SpeculativeGeneration) |
| 2〜4 行の 2-bit 行列積 `FewRowQMV`: MLX の qmv_fast と同じ並びで、重みを 1 回読んで各行に掛ける。`ISHIZUKI_FEW_ROW_QMV=0` で切れる | `mtp.patch` (Core/FewRowQMV.swift、PackedLinear) |
| `timings` に `draft_n` / `draft_n_accepted`、要求ごとの `speculative.n_max` (0 で下書きなし、llama-server と同じ名前)、`ISHIZUKI_SPEC_TRACE=1` で 1 ターンの時間の内訳をログに出す | `mtp.patch` (APIServer / SpeculativeGeneration) |
| `ishizuki-serve --draft mtp\|all\|off` (既定 mtp: prompt lookup は使わない) | [`serve.swift`](../../bench/m6/ishizuki-driver/serve.swift) |
| 計測器 2 本: `ishizuki-verify-bench` (モデルなし、乱数の重みで行数ごとの行列積)、`ishizuki-decode-probe` (モデル 1 回読み込み、forward の内訳と seed 固定の生成比較。モデルのプロセスなのでサーバと同時に動かさない) | [`verify_bench.swift`](../../bench/m6/ishizuki-driver/verify_bench.swift)・[`decode_probe.swift`](../../bench/m6/ishizuki-driver/decode_probe.swift) |

組み方は `ISHIZUKI_BIN_DIR=ishizuki-bin-mtp3 bench/m6/ishizuki_ship.sh` (動いているサーバの `ishizuki-bin` に上書きしない置き場)。ship は Sources を固定コミットに戻してから 3 本のパッチを順に当てる形に変えた (mtp.patch が api.patch の足した行に触るため、1 本ずつの当たり判定ができない)。

### 14-2. 結果

**有効にしただけ** (巻き戻し・MLX の行列積のまま) では速くならなかった。サーバで同じ質問 400 トークンを交互に 3 回: MTP なし 16.2〜16.5 tok/s、MTP 15.1〜16.5 tok/s、受理 53〜61%。
`ISHIZUKI_SPEC_TRACE` の内訳は 1 ラウンド (257 回の平均) で検証の forward 92.6 ms・下書き 4.8 ms・判定 2.5 ms。

`ishizuki-decode-probe` (本文 `bench/m6/prompts/ja-twomachine.rendered.txt` 2,316 トークンの後、10 回の中央値。出力は `scratch/bonsai27b/mtp/probe-2.txt`):

| forward (logits は全位置) | 全体 ms | delta-net 48 層 | attention 16 層 | 出力ヘッド |
| --- | ---: | ---: | ---: | ---: |
| 1 トークン | 63.2 | 55.2 | 19.2 | 2.7 |
| 2 トークン | 79.3 | 64.2 | 25.2 | 4.5 |
| 2 トークン + FewRowQMV | 76.9 | 63.5 | 25.3 | 3.0 |
| 3 トークン | 110.1 | 86.7 | 34.4 | 4.5 |

(層ごとの列は層ごとに eval を挟んだ値で、合計は全体の値と合わない。)
92.6 ms と 79.3 ms の差は、外れの次のラウンドが 3 トークンの forward になっていたことで説明がつく (外れは 4 割)。

生成 (公式値、300 トークン、seed 1 / 2 / 3、同じプロセス):

| 設定 | tok/s | 受理 |
| --- | --- | --- |
| MTP なし | 15.99 / 15.94 / 15.96 | — |
| MTP、ブロックごと巻き戻し (元の作り) | 15.15 / 15.92 / 14.49 | 330 / 572 |
| MTP、`keep` | 18.22 / 17.95 / 17.52 | 339 / 562 |
| MTP、`keep` + FewRowQMV | 18.90 / 18.47 / 17.98 | 339 / 562 |
| MTP なし + FewRowQMV | 15.94 / 15.96 / 15.97 | — |

- `keep` + FewRowQMV の行は `keep` だけの行と同じ seed で受理数まで一致した。2 トークンの検証の logits は FewRowQMV のオンとオフで最大 2.93 違うが、argmax は一致した (1 回)。
- **サーバ** (`keep` + FewRowQMV、`ishizuki-bin-mtp3`、同じ質問 400 トークン、交互に 3 回): MTP 19.64 / 18.11 / 20.39 tok/s (受理 149/251・134/266・164/237)、MTP なし (`speculative.n_max` 0) 16.47 / 16.50 / 16.50 tok/s。検証の forward は 1 ラウンド 70〜74 ms。Swapouts +0。
- 受理率 53〜69% は llama.cpp の子サーバ (エージェント課題で 80% 前後、11 §6) より低い。課題も文脈も違うので並べて比べていない。キャッシュ済みの前半を再利用したターンでは、MTP ヘッドが見る文脈は新しく読んだ分だけ (ishizuki の作り、未変更)。

### 14-3. 2 トークンの検証がなぜ無料にならないか (実測)

`ishizuki-verify-bench` (乱数の重み、2-bit group 128、16 回を 1 回の eval にまとめた中央値、µs):

| 形 | 1 行 (MLX qmv) | 2 行 (MLX) | 2 行 (VerifyMatmul) | 2 行 (FewRowQMV) | 12〜64 行 (MLX、NAX の qmm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp gate/up 17408×5120 | 142〜164 | 226 | 325 | 212 | 611〜668 |
| mlp down 5120×17408 | 150〜155 | 223 | 323 | 211 | 618〜663 |
| attn q 12288×5120 | 95 | 163 | 238 | 154 | 436〜475 |

- 1 行は M6 の帯域 (144.4 GB/s、m6-prefill/03) 近くで読んでいる。2 行にすると重みを 1 回しか読まない FewRowQMV でも 1.3 倍で、復元と積和の演算の側が詰まる。
- ishizuki の VerifyMatmul (M1 Max で決めた閾値で 2 行では使われない) は M6 では 2 行で 2 倍。
- MLX は qmv の行数の上限を超えると tensor ops (NAX) の qmm に切り替えるが、この形では行数によらず 1 行の約 4 倍。2 行を水増しして NAX に乗せる手は遅くなる。
- tensor ops で積和を持たせる few-row のカーネルは書いていない。

### 14-4. tensor ops (MSL 4.0 `matmul2d`) で 2 行の検証を書いた — M6 では 1 行の qmv の 3.7〜4.5 倍で、採らない (2026-09-28、実測)

ユーザー指示: 「tensor ops で MTP のボトルネックの大きいところから」。一番大きいのは 2 トークンの検証で増える行列積 (§14-3、特に MLP)。
MLX の実行時カーネルは macOS 26 以降で MSL 4.0 で組まれる (mlx `device.cpp` の `get_metal_version`) ので、ishizuki の `MLXFast.metalKernel` から tensor ops が使える。
`FewRowNAX` (mtp.patch、`Core/FewRowNAX.swift`) を書いた: [qwen38_nax.metal](../../Sources/Tsugumi/Metal/Qwen/qwen38_nax.metal) の `qfn_dense_nax` と同じ形で、
threadgroup ごとに重み NR0 行 × 活性 8 行 (水増し)、K は 128 (1 グループ) ずつ、各スレッドが 2-bit を half に復元して threadgroup メモリに置き、`matmul2d` (4 simdgroup、float 累算) に渡す。**PackedLinear には結線していない。**

`ishizuki-verify-bench` (乱数の重み、2-bit group 128、中央値、µs。出力は `scratch/bonsai27b/mtp/verify-bench-nax.txt`)。値は MLX と一致 (max |diff| 0.005〜0.012、MLX 同士の差と同じ桁):

| 形 | 1 行 (MLX qmv) | 2 行 FewRowNAX | 4 行 | 8 行 | 参考: 2 行 FewRowQMV |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp gate/up 17408×5120 | 146 | 675 | 686 | 712 | 212 |
| mlp down 5120×17408 | 152 | 682 | 694 | 763 | 211 |
| gdn in_proj_qkv 10240×5120 | 74 | 409 | 415 | 429 | 134 |
| attn q 12288×5120 | 96 | 484 | 492 | 510 | 154 |

振った変数 (gate/up、µs):
- threadgroup の重みの行数: 64 → 675、32 → 537〜549、16 → 554。128 は動かなかった (threadgroup メモリ超え)。
- 活性の行数 (水増しの幅): 8 → 549、16 → 641、32 → 1,295。2 と 4 は動かなかった (出力なし。`matmul2d_descriptor` の最小幅の制約とみているが、エラーは確かめていない)。
- 左右のオペランドの入れ替え (`ISHIZUKI_NAX_SWAP`): 553 / 645 / 1,296 で同じ。
- 復元を消して `matmul2d` だけ (`ISHIZUKI_NAX_MODE=2`、値は無意味): 行数 32 で 404〜412、64 で 449。**tensor op 単体で 1 行の qmv の約 3 倍かかる。**復元の分は残りの 130 µs ほど。
  (`MODE=1` = `matmul2d` を外した版は、復元の結果が読まれないのでコンパイラが消し、30〜67 µs になった。比べる値にならない。)

読み: 2 行の検証で必要な積和は 17408×5120×2 ≈ 0.18 G 回で、M6 の tensor ops (MPS の fp16 GEMM で 18.6 TFLOP/s、m6-prefill/03) なら 20 µs で済む量だが、`matmul2d` は水増しした 8 行ぶんを、8 行 × 32 行 × 128 の小さなタイルでは 2.6 TFLOP/s でしか回らない。
活性の幅に比例して時間が増え、幅 8 が下限で、それでも帯域の時間 (1 行 qmv) の 3.7 倍。**M6 では、2 トークンの検証の行列積を tensor ops に移すと遅くなる。**MLX の NAX qmm (§14-3、12 行以上で 1 行の約 4 倍) と同じ結論。
attention の本体や delta-net の規則は、2 トークンではこれより小さい演算なので、同じ理由で tensor ops の出番は無い。
tensor ops が効くのは、1 回の検証で 32 トークン以上を並べる形 (下書きを長くする・木で並べる) だけで、MTP ヘッドは 1 トークンしか出さない。

## 15. 2 行の検証の行列積を half2 のマジック数で復元する — 検証 77.0 → 72.2 ms、生成 18.4 → 19.5 tok/s。下書きに罰則を当てる案は受理が下がって既定オフ (2026-09-28、実測)

ユーザー指示: 「M6 ishizuki MTP の改善続けて」。

### 15-1. 何を変えたか

- `FewRowQMV.applyMagic` (mtp.patch、`Core/FewRowQMV.swift`): qmv_fast と同じ並び (1 スレッド 16 値、simdgroup あたり出力 4 行) だが、2-bit の符号を 32-bit 語の AND と OR 1 回ずつで 2 個まとめて half2 にする (`(u & 0x00030003 << 2j) | 0x64006400` = 1024 + 4^j × 符号、j ≤ 4 は誤差なし。符号 5〜7 と 13〜15 は `u >> 10` から取る)。
  1024 のぶんはブロックごとに活性の和で 1 回引く。MLX の復元 (マスク・整数→float・FMA) より 1 重みあたりの命令が少ない。1〜4 行に対応。PackedLinear では **2 行のときだけ**使う (`ISHIZUKI_QMV_MAGIC=0` で FewRowQMV に戻る)。
- 下書きの argmax を、本体と同じ禁止トークン・repetition・presence (同じ窓) を当てたロジットで取る口 (`MTPDrafter.scoring`、`Sampler.penalizedScores`)。**既定オフ** (`ISHIZUKI_MTP_DRAFT_PENALTY=1` で有効)。15-3 のとおり受理が下がったため。
- 計測器: `ishizuki-verify-bench` に magic の節 (`VB_ONLY_MAGIC=1` でその節だけ)、`ishizuki-decode-probe` を MLX / FewRowQMV / magic の 3 通りの forward と 5 設定の生成に組み替えた。

### 15-2. 行列積単体 (`ishizuki-verify-bench`、乱数の重み、2-bit group 128、µs の中央値、30 回)

| 形 | 1 行 MLX | 1 行 magic | 2 行 FewRowQMV | 2 行 magic | 3 行 magic |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp gate/up 17408×5120 | 163 | 167〜168 | 211 | 196 | 284 |
| mlp down 5120×17408 | 150 | 157 | 210 | 195 | 280 |
| attn q 12288×5120 | 93〜94 | 91〜94 | 153〜154 | 144 | 205 |
| 1 バックボーン分 (7 形 × 層ごとの本数) | 42.0 ms | 42.6 ms | 59.7〜59.8 ms | 55.7〜55.8 ms | — |

- 値は MLX と一致 (max |diff| 0.004〜0.012、FewRowQMV と同じ桁)。
- 1 行は MLX と同じ時間で、帯域で頭打ち。2 行は 1 行の 1.2 倍 (FewRowQMV は 1.3 倍)。3 行で 1.7 倍に跳ぶ。
- 振ったが効かなかったもの: simdgroup あたりの出力行 4 / 8 / 16 (16 は 2 行で悪化)、1 ブロックの読みを先にまとめる並び。
- 同じ実行バイナリの最初の数回の測定では 1 行 magic が 191 µs と出た。その後の 3 回は 167〜173 µs。原因は見ていない。

### 15-3. モデルの forward と生成 (`ishizuki-decode-probe`、本文 2,316 トークン、`scratch/bonsai27b/mtp/probe-3.txt`)

2 トークンの検証の logits は MLX 比 max |diff| 2.96 (FewRowQMV は 2.93)、argmax 一致 (1 回)。

| forward (logits は全位置) | MLX | FewRowQMV | magic |
| --- | ---: | ---: | ---: |
| 1 トークン | 63.3 ms | 63.1 ms | 63.2 ms |
| 2 トークン | 79.3 ms | 77.0 ms | **72.2 ms** |

生成 (公式値 0.7 / 20 / 0.8 / 0.0 / presence 1.5、300 トークン、seed 1 / 2 / 3、同じプロセス、全部 `keep`):

| 設定 | tok/s | 受理 |
| --- | --- | --- |
| MTP なし | 16.03 / 16.06 / 15.94 | — |
| MTP、FewRowQMV (14 の既定) | 18.86 / 18.45 / 17.94 | 339 / 562 |
| **MTP、magic** | **19.94 / 19.48 / 18.98** | 339 / 562 |
| MTP、magic + 下書きに罰則 | 19.29 / 19.39 / 17.84 | 320 / 581 |
| MTP、FewRowQMV + 下書きに罰則 | 18.23 / 18.34 / 16.86 | 320 / 581 |

- magic は FewRowQMV と同じ seed で受理数まで一致。
- **下書きに罰則を当てると受理が 60.3% → 55.1% に下がった。**「presence で押し下げられる語を下書きしているから受理が低い」という読みは、この本文では外れ。既定オフにした。
- Swapouts +0 (165040 のまま)。

### 15-4. サーバ (`ishizuki-bin-mtp4`、`scratch/bonsai27b/mtp/server-ab-2.txt`)

M6 のサーバを止めて probe を流し、`ishizuki-bin-mtp4` で立て直した (引数は前と同じ `--slots 2 --ctx 32768`、`ISHIZUKI_SPEC_TRACE=1`)。
同じ質問 (「日本の鉄道の歴史を、明治から現代まで順に説明してください。」) 400 トークン、交互に 3 回:

- MTP なし (`"speculative.n_max": 0`、**平らなキー**): 16.25 / 16.23 / 16.20 tok/s
- MTP: 19.40 / 18.54 / 19.37 tok/s、受理 137/263・126/274・138/262
- 入れ子の `"speculative": {"n_max": 0}` は無視されて下書きする (最初の 3 組はこれで、全部 MTP: 19.97 / 19.26 / 19.01 / 19.22 / 18.00 / 19.11)。
- 検証の forward は 1 ラウンド 69.7〜70.1 ms (spec-trace)。Swapouts +0。
- §14-2 とは質問が違うので、サーバの tok/s は前と並べない。

### 15-5. 残り (次の手)

1 ラウンド ≒ 検証 70 ms + 下書き 3.2 ms + 判定 2.1 ms + その他 3.5 ms (spec-trace ÷ ラウンド数)、1 ラウンドで 1.5 トークン。
- 2 行 magic は 1 行の 1.2 倍。1 行 (帯域) まで縮めば検証はさらに約 13 ms 縮む (15-2 の 1 バックボーン分の差 55.8 − 42.0)。1 重みあたりの命令をさらに減らす手が要る。
- 受理 50〜60%。llama.cpp の子サーバの 80% 前後 (11 §6) とは課題が違い、同じ本文で比べていない。MTP ヘッドの実装の食い違いを疑うなら、同じ本文・greedy で llama.cpp と受理を並べる 1 本が先。

## 16. 2 トークンの検証の 2 行目が誤っていた — 量子化 KV の attention を頭とクエリで 1 つの積にまとめて直した。受理 59% → 73%、生成 19.1 → 21.6 tok/s (2026-09-28、実測)

ユーザー指示: 「2 行を 1 行並みにする方を続けて」。行列積を詰める途中で、行列積の外の増え方を調べて見つけた。

### 16-1. 行列積 (2 行を 1 行並みに) — 自前カーネルの読みで頭打ち

`ishizuki-verify-bench` の同じ重みを 16 回掛ける測り方は、重みの一部がシステムキャッシュに残って 1 行の MLX を実際より速く見せていた。重みを 4 組用意して回す測り方 (冷え) を足した (gate/up、µs、30 回の中央値):

| | 1 行 MLX | 1 行 magic | 2 行 FewRowQMV | 2 行 magic | 2 行 MLX | 読みだけ (1 行 / 2 行) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 同じ重みを 16 回 | 162〜167 | 167〜192 | 211〜212 | 196〜197 | — | 180〜192 |
| 4 組を回す | 180〜181 | 186〜192 | 212 | 197〜200 | 226 | 192 / 192 |
| 1 バックボーン分 (冷え) | 50.3〜50.6 ms | 54.8〜55.2 ms | 60.0〜60.1 ms | 56.4〜57.2 ms | 63.1〜63.4 ms | — |

- 2 行 magic は自分の読みだけの版 (演算を外した版、`ISHIZUKI_QMV_MODE=4`) より 5〜8 µs 多いだけ。2 行目の分はほぼ只で、残る差は自前カーネルの読みそのものが MLX 組み込みの 1 行より 6〜7% 遅いこと。
- 読みの差を詰めようとして効かなかったもの: 1 ブロックの読みを先にまとめる、活性を 4 個ずつベクトルで読む、1 スレッド 2 語 (2 行でレジスタが溢れたとみられ 281 µs)、グリッドを y 向き、スレッドグループを MLX と同じ (32, 2, 1)、k のループを展開しない、simdgroup あたりの出力行 4 / 8 / 16。
- 実 forward でも同じ傾向: 1 行を magic に通すと 1 トークンの forward が 63.1 → 66.0 ms (`probe-4.txt`)。**1 行は MLX のまま**にした (`ISHIZUKI_QMV_MAGIC1=1` で比べられる)。

### 16-2. 行列積の外: 量子化 KV の attention

実 forward の 1 → 2 トークンの増分 (magic で 63.1 → 73.1 ms) のうち、行列積の増分は冷えたベンチで約 2 ms。§14 の層別でも attention 16 層が 19.2 → 25.2 ms と、行列積の増分 (1 層 0.04 ms 程度) より大きく増えていた。
`Attention.quantizedAttention` は、KV 頭 1 本あたり 6 本のクエリ頭をブロードキャストのバッチにして、クエリの数を行 (M = l) にしている。これをモデル無しで測った (`VB_ATTN=1`、キー 3-bit・値 4-bit・group 64・密な窓 128、24 頭 / KV 4 頭、頭 256、1 層の µs):

| キャッシュ | 1 クエリ | 2 クエリ (今の形) | 2 クエリ (まとめた形) | 今の形の f32 比の誤差 | まとめた形の誤差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | — | 184 | 101 | 3.3e-4 | 3.3e-4 |
| 1,150 | — | 221 | 125 | 3.2e-4 | 3.3e-4 |
| 1,200 | — | 212 | 125 | **0.19** | 3.3e-4 |
| 2,316 | 124〜222 | 272 | 109〜110 | **0.125〜0.161** | 1.3e-4 |
| 3,000 | — | 345 | 135 | **0.122** | 1.0e-4 |
| 8,000 | — | 764 | 286 | **0.076** | 6.5e-5 |

(キャッシュ = 密な窓 128 を含む全トークン数。1 クエリの時間は同じ本文で 124〜222 と揺れた。)
- ずれているのは**値との積 (転置なしの `quantizedMM`) の 2 行目だけ**。1 行目とスコア側は 1 行ずつ掛けた値と一致する (2,316 で行ごとに確かめた)。
- 量子化部分が 1,022 トークンでは正しく、1,072 トークンからずれる。MLX は転置なしの 1 行ずつの積を、K (ここではトークン数) が 1024 以上で `qvm_split_k` に切り替える (`quantized.cpp`)。**2 行とブロードキャストのバッチの組み合わせで、この経路が 2 行目を誤る**とみている。MLX 側のコードは読んでいない。
- まとめた形: KV 頭ごとに、6 本のクエリ頭と l 個のクエリを 1 つの積の行 (M = 6l) にし、マスクは頭の数だけ縦に並べる。M = 12 はスコア側で qmm になり、スコアの積は 157 → 31 µs。

### 16-3. 直したもの (mtp.patch、`Text/Attention.swift`)

2〜8 クエリで、KV 頭 1 本に複数のクエリ頭がある量子化 KV の attention は、まとめた形で計算する (`ISHIZUKI_ATTN_FOLD=0` で今までの形)。マスクが 2 次元でないときは今までの形のまま。1 クエリと prefill (512) は変えていない。

### 16-4. 実モデルでの確かめ (`ishizuki-decode-probe`、本文 2,316 トークン、`scratch/bonsai27b/mtp/probe-5.txt`)

2 トークンの forward の各行を、1 トークンずつ順に流した logits と比べた (magic、TV は温度 0.7 の softmax の全変動距離):

| | 1 行目 (下書きの判定) | 2 行目 (受理後に次を引く分布) |
| --- | --- | --- |
| 今までの形 | max 差 0.052、argmax 一致、TV 0.006 | **max 差 3.83、argmax 不一致、TV 0.40** |
| まとめた形 | max 差 0.051、argmax 一致、TV 0.006 | max 差 0.036、argmax 一致、TV 0.003 |

**§14・§15 の MTP は、キャッシュの量子化部分が 1024 トークンを超える文脈では、下書きを受理したラウンドの次のトークンを 4 割ずれた分布から引いていた**。次の下書きに渡す hidden state も同じ位置のもの。§15-4 のサーバの測定 (短い質問、1024 トークンに届かない) には当たらない。§14-2・§15-3 の probe (2,316 トークン) には当たる。
16 §11-2 の Online (ishizuki、MTP なし) には当たらない (下書きが無ければ 2 トークンの forward は無い)。

forward: 1 トークン 63.1 ms (変わらず)、2 トークン 73.2 → 71.0 ms。

生成 (公式値、300 トークン、seed 1 / 2 / 3、同じプロセス、`keep` + magic):

| 設定 | tok/s | 受理 |
| --- | --- | --- |
| MTP なし | 16.08 / 16.02 / 15.96 | — |
| MTP、今までの形 | 19.71 / 18.98 / 18.73 | 335 / 565 (59%) |
| **MTP、まとめた形** | **20.83 / 21.72 / 22.23** | **381 / 520 (73%)** |

Swapouts +0。

### 16-5. サーバ (`ishizuki-bin-mtp5`、`scratch/bonsai27b/mtp/server-ab-3.txt`)

- 短い質問 (§15-4 と同じ、400 トークン): MTP なし 16.24〜16.52、MTP 19.22 / 19.23 / 19.90 tok/s、受理 135/265・135/266・144/257。§15-4 (19.40 / 18.54 / 19.37) と同じ水準。キャッシュが 1024 に届かないので、まとめた形の差は出ない。
- 長い本文 (`/v1/completions`、probe と同じ 2,316 トークン、300 トークン、交互に 3 回):

| サーバ | MTP なし | MTP | 受理 |
| --- | --- | --- | --- |
| まとめた形 (既定) | 15.94 / 15.45 / 15.44 | **20.48 / 21.59 / 20.44** | 115/185・125/176・115/185 |
| `ISHIZUKI_ATTN_FOLD=0` | 15.93 / 15.96 / 15.86 | 17.35 / 19.51 / 19.75 | 93/207・115/185・118/182 |

- 本文は 2 回目から prefix store の再利用になる (MTP ヘッドが見る文脈は新しく読んだ分だけ、§14-2)。Swapouts +0。
- M6 のサーバは `ishizuki-bin-mtp5` の既定 (まとめた形・magic、`ISHIZUKI_SPEC_TRACE=1`、`--slots 2 --ctx 32768`) で動いている。

### 16-6. 残り

- 2 行の行列積は自前カーネルの読みで頭打ち (16-1)。MLX 組み込みの 1 行の読みに届かせるには、MLX の metallib に入れる (mlx-swift の checkout に当てる) 形を試すことになる。
- 長い文脈ほど attention の 1 クエリ側も重い (1 クエリの経路は変えていない)。8,000 トークンでは 2 クエリのまとめた形 (286 µs) が今の 1 クエリの形より速い可能性がある。1 クエリを測っていない。
