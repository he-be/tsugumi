# 10. Bonsai 2 27B を Mac アプリ (GUI) で試せるようにする計画

2026-09-20。**計画であり、まだ何も実装していない。**表記は [qwen38/01](../qwen38/01-Q2-FIRST-LIGHT.md) と同じ
(実測 / 導出 / 未確認)。この文書で使う言葉:

- **遠隔クライアント**: `RemoteInferenceClient`。アプリのツールループの下で、推論だけを HTTP 越しの
  `llama-server` に投げる `AppModelLifecycleClient` の実装。今は `Sources/TsugumiToolLoopCheck/` にしか無い。
- **子サーバ**: アプリが自分で起動して自分で止める `llama-server` のプロセス。

[09](09-ONLINE-FULL-RUN.md) で、アプリと同じ Online のツールループが Bonsai (PQ2_0 + MTP n_max 1、32K) の
`llama-server` の上で 14 ターン回答まで届いた。ただし回したのは `TsugumiToolLoopCheck` で、GUI からは選べない。

## 0. 結論

1. **ネイティブのランナーには移植しない。GUI から `llama-server` を子サーバとして立てる。**
   移植 ([05](05-BONSAI-PQ2-0.md) §11-2: Metal の Hadamard → PQ2_0 の検査 → ランナーの参照一致 → Vision) は
   「GUI で試す」ためには重い。遠隔クライアントは既にアプリのループを通っているので、足すのは起動・停止と選択だけ。
2. **段は 6 つ** (§3)。S1 遠隔クライアントを `TsugumiAppCore` に移す → S2 サンプリングを公式値にする →
   S3 モデル種別 `.bonsai27b` を足す → S4 子サーバの起動・停止 → S5 種別でクライアントを振り分ける →
   S6 検査を通してから GUI で触る。
3. **先に直すものが 2 つある。**どちらも 09 の実行に既に入っていた。
   - **サンプリングが公式値とずれている** (§1-3、実測)。遠隔クライアントは `presence_penalty` と `min_p` を送っておらず、
     サーバの既定 (0.0 / 0.05) で走っていた。公式は thinking 無効で 1.5 / 0.0。S2 で直す。
   - **会話の切り替えでスワップが出る** (09 §4)。GUI ではチャットを切り替えるたびに来る。S6 の検査に入れ、
     GUI で触る前に数字を出す。
4. **最初の GUI 試用に入れないもの**: Vision (05 §7 で画像 1 枚 297 MB のスワップ)、thinking 有効、
   ダウンロード (重みもサーバもこの機体のローカルのパス)、速度計の針 (§4)。

## 1. 今の結線 (実測、コードを読んだ結果)

### 1-1. アプリはローカルのクライアントしか持たない

| 場所 | 中身 |
| --- | --- |
| `Sources/TsugumiApp/Mac/App/TsugumiMacApp.swift:31` | `AppModel(client: DecodeServiceInferenceClient(), …)` の 1 本で固定 |
| `Sources/TsugumiApp/Core/Inference/` | `RealInferenceClient` (同一プロセス) と `DecodeServiceInferenceClient` (launchd のジョブ + socket) の 2 つ |
| `Sources/TsugumiToolLoopCheck/RemoteInferenceClient.swift` (427 行) | 遠隔クライアント。`AppModelLifecycleClient` と `AppInferenceRuntimeReporting` を満たす。`TsugumiAppCore` と `Tsugumi` にしか依存しない |
| `Sources/TsugumiToolLoopCheck/main.swift:373-398` | `--endpoint` があれば遠隔クライアントを `AppModel` に渡す。ループ・プロンプト・ツール実行はアプリのまま |

### 1-2. モデルの選択は種別 (`AppModelKind`) で決まる

- 種別は `manifest.json` の `arch.family` から読む (`AppModelKind.probe`)。今は Gemma / Ornith / `qwen4exp` (= `.qwen38`) の 3 つ。
- 能力 (Vision・ツール・thinking の既定)、公式サンプリング、コンテキストの選択肢、置き場の名前が種別に集まっている。
- `.qwen38` が前例になる: `.moepack` ではなくローカルのディレクトリで、ダウンロード元を持たない
  (`PrebuiltModelSource.qwen38`)。`.qwen38` を分岐に書いている箇所は `Sources` 全体で 42 行・14 ファイル。
- 09 では `--model ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2` を渡して種別を `.qwen38` に見せた
  (プロンプト・ツール呼び出しの構文・サンプリングを Qwen 用にするため)。GUI ではこの借り方はできない。

### 1-3. 遠隔クライアントが送るサンプリング (実測)

`requestBody` が送るのは `temperature` / `top_k` / `top_p` だけ (`RemoteInferenceClient.swift:295-302`)。

| | 公式 (thinking 無効、[01](01-FEASIBILITY.md) §4) | 09 で実際に効いた値 |
| --- | ---: | ---: |
| temperature / top_p / top_k | 0.7 / 0.80 / 20 | 0.7 / 0.80 / 20 |
| presence_penalty | **1.5** | **0.0** (送っていない。サーバの既定、`common/common.h:242`) |
| min_p | **0.0** | **0.05** (送っていない。同 `:232`) |

ローカルの Qwen3.8 は `Qwen38Sampler` がエンジンの中で presence_penalty を掛けるので、`AppGenerationRequest` には
この項目が無い。遠隔の経路だけが抜けている。**05 §6 と 09 の数字はこのずれを含む。**

### 1-4. 遠隔クライアントは llama-swap の経路を前提にしている

`/upstream/<id>/props`・`/tokenize`・`/apply-template` を叩く。素の `llama-server` にはこの経路が無いので、
09 は `Scripts/qwen38_27b/upstream_relay.py` を挟んだ。PC (192.168.0.199) の反復は llama-swap 越しなので、この経路は残す。

## 2. 作るもの

```
AppModel ── KindRoutingInferenceClient ─┬─ DecodeServiceInferenceClient   (Gemma / Ornith / Qwen3.8、今のまま)
                                        └─ LlamaServerInferenceClient     (.bonsai27b)
                                              ├─ 子サーバの起動・/health 待ち・停止
                                              └─ RemoteInferenceClient (直叩き、127.0.0.1)
```

モデルのディレクトリは `.qwen38` と同じくローカル置きで、`manifest.json` が中身を名指しする:

```json
{ "arch": { "family": "qwen3_8_dense_llamacpp" },
  "llama_server": "~/LLM/prism-llamacpp/src-b10709/build/bin/llama-server",
  "gguf": "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf",
  "server_args": ["-ngl", "99", "-fa", "on", "-np", "1", "--jinja",
                  "--spec-type", "draft-mtp", "--spec-draft-n-max", "1"] }
```

`-c` はアプリのコンテキスト設定から渡す。置き場は `~/LLM/Ternary-Bonsai-2-27B-MTP/` (重みは既にここにある)。

## 3. 段と、それぞれの合否

| 段 | やること | 合否 |
| --- | --- | --- |
| **S1** | `RemoteInferenceClient` を `TsugumiAppCore` に移す。経路の前置き (`upstream/<id>/` か無し) を引数にする。`TsugumiToolLoopCheck` は移した先を使う | `--endpoint` を中継なしで `llama-server` に向けて 1 会話が通る。PC の llama-swap 向けも 1 会話通る |
| **S2** | 遠隔の要求に `presence_penalty` と `min_p` を足す。値は種別と thinking の有無から 01 §4 の表で選ぶ (GGUF のメタデータからは読まない) | サーバログの要求に 1.5 / 0.0 が出る。単体テストで要求の中身を見る |
| **S3** | `AppModelKind.bonsai27b` を足す (`probe` に family、能力、公式サンプリング、コンテキスト 4K〜32K、置き場、`PrebuiltModelSource` はダウンロード無し)。`.qwen38` の 42 行を 1 つずつ見て、Bonsai がどちら側かを決める | `swift build` が通り、既存の `TsugumiAppCoreTests` が緑。`probe` の単体テスト |
| **S4** | `LlamaServerInferenceClient`: `ensureLoaded` で子サーバを起動し `/health` を待ち `props` を読む。`unload`・アプリ終了・コンテキスト変更で子孫ごと止める。前回の異常終了で残ったサーバは起動時に pid ファイルで見つけて止める。ポートは空きを取る | 起動 → 1 問 → `unload` の後に `llama-server` が残っていない。アプリを kill した次の起動で、残ったサーバが止まる |
| **S5** | `KindRoutingInferenceClient`: `ensureLoaded` のディレクトリから種別を読み、内側を選ぶ。切り替えるときは先に反対側を `unload` する (18 GB に 2 つは載らない)。`TsugumiMacApp.swift:31` をこれに替える | Gemma → Bonsai → Gemma と選び直して、それぞれ 1 問答えられる。反対側のプロセスが残らない |
| **S6** | 検査を通す (§3-1)。通ってから GUI で触る | §3-1 の 3 つ |

### 3-1. S6 の検査 (GUI で触る前に自動で出す)

1. **14 ターン**: `bonsai_online_run.sh` を中継なし・S2 のサンプリングで流し直す。09 §2 と同じ表を 2 セット。
2. **会話の切り替え**: 同じ子サーバで 2 会話を続けて流し、切り替えの前後 2 分の Swapouts・空き率・メモリ上位を 5 秒刻みで残す。
   09 §4 は 2 回とも見張りを超えた (+1.86 GiB / +0.60 GiB)。**まず既定のまま再現を取り、次に `--cache-ram` と
   `--ctx-checkpoints` を 1 つずつ変えた実行を並べる** (09 §4 の時点で既定のままだった 2 つ。原因かどうかは未確認)。
   この数字が出るまで、GUI の試用は「チャットを切り替えるとスワップが出る」前提になる。
3. **Stop**: 生成の途中で `cancel()` し、サーバログに `cancel task` が出て次の要求が通ること。

GUI で見るのは、モデルの選択・ロード中の表示・Online の 1 問・チャットの切り替え・Stop・終了後にプロセスが残らないこと。

## 4. GUI の中で Bonsai だけ違って見えるところ

| 項目 | どうなるか |
| --- | --- |
| 速度計の針 | 出さない。針は「借りられるメモリ ÷ 重み 12 GB」で、分母が Gemma のもの。遠隔クライアントの `loadedRuntimeOwnBytes` は nil |
| Inspector の tok/s・prefill・cache | 出る。09 の `turn-metrics.jsonl` に `tokensPerSecond`・`prefillSeconds`・`cachedPromptTokens` が入っている (実測) |
| ロード時間 | 約 17 秒 (05 §5、実測) |
| サンプリングの欄 | 固定表示 (`samplingIsLocked`)。値は 01 §4 |
| Vision | 無効。mmproj はあるが最初は載せない |
| thinking | 無効のみ。09 が確かめたのは無効だけ |
| 速さ | 1 ターン 37〜162 秒、decode 15〜18 tok/s、prefill 81〜92 tok/s (09、実測。S2 の前の値) |

## 5. 入れないもの

- ネイティブのランナーへの移植 (05 §11-2)。
- 重みと `llama-server` の配布。どちらもこの機体のローカルのパスで、`.app` には同梱しない。
- PTQ1_0。MTP を足した一本が無い ([08](08-PP-TG-CEILING.md) §6-4)。

## 6. 決めてもらうこと

1. **S6-2 の結果が悪いままでも GUI に入れるか。**切り替えのスワップが引数で消えなかった場合、
   チャットを切り替えるたびに子サーバを立て直す (約 17 秒) 形が残る。
2. 種別の表示名 (案: 「Bonsai 2 27B PQ2_0 (MTP)」)。
