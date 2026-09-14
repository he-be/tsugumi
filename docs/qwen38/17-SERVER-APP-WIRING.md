# 17. server / Mac アプリへの結線 (15 §2 G-1 の一部) — 現状の速度のまま使うための最小の経路

書いた日: 2026-09-14。対象は [16](16-WEIGHT-DIET.md) までの `Qwen38Runner` (両サイドカー既定 on)。表記は 01 と同じ (実測 / 導出 / 未確認)。
[15](15-DECODE-PLAN.md) の順序では G は 6 番だが、ユーザーの依頼 (「現状で一旦 GUI など配線して使いたい」) で G-1 の結線だけを先に入れた。
G-0 のチェックポイント (P−1・U−m・`<tool_call>` の直前) と G-1 の INV-1 (描き直し == 生成)、G-2 は**まだ無い**。

## 0. 結論

1. `TsugumiServer --model ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2 -c 12288 --draft-block-size 2` と、Mac アプリのモデル選択「Qwen3.8-Flash-Next IQ2 (MTP)」で動く。
   経路は `Qwen38ServerSession` (server) → `Qwen38Engine` (Tsugumi) → `Qwen38Runner`。アプリは `RealInferenceClient` が同じセッションを抱く (DecodeService の中)。
2. 検査 CLI と同じトークンが出る: greedy 200 トークンで、MTP off の server == `--qwen38-generate` (code)、MTP on の server == その off 出力 (code / explain)。
3. 12K の長いプロンプト (11,679 トークン) で 2 ターン: 1 ターン目 prefill 129.7 s・200 トークン 6.92 tok/s・受理 0.761、
   2 ターン目は cache_n 11,878 / prompt_n 24 (prefill 2.4 s)・84 トークン 7.08 tok/s・受理 0.766。wired 最大 14.38 GB、file-backed 最小 1.43 GB、Swapouts 0 (n = 1)。
4. prompt cache は「生きている状態の厳密な延長」だけ。再生成・指示つき再生成・履歴を編集したターンは全部 prefill し直す (15 §2 G の要件表の Ornith 列より下)。
   → 同日 [18](18-PROMPT-CACHE-CHECKPOINTS.md) でチェックポイントを入れ、要件表の場面は Gemma 以下になった。
5. ツール定義の描き方が上流と違う (§3-1)。Ornith と共有の既存の経路で、ツールを渡すリクエストのプロンプトは Python / HF の描画と一致しない。

## 1. 置いたもの

| 層 | ファイル | 中身 |
| --- | --- | --- |
| Tsugumi | `Runtime/Qwen38/Qwen38Sampler.swift` | 検査 CLI の `Q38Sampler` を移設 (CLI は typealias)。文法ゲートつきの `sample(_:gate:position:)` を追加 (拒否されたときだけマスクを作って引き直す) |
| Tsugumi | `Runtime/Qwen38/Qwen38Runner.swift` | `reset()`: GDN 状態・conv 履歴・PLE 履歴を 0、`plePrev` を初期値。KV・インデクサ鍵・MTP KV は位置ごとなので触らない |
| Tsugumi | `Runtime/Qwen38/Qwen38Completion.swift` | `Qwen38Engine`: チャンク prefill (MTP ヘッドにも同じ行を通す)、素の decode / MTP n_max 1 (`Qwen38MTPCheck.speculativeLoop` の chain 1 と同じ判定)、停止・取り消し・コールバック。`position` と MTP の未消化行 (`mtpTokens` / `mtpRows`)・`pendingH` を要求をまたいで持つ。`Qwen38ModelDirectory`: manifest の読み取り |
| server | `Core/Qwen38ServerSession.swift` | `QwenServerSession` と同じ形。トークナイザ・テンプレート・XML ツール呼び出しの文法とデコーダ・reasoning splitter・`QwenPromptCache` は Ornith のもの。サンプラは 0.7 / 0.8 / 20 / presence 1.5 に固定し、上書きを `approximations` に書く |
| server | `Command/main.swift`・`ServerArguments.swift` | `arch.family == qwen4exp` で振り分け。`-c` は丸め前の値 (`requestedContext`) を使う (12K は測定表に無い) |
| server | `QwenGenerationPlan.swift` | 公式サンプラの値を家族ごとに渡す (`OfficialSampler.ornith` / `.qwen38`) |
| app | `AppModelKind.qwen38` ほか | 判別は manifest の `qwen4exp`。MTP 2・thinking 既定 off・サンプラ固定・文脈 4K / 8K / 12K (既定 12K、`AppContextLengthOption.twelveK` を追加)。インストール検査は manifest と `tokenizer/` があれば完了扱い、ダウンロード元は無い。ツール (Web 検索) は Gemma だけのまま |

モデルディレクトリ (2026-09-14 に作った):

```
~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/manifest.json   {"arch": {"family": "qwen4exp"}, "qwen38": {"gguf": "...MTP.gguf", "ple": "ple/...Q4_1.gguf"}}
~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/tokenizer -> ../Qwen3.8-Flash-Next-tokenizer
scratch/Qwen3.8-Flash-Next-DS4-IQ2 -> ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2   (アプリの既定の置き場)
```

`down/`・`bf16/` のサイドカーはランナーが GGUF の隣から拾う (16)。

## 2. 実測 (scratch/qwen38/serve/)

server は `Scripts/qwen38/guarded.sh` 越し、メモリは `scratch/qwen38/serve/memwatch.sh` (memlog.sh の server 版)。

### 2-1. 描画とトークン化

| プロンプト | server の描画 → `/tokenize` | Python (`chat_prompts.py`) | 一致 |
| --- | ---: | ---: | --- |
| code | 80 | 80 | ○ |
| long | 11,679 | 11,679 | ○ |
| tool (ツール 2 本) | 423 | 468 | × (§3-1) |

### 2-2. 出力の一致 (greedy、`Q38_SERVER_GREEDY=1`、4K、max_tokens 200)

| 腕 | code | explain | 参照 |
| --- | --- | --- | --- |
| server MTP off | 一致 (626 字)、6.20 tok/s | — | `spec14/code-greedy-off.out` (CLI、W 前。W はビット一致なので同じ列のはず) |
| server MTP on | 一致 (626 字)、8.97 tok/s | 一致 (1,004 字)、8.76 tok/s | 同上 `*-greedy-off.out` |

比べたのは detokenize した本文 (停止トークンを除く)。tool は §2-1 のとおりプロンプトが違うので比べていない (MTP on で `tool_calls` 2 本がパースされて返ることだけ確認)。

### 2-3. 延長 (4K、greedy、MTP off、短い 3 ターン)

| ターン | cache_n | prompt_n | 生成 |
| --- | ---: | ---: | ---: |
| 1 | 0 | 27 | 9 |
| 2 | 35 | 25 | 3 |
| 3 | 62 | 22 | 2 |

cache_n = 前の prompt + 生成 − 1 (最後に出した `<|im_end|>` は食っていない)。描き直しが生成と一致した場合だけの数字で、thinking off・ツール無し。

### 2-4. 12K (公式サンプラ、MTP on、`long`、n = 1)

| | 1 ターン目 | 2 ターン目 (「2 文で要約」) |
| --- | ---: | ---: |
| prompt / cache_n / prompt_n | 11,679 / 0 / 11,679 | 11,902 / 11,878 / 24 |
| prefill | 129.7 s (trunk + MTP ヘッド) | 2.4 s |
| 生成 / tok/s (最初のトークン込み) | 200 / 6.92 | 84 / 7.08 |
| 受理 (rounds) | 0.761 (113) | 0.766 (47) |
| wall | 158.5 s | 14.2 s |

走行全体で wired 最大 14.38 GB、file-backed 最小 1.43 GB、server RSS 最大 1.76 GB、Swapouts 0。

### 2-5. アプリ経路 (`Scripts/app/smoke_decode.py qwen38`、DecodeService、12K、MTP on)

| | 結果 |
| --- | --- |
| load | ready 0.9 s (重みは mmap、実読みは最初の要求で) |
| "What is 7 times 8?" | `56`、3 トークン、受理 1/1 |
| "reverse a string" | `` `s[::-1]` ``、6 トークン、受理 2/3 |

GUI の画面そのもの (モデル選択・Inspector) は起動して見ていない (**未確認**)。

## 3. 分かったこと・残り

### 3-1. ツール定義の `tojson` が上流と違う (Ornith と共有)

swift-jinja の `tojson` (`.build/checkouts/swift-jinja/Sources/Jinja/Filters.swift:1075`) は `JSONEncoder` + `.sortedKeys` で、空白なし・キーを辞書順・`/` を `\/`・非 ASCII を `\u` にする。
上流の描き方 (HF の `tojson` = `json.dumps(ensure_ascii=False)`、挿入順・`", "` / `": "`) と違うので、ツールを渡すプロンプトは学習時の形と別の列になる (tool で 423 vs 468 トークン)。
メモリ `ornith-tojson-roundtrip` の `\/` と同じ根。品質への影響は測っていない。直すならテンプレートに渡す前に上流と同じ JSON 文字列を作るフィルタを差し込む (G-1 の INV-1 と同じ作業場所)。

### 3-2. 15 §2 G に対する位置

- G-0: `reset()` だけ入った。`captureCheckpoint` / `restore` / `--qwen38-resume` は無い。
- G-1: 結線は入った。INV-1 (文法を正準形に縛る・テンプレート変種) は無い。
- G-2: 無い。いまの cache は `QwenPromptCache` にチェックポイントを 1 本も登録しない形。

### 3-3. 既知の制限

- MTP は n_max 1 固定 (`--draft-block-size 2`)。n_max 2 (14) は検査 CLI だけ。
- thinking を on にしてもサンプラは non-thinking の値のまま (運用点は thinking off)。
- 取り消し・例外の後は `reset()` して次の要求は全部 prefill。
- 12K を超える文脈は選べない (server は `-c` をそのまま受けるので、16K 以上を渡すと 09 の Swapouts の縁)。
