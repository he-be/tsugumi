# 13. Q0 基準線 (F0) — MBP 単体で本文 1 は 31.5〜34.1 s、本文 2 (記録の再生 1 ターン) は 168.7 s。閾値は本文 2 で 3.06 s (実測、2026-09-25)

証拠: [`bench/qfn/results/2026-09-25-Mac15-6-ledger-q0/`](../../bench/qfn/results/2026-09-25-Mac15-6-ledger-q0/) (MBP 単体)。

```
MBP: 測定: 2026-09-25 / Mac15,6 Apple M3 Pro 18GB / macOS 15.7.5 (24G624) / Swift 6.3.3 / SDK 26.5 / Xcode 26.6 / APPLE SSD AP1024Z / commit 9487cc8 / minos 15.0
```

コマンド (`command.txt`):

```
bench/qfn/pair_bench.sh ledger-q0 6 --solo
本文 1: .build/release/TsugumiServer --model ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2 --host 127.0.0.1 --port 8093 --ctx-size 32768 --draft-block-size 2
本文 1 request: POST /v1/chat/completions {messages: bench/m6/prompts/<body>.json, max_completion_tokens: 1, cache_prompt: false, chat_template_kwargs: {enable_thinking: false}}
本文 2: .build/release/TsugumiToolLoopCheck --model ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2 --conversations docs/experiments/knowledge-sources/questions-v2-stage2.json --only N5-passport --network offline --max-rounds 6 --thinking off --context 32768 --replay docs/experiments/knowledge-sources/ple-ab/bf16/offline/rep1 --replay-model --out <b2-rN>
each under: caffeinate -i Scripts/qwen38/guarded.sh <log> <cmd>
```

反復 n = 6 (本文 1 は 5 本 × 6 巡 = 30 プロセス、本文 2 は 6 プロセス)。全 36 run が exit 0。`env.txt` は空 (`Q38_` / `TF_` なし)。

**台帳** ([06](06-LEDGER.md)): 最初の行。単体の列に 6 行 (本文 1 の 5 本、本文 2)。

## 0. 結論

1. **本文 1** (自然文 5 本、新しいプロセスで 1 本ずつ、(c)) の中央値は 31.523〜34.110 s。本文ごとの揺れ幅 (最大 − 最小) は 0.191〜0.485 s。
2. **本文 2** (記録の再生 1 ターン、7 ラウンド、1 プロセス、(b)) の prefill 合計の中央値は **168.702 s** (167.973〜169.502)、揺れ幅 **1.529 s**。
3. **閾値** (04 Q0、揺れ幅 × 2、**導出**): 物差しの本文 2 (P3) で **3.06 s**。本文 1 は本文ごとに 0.38〜0.97 s (§1.1 の最右列)。以降の Gate の「縮んだ」は、これを下回らない秒で言う。

## 1. 数字

### 1.1 本文 1 (`summary-b1.tsv`、`timings.prompt_ms`)

| 本文 | トークン | 中央値 s | 最小 | 最大 | 揺れ幅 s | 閾値 s (× 2、導出) | 1 プロセスの Pageins MB (中央値) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ja-twomachine | 2,316 | 31.903 | 31.699 | 32.053 | 0.354 | 0.71 | 51,181 |
| ja-m6readme | 2,347 | 31.523 | 31.356 | 31.841 | 0.485 | 0.97 | 51,618 |
| en-readme | 2,408 | 33.940 | 33.800 | 34.156 | 0.356 | 0.71 | 55,266 |
| en-server | 2,412 | 34.110 | 33.959 | 34.150 | 0.191 | 0.38 | 56,518 |
| code-runner | 2,442 | 32.212 | 31.933 | 32.380 | 0.447 | 0.89 | 50,723 |

- トークン数は Qwen3.8 のテンプレートで描いた値 (`prompt_n`)。Gemma で数えた 2345〜2499 (m6-prefill/43) とは違う。
- Pageins はサーバーの起動 (GGUF の mmap、重みの読み込み) を含むプロセス全体の前後差で、prefill だけの読みではない。
- Swapouts は 30 本中 1 本 (3 巡目の en-server) で 8 ページ、他は 0。

### 1.2 本文 2 (`summary-b2.tsv`、`b2-rN/rounds.jsonl` の `prefill_s`)

| | 中央値 s | 最小 | 最大 |
| --- | ---: | ---: | ---: |
| 1 ターンの prefill 合計 | 168.702 | 167.973 | 169.502 |
| 各ラウンドのエンジン呼び出しの合計 (`ended − started`) | 169.154 | 168.420 | 169.957 |

| ラウンド | 新規トークン | prefill 中央値 s | 最小 | 最大 | 元の記録 (n = 1) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,225 | 20.492 | 20.410 | 21.244 | 20.849 |
| 2 | 719 | 15.694 | 15.645 | 15.822 | 15.254 |
| 3 | 1,583 | 21.634 | 21.423 | 21.849 | 21.945 |
| 4 | 2,021 | 25.966 | 25.736 | 26.220 | 26.288 |
| 5 | 2,130 | 31.632 | 31.477 | 31.908 | 30.682 |
| 6 | 1,962 | 25.880 | 25.758 | 26.385 | 25.994 |
| 7 | 1,948 | 27.099 | 27.002 | 27.226 | 26.920 |

- 新規トークンは同じ 1 本の記録を流しているので 6 本とも同じ。2,048 を超えるのはラウンド 5 だけ (2 チャンク)。
- 「元の記録」は 2026-09-17 のアプリの走行 (`ple-ab/bf16/offline/rep1`、合計 167.931 s) で、n = 1。並べるだけで比べない。
- 1 プロセスの Pageins は 263,889〜266,593 MB。Swapouts は 0〜40 ページ。
- 6 本とも 7 ラウンドを完走し、`checks` は `answer` / `error` とも真。

## 2. 測り方と逸脱

- **本文 1**: m6-prefill/43 の `server-prompts-warm2` の形を QFN に当てた。1 本ずつ新しい `TsugumiServer` を起動し、`/health` を待ってから 1 リクエストを送り、応答を受けたら止める。5 本の順を巡ごとに 1 つずらし、同じ本文を続けて流さない (43 §3)。プロセスの間は 20 秒。
- **本文 2**: `TsugumiToolLoopCheck --replay RUN --replay-model` (9487cc8 で追加)。アプリの `AppModel` が記録どおりのツール呼び出しを実行してリクエストを組み、各リクエストを同じプロセスのアプリのエンジン (`RealInferenceClient`、Qwen3.8 セッション、prompt cache とチェックポイント込み) に 1 トークンで通す。ラウンドは記録どおりで、生成で分岐しない。
- 記録の選び方: Mac の QFN・BF16 PLE・Offline・アプリの AppModel で取った記録のうち、ラウンド数が 04 Q0 の範囲 (3〜7) の上端のもの (N5-passport、7 ラウンド)。
- 逸脱:
  1. **エンジンへのリクエストは `tool_choice none`**。`auto` のまま 1 トークンにすると、1 トークン目が呼び出しの開始トークンになって呼び出しの解析が失敗する (試走で起きた)。`none` はサンプラでその 1 トークンを禁じるだけで、描いたプロンプトが元のリクエストと同じトークン列であることを、ラウンドごとに再生側で検査している (違えば止まる)。
  2. **前のラウンドの生成文を prefill し直す。**元の記録では、前のラウンドでモデルが書いた文 (29〜46 トークン) が KV に残った状態から次のラウンドを始めていた (`cached` = 前の prompt + generated − 1)。再生ではエンジンが書くのは 1 トークンで、次のプロンプトには記録の文が入るので、その分も新規トークンとして prefill する。
  3. **ラウンド 1 のプロンプトが記録より 37 トークン短い** (1,262 → 1,225)。以降のラウンドも 37〜41 トークン短い。09-17 以降にアプリの system 文かツール定義が変わったためと見られるが、差分は確かめていない。
  4. KV の型 (Q8_0) とチャンク (2048) はコードの既定 (qwen38/23) のままで引数では渡していない。起動行には出ないので、この run の中では確かめていない。
  5. サーバーの応答の `model` 欄とログの `model=` は `gemma-4-26b-a4b-it` だが、同じログの `family=qwen4exp` と `prompt cache qwen38` の行から、動いたのは QFN。表示のずれとして記録するだけ。
- `Q38_ROUND_LOG` の段の秒 (04 Q0 の「別の run」) は取っていない。

## 3. 残るもの

- 段の秒の run (`Q38_ROUND_LOG=…`、本文 2 を n = 3)。台帳の行ではない。
- 次は 12 §2 の順で Q2 (PLE の先読み)。
