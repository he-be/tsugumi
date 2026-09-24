# 45. M6 のサーバーを MBP の pi から使った 1 回の記録 — 8 リクエストの prefill 秒 (実測、2026-09-25)

証拠: [`bench/m6/results/2026-09-25-Mac18-5-server-pi-manual/`](../../bench/m6/results/2026-09-25-Mac18-5-server-pi-manual/) (M6)。
`server.log` はサーバーの標準出力・標準エラーをそのまま写したもの。**スクリプトではなく手で集めた** (`prefill_bench.sh` の形ではない。`summary.tsv` と vm_stat は無い)。

```
測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 9ae7eb0 / minos 15.0
```

```
.build/release/TsugumiServer --model scratch/gemma4-qat-sym.gturbo --host 0.0.0.0 --port 8091 \
  --ctx-size 65536 --verification trusted-install --draft-block-size 4
```

クライアントは MBP の pi (`openai-completions`、プロバイダ `m6-tsugumi`)。タスクはユーザーが手で流した。
起動行は `context=65536 slots=32 expert_io=mmap mtp=4 reasoning_budget=-1`、prefill のカーネルは `shared qmm kernel=mpp-nax` / `moe tiled kernel=mpp`。
外付け SSD はマウント済み、モデルに `expert-mirror.json` あり。ログに写しの警告は無い。

**反復 n = 1。解釈は書かない。**比べる相手 (変更前の commit、`TF_EXPERT_MIRROR=0`) は流していない。
台帳 ([06](06-LEDGER.md)) の量 (CLI・固定の本文・新しいプロセスの 1 回目) とは測った量が違うので、台帳 §1 は動かない。

## 1. 数字

`prompt_n` と `prompt_ms` は応答の `timings` と同じ値で、9ae7eb0 から完了行に出している。
prefill tok/s は `prompt_n ÷ prompt_ms` (**導出**)。decode tok/s は `(completed in − prompt_ms) ÷ completion` (**導出**、decode 以外の処理の時間を含む)。

| # | 時刻 (UTC) | prompt | cached | prompt_n | prompt_ms | prefill tok/s (導出) | completion | completed in | decode tok/s (導出) | finish | accept |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 22:41:26 | 4203 | 0 | 4203 | 6787.5 | 619 | 64 | 9.256 s | 25.9 | stop | 1.520 |
| 2 | 22:41:53 | 4205 | 4197 | 8 | 218.9 | 37 | 119 | 2.839 s | 45.4 | tool_calls | 1.975 |
| 3 | 22:41:55 | 4414 | 4308 | 106 | 336.7 | 315 | 77 | 2.037 s | 45.3 | tool_calls | 1.923 |
| 4 | 22:42:27 | 7687 | 4414 | 3273 | 4573.1 | 716 | 994 | 32.341 s | 35.8 | stop | 1.706 |
| 5 | 22:42:35 | 8370 | 7687 | 683 | 2032.8 | 336 | 109 | 5.396 s | 32.4 | tool_calls | 1.725 |
| 6 | 22:42:51 | 8773 | 8457 | 316 | 920.2 | 343 | 534 | 15.823 s | 35.8 | stop | 1.675 |
| 7 | 22:43:00 | 9098 | 8773 | 325 | 519.3 | 626 | 259 | 7.641 s | 36.4 | tool_calls | 1.846 |
| 8 | 22:43:37 | 20245 | 9252 | 10993 | 17857.6 | 616 | 457 | 37.121 s | 23.7 | stop | 1.673 |

- 1 本目はプロセス起動後の最初のリクエスト。
- 全 8 本が完了し、`failed` の行は無い。測定の後 (停止前) の M6 の `vm.swapusage` は used 92.25 MB (測定中の Swapouts は取っていない)。
- 取っていないもの: Pageins、expert io、GPU の時間。

## 2. 残るもの

- 同じ会話を変更前の commit / 写しなしで流した値は無い。
