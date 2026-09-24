# 06. 台帳 — 進捗は既定行の秒だけ

**この文書の表は `bench/m6/ledger.sh` の出力を貼ったもので、手で行を書かない。**
prefill について「速くなった」と言うときは、必ず §1 の行を指す。指せない主張は進捗ではない。
2026-09-24 にこう決めた理由は [18](18-RESET-ABSOLUTE-FLOOR.md)。

## 0. 規則

1. **測る量は 1 つ。**M6、既定のバイナリ (`TF_` 環境変数なし、追加引数なし)、`prefill_bench.sh` の固定コマンド、新しいプロセスの 1 回目の prefill 壁時計 (秒)。
   ラベルは `ledger-<何>`。`prefill_bench.sh` は `ledger-*` に `TF_` か追加引数が付いていると測らずに止まる。
   **本文は自然文の組** (`bench/m6/prompts/` の 5 本、[43 §3](43-LEDGER-PROMPT-REPEATED-SENTENCE.md)) で、run ごとに本文を替える。行は本文ごとの中央値を並べる。
   §1 の既存の行はすべて `bench/l.json` (同じ 1 文の 50 回繰り返し) で取ったもので、繰り返し文の履歴として残し、進捗と目標の判定には使わない。`prefill_bench.sh` と `ledger.sh` はまだ `bench/l.json` 固定で、この形は未実装。
2. **比の列は無い。**倍率・pp・%・「下限の何倍か」は、この文書にも README の現在地にも書かない。変更の効果は秒の差で書く。
3. **行が増えるのは証拠ディレクトリができたときだけ。**同じ commit で n ≥ 6。表は `bench/m6/ledger.sh` で作り直して貼る (§2 は `--all`)。
4. **`TF_` 付きの構成は候補であって進捗ではない** (§2)。検査を通って秒が縮んだ候補は既定に入れる ([04 原則 2](04-GATES.md))。既定に入ったら、次の `ledger-*` の行が動いたかどうかでだけ確かめる。
5. **数字の中身**: 層ファイルの SHA-256 検証は含まない (`--verification trusted-install`、[35](35-PREFILL-WITHOUT-VERIFY.md))。空の expert キャッシュと、SSD からの読みは含む。[04 B0](04-GATES.md) の (c)。
   読む量は本文で変わる (繰り返し文 `bench/l.json` は 1 run 3.0〜3.7 GiB、自然文は 5.9〜12.0 GiB、[43 §1](43-LEDGER-PROMPT-REPEATED-SENTENCE.md))。同じ本文を続けて流すとその本文のページがキャッシュに残るので、行には 1 run の Pageins と expert io を添える。
   同じプロセスの 2 回目 (b) の列は、測る口ができてから足す。それまで (b) の数字は存在しない。

## 1. 既定 (進捗はここだけ)

| 日付 | commit | ラベル | n | prefill 壁時計 中央値 s | 最小 | 最大 | 失敗 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-09-24 | c08e2de | ledger-full-attn-mpp | 6 | 2.740 | 2.722 | 3.110 | 0 |
| 2026-09-24 | ef29110 | ledger-trusted | 6 | 2.897 | 2.880 | 3.123 | 0 |
| 2026-09-25 | 619c515 | ledger-advise-layer | 6 | 1.768 | 1.747 | 1.922 | 0 |
| 2026-09-25 | 9d66daa | ledger-expert-mirror | 6 | 1.716 | 1.706 | 3.747 | 0 |
| 2026-09-25 | 084de6b | ledger-full-attn-unfused | 6 | 2.626 | 2.617 | 3.030 | 0 |
| 2026-09-25 | 275d0f5 | ledger-moe-nax | 6 | 1.914 | 1.887 | 2.431 | 0 |
| 2026-09-25 | 122e144 | ledger-router-nax | 6 | 2.267 | 2.252 | 2.744 | 0 |
| 2026-09-25 | 994a0ef | ledger-swa-unfused | 6 | 2.482 | 2.471 | 2.984 | 0 |

**読み (2026-09-25、[44](44-EXPERT-MIRROR-EXTERNAL-SSD.md))**: 9d66daa (1.716 s) から、M6 の既定は外付け SSD の写し (`expert-mirror.json`) からエキスパートの 32% を読む。行の証拠ディレクトリに `expert-mirror.json` が写っている。写しなしの値は `TF_EXPERT_MIRROR=0` の候補として取る。繰り返し文なので、この行の縮みは進捗の判定に使わない (自然文の比較は 44 §4)。
**読み (2026-09-25、[43](43-LEDGER-PROMPT-REPEATED-SENTENCE.md))**: 上の行はすべて繰り返し文 `bench/l.json` の値。自然文の既定は 2.595〜4.568 s (`w1-def-*`、台帳の形ではまだ取っていない)。以下は繰り返し文の中での経緯。
**読み (2026-09-24)**: 表は `--verification trusted-install` で取った行だけ (ef29110 から、[35](35-PREFILL-WITHOUT-VERIFY.md))。ef29110 より前の行は層ファイルの SHA-256 検証を含む量で、`ledger.sh` は載せない。ef29110 の 2.897 s が、この量の最初の行。
c08e2de (2.740 s) と 084de6b (2.626 s) は全注意の 2 回の既定の変更 ([36](36-FULL-ATTN-UNFUSED.md))、994a0ef (2.482 s) は SWA の変更 ([37](37-SWA-ATTN-UNFUSED.md))、122e144 (2.267 s) は router の変更 ([38](38-ROUTER-NAX.md))、275d0f5 (1.914 s) は routed MoE の GEMM の変更 ([40](40-MOE-GEMM-NAX.md))、619c515 (1.768 s) は層ごとの先読みの頼み方の変更 ([42](42-ADVISE-LAYER-AT-ROUTING.md))。この測定環境では同じ構成の n = 6 の中央値が 0.05 s 程度揺れる (36 §2)。

## 2. 既定でない構成 (候補。進捗ではない)

`prefill_bench.sh` で取った全ディレクトリ。profile 付き (`TF_PREFILL_*_PROFILE`) は計測で膨らむ。greedy (`--max-new 64`) と失敗 (exit ≠ 0) も残す。

| 日付 | commit | ラベル | n | prefill 壁時計 中央値 s | 最小 | 最大 | 失敗 | TF_ (無ければ既定) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-24 | 4ea5dd0 | a1-default-a | 3 | 2.906 | 2.875 | 3.135 | 0 | 既定 |
| 2026-09-24 | 1cfa97a | a1-default-b | 3 | 2.912 | 2.842 | 2.953 | 0 | 既定 |
| 2026-09-24 | 9cb18d2 | a1-greedy-default | 1 | 3.017 | 3.017 | 3.017 | 0 | 既定 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | 9cb18d2 | a1-greedy-mpp64 | 1 | 3.139 | 3.139 | 3.139 | 0 | TF_PREFILL_ATTN_FULL=mpp64 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | ebbd919 | a1-mpp64-a | 3 | 2.809 | 2.715 | 2.913 | 0 | TF_PREFILL_ATTN_FULL=mpp64 |
| 2026-09-24 | b6ba63e | a1-mpp64-b | 3 | 2.773 | 2.738 | 2.883 | 0 | TF_PREFILL_ATTN_FULL=mpp64 |
| 2026-09-24 | 4b9f660 | a2-default-a | 3 | 2.769 | 2.765 | 3.021 | 0 | 既定 |
| 2026-09-24 | b663a5d | a2-default-b | 3 | 2.735 | 2.734 | 2.863 | 0 | 既定 |
| 2026-09-24 | 402f21e | a2-host-default | 3 | 2.756 | 2.739 | 2.873 | 0 | TF_PREFILL_HOST_PROFILE=1 |
| 2026-09-24 | 305edd8 | a2-host-unfused | 3 | 2.801 | 2.778 | 2.873 | 0 | TF_PREFILL_ATTN_FULL=unfused TF_PREFILL_HOST_PROFILE=1 |
| 2026-09-24 | 86be1e2 | a2-unfused-a | 3 | 2.848 | 2.794 | 2.878 | 0 | TF_PREFILL_ATTN_FULL=unfused |
| 2026-09-24 | 445bf40 | a2-unfused-b | 3 | 2.744 | 2.729 | 2.864 | 0 | TF_PREFILL_ATTN_FULL=unfused |
| 2026-09-24 | 3a28a91 | a3-host-unfused | 3 | 2.774 | 2.626 | 2.785 | 0 | TF_PREFILL_ATTN_FULL=unfused TF_PREFILL_HOST_PROFILE=1 |
| 2026-09-24 | 973a9b1 | f2-idle-trusted | 0 | - | - | - | 3 | TF_PREFILL_HOST_PROFILE=1 |
| 2026-09-24 | c08e2de | ledger-full-attn-mpp | 6 | 2.740 | 2.722 | 3.110 | 0 | 既定 |
| 2026-09-24 | ef29110 | ledger-trusted | 6 | 2.897 | 2.880 | 3.123 | 0 | 既定 |
| 2026-09-24 | 29948fd | s2-trusted-mpp64 | 0 | - | - | - | 3 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 29948fd | s2-trusted-simdgroup | 0 | - | - | - | 3 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=simdgroup |
| 2026-09-24 | ef29110 | trusted-host | 3 | 2.880 | 2.879 | 2.992 | 0 | TF_PREFILL_HOST_PROFILE=1 |
| 2026-09-25 | d7b08f5 | a3-default-a | 3 | 2.778 | 2.734 | 2.920 | 0 | 既定 |
| 2026-09-25 | 94f4a65 | a3-default-b | 3 | 2.756 | 2.724 | 2.852 | 0 | 既定 |
| 2026-09-25 | c3c31ab | a3-greedy-unfused | 1 | 2.760 | 2.760 | 2.760 | 0 | TF_PREFILL_ATTN_FULL=unfused 追加引数: --max-new 64 --temperature 0 |
| 2026-09-25 | 90ea97a | a3-unfused-a | 3 | 2.702 | 2.629 | 2.765 | 0 | TF_PREFILL_ATTN_FULL=unfused |
| 2026-09-25 | a3ee85d | a3-unfused-b | 3 | 2.669 | 2.607 | 2.778 | 0 | TF_PREFILL_ATTN_FULL=unfused |
| 2026-09-25 | 44b3c4f | e1-default-a | 3 | 2.330 | 2.322 | 2.464 | 0 | 既定 |
| 2026-09-25 | b8c728d | e1-default-b | 3 | 2.318 | 2.253 | 2.434 | 0 | 既定 |
| 2026-09-25 | 2993068 | e1-fused-a | 3 | 2.297 | 2.264 | 2.448 | 0 | TF_PREFILL_QKV_EPILOGUE=fused |
| 2026-09-25 | 8de0104 | e1-fused-b | 3 | 2.283 | 2.248 | 2.389 | 0 | TF_PREFILL_QKV_EPILOGUE=fused |
| 2026-09-25 | 5f22a7c | e1-host-fused | 3 | 2.342 | 2.286 | 2.425 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_QKV_EPILOGUE=fused |
| 2026-09-25 | 385d6d6 | l1-advise-a | 3 | 1.877 | 1.815 | 1.920 | 0 | TF_PREFILL_ADVISE_LAYER=1 |
| 2026-09-25 | ad61bf8 | l1-advise-b | 3 | 1.836 | 1.796 | 1.965 | 0 | TF_PREFILL_ADVISE_LAYER=1 |
| 2026-09-25 | 16ec44a | l1-default-a | 3 | 1.941 | 1.912 | 2.051 | 0 | 既定 |
| 2026-09-25 | 8a403b7 | l1-default-b | 3 | 1.933 | 1.893 | 2.009 | 0 | 既定 |
| 2026-09-25 | 6dbeba0 | l1-greedy-default | 2 | 2.034 | 1.973 | 2.096 | 0 | 既定 追加引数: --max-new 64 --temperature 0 |
| 2026-09-25 | a229299 | l1-host-advise | 3 | 1.891 | 1.819 | 2.209 | 0 | TF_PREFILL_ADVISE_LAYER=1 TF_PREFILL_HOST_PROFILE=1 |
| 2026-09-25 | 619c515 | ledger-advise-layer | 6 | 1.768 | 1.747 | 1.922 | 0 | 既定 |
| 2026-09-25 | 9d66daa | ledger-expert-mirror | 6 | 1.716 | 1.706 | 3.747 | 0 | 既定 |
| 2026-09-25 | 084de6b | ledger-full-attn-unfused | 6 | 2.626 | 2.617 | 3.030 | 0 | 既定 |
| 2026-09-25 | 275d0f5 | ledger-moe-nax | 6 | 1.914 | 1.887 | 2.431 | 0 | 既定 |
| 2026-09-25 | 122e144 | ledger-router-nax | 6 | 2.267 | 2.252 | 2.744 | 0 | 既定 |
| 2026-09-25 | 994a0ef | ledger-swa-unfused | 6 | 2.482 | 2.471 | 2.984 | 0 | 既定 |
| 2026-09-25 | 2d9c7a4 | m1-default-a | 3 | 2.360 | 2.318 | 2.529 | 0 | 既定 |
| 2026-09-25 | 629c885 | m1-default-b | 3 | 2.299 | 2.255 | 2.402 | 0 | 既定 |
| 2026-09-25 | dd8a007 | m1-greedy-nax | 2 | 2.146 | 2.005 | 2.287 | 0 | TF_PREFILL_MOE_NAX=1 追加引数: --max-new 64 --temperature 0 |
| 2026-09-25 | a6d5b7c | m1-host-nax | 3 | 1.939 | 1.901 | 2.021 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE_NAX=1 |
| 2026-09-25 | dbdbfc0 | m1-nax-a | 3 | 1.960 | 1.915 | 2.132 | 0 | TF_PREFILL_MOE_NAX=1 |
| 2026-09-25 | 905f2de | m1-nax-b | 3 | 1.935 | 1.911 | 2.037 | 0 | TF_PREFILL_MOE_NAX=1 |
| 2026-09-25 | 3184baf | n1-profile-detail | 3 | 1.938 | 1.891 | 2.053 | 0 | TF_PREFILL_GPU_PROFILE=2 |
| 2026-09-25 | adaf3ef | n1-profile | 3 | 1.957 | 1.898 | 2.070 | 0 | TF_PREFILL_GPU_PROFILE=1 |
| 2026-09-25 | f96892d | p1-one-chunk | 3 | 1.668 | 1.620 | 1.798 | 0 | TF_BENCH_MESSAGES=/Users/mh/dev/tsugumi/bench/m6/l-1chunk.json |
| 2026-09-25 | f3e951b | r1-default-a | 3 | 2.541 | 2.446 | 2.676 | 0 | 既定 |
| 2026-09-25 | 401ad5e | r1-default-b | 3 | 2.617 | 2.489 | 2.631 | 0 | 既定 |
| 2026-09-25 | 63438f0 | r1-greedy-nax | 2 | 2.531 | 2.430 | 2.632 | 0 | TF_PREFILL_ROUTER=nax 追加引数: --max-new 64 --temperature 0 |
| 2026-09-25 | c168bd4 | r1-host-nax | 3 | 2.291 | 2.274 | 2.390 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_ROUTER=nax |
| 2026-09-25 | 0ee801c | r1-nax-a | 3 | 2.315 | 2.258 | 2.381 | 0 | TF_PREFILL_ROUTER=nax |
| 2026-09-25 | d350bbf | r1-nax-b | 3 | 2.292 | 2.268 | 2.396 | 0 | TF_PREFILL_ROUTER=nax |
| 2026-09-25 | c0c2469 | s1-default-a | 3 | 2.733 | 2.637 | 2.853 | 0 | 既定 |
| 2026-09-25 | ed16567 | s1-default-b | 3 | 2.762 | 2.632 | 2.842 | 0 | 既定 |
| 2026-09-25 | eb5e1f6 | s1-greedy-default | 2 | 2.720 | 2.629 | 2.811 | 0 | 既定 追加引数: --max-new 64 --temperature 0 |
| 2026-09-25 | 983fb02 | s1-greedy-unfused | 1 | 2.569 | 2.569 | 2.569 | 0 | TF_PREFILL_ATTN_SWA=unfused 追加引数: --max-new 64 --temperature 0 |
| 2026-09-25 | 312c2cb | s1-host-unfused | 3 | 2.533 | 2.429 | 2.573 | 0 | TF_PREFILL_ATTN_SWA=unfused TF_PREFILL_HOST_PROFILE=1 |
| 2026-09-25 | 0fdb98d | s1-unfused-a | 3 | 2.554 | 2.545 | 2.626 | 0 | TF_PREFILL_ATTN_SWA=unfused |
| 2026-09-25 | 78fbeca | s1-unfused-b | 3 | 2.476 | 2.416 | 2.624 | 0 | TF_PREFILL_ATTN_SWA=unfused |
| 2026-09-25 | 486f102 | u1-profile-detail | 3 | 2.440 | 2.438 | 2.556 | 0 | TF_PREFILL_GPU_PROFILE=2 |
| 2026-09-25 | 5430e0f | u1-profile | 3 | 2.589 | 2.425 | 2.938 | 0 | TF_PREFILL_GPU_PROFILE=1 |
| 2026-09-25 | ee0d30d | v1-ahead1-a | 3 | 2.633 | 2.548 | 2.837 | 0 | TF_PREFILL_ADVISE_AHEAD=1 |
| 2026-09-25 | 8198b1c | v1-ahead1-b | 3 | 2.614 | 2.506 | 2.785 | 0 | TF_PREFILL_ADVISE_AHEAD=1 |
| 2026-09-25 | 516a50a | v1-ahead2-a | 3 | 2.757 | 2.636 | 2.835 | 0 | TF_PREFILL_ADVISE_AHEAD=2 |
| 2026-09-25 | dab01c1 | v1-ahead2-b | 3 | 2.759 | 2.606 | 2.911 | 0 | TF_PREFILL_ADVISE_AHEAD=2 |
| 2026-09-25 | c4f33ea | v1-default-a | 3 | 1.937 | 1.909 | 2.046 | 0 | 既定 |
| 2026-09-25 | 104af99 | v1-default-b | 3 | 1.952 | 1.926 | 2.071 | 0 | 既定 |
| 2026-09-25 | 3537b23 | v2-host-ahead1 | 3 | 2.719 | 2.431 | 2.981 | 0 | TF_PREFILL_ADVISE_AHEAD=1 TF_PREFILL_HOST_PROFILE=1 |
| 2026-09-25 | 6dbeba0 | w1-def-code-runner | 4 | 5.483 | 4.502 | 6.592 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/code-runner.json |
| 2026-09-25 | 6dbeba0 | w1-def-en-readme | 4 | 4.407 | 4.375 | 4.519 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/en-readme.json |
| 2026-09-25 | 6dbeba0 | w1-def-en-server | 4 | 4.568 | 4.496 | 5.730 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/en-server.json |
| 2026-09-25 | 6dbeba0 | w1-def-ja-m6readme | 4 | 3.441 | 3.116 | 3.829 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/ja-m6readme.json |
| 2026-09-25 | 6dbeba0 | w1-def-ja-twomachine | 4 | 2.595 | 2.503 | 2.618 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/ja-twomachine.json |
| 2026-09-25 | 6dbeba0 | w1-def-l | 4 | 1.885 | 1.803 | 2.505 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/l.json |
| 2026-09-25 | 6dbeba0 | w1-noadv-code-runner | 4 | 4.861 | 4.822 | 4.959 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/code-runner.json TF_PREFILL_ADVISE_LAYER=0 |
| 2026-09-25 | 6dbeba0 | w1-noadv-en-readme | 4 | 4.798 | 4.791 | 4.836 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/en-readme.json TF_PREFILL_ADVISE_LAYER=0 |
| 2026-09-25 | 6dbeba0 | w1-noadv-en-server | 4 | 4.864 | 4.803 | 4.983 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/en-server.json TF_PREFILL_ADVISE_LAYER=0 |
| 2026-09-25 | 6dbeba0 | w1-noadv-ja-m6readme | 4 | 4.098 | 4.065 | 4.172 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/ja-m6readme.json TF_PREFILL_ADVISE_LAYER=0 |
| 2026-09-25 | 6dbeba0 | w1-noadv-ja-twomachine | 4 | 2.794 | 2.755 | 2.964 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/ja-twomachine.json TF_PREFILL_ADVISE_LAYER=0 |
| 2026-09-25 | 6dbeba0 | w1-noadv-l | 4 | 1.919 | 1.911 | 1.925 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/l.json TF_PREFILL_ADVISE_LAYER=0 |
| 2026-09-25 | 6dbeba0 | w1-old-code-runner | 4 | 5.498 | 5.422 | 7.093 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/code-runner.json TF_PREFILL_ADVISE_LAYER=0 TF_PREFILL_ATTN_FULL=qblock TF_PREFILL_ATTN_SWA=mpp TF_PREFILL_MOE_NAX=0 TF_PREFILL_ROUTER=gemm |
| 2026-09-25 | 6dbeba0 | w1-old-en-readme | 4 | 5.444 | 5.404 | 5.520 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/en-readme.json TF_PREFILL_ADVISE_LAYER=0 TF_PREFILL_ATTN_FULL=qblock TF_PREFILL_ATTN_SWA=mpp TF_PREFILL_MOE_NAX=0 TF_PREFILL_ROUTER=gemm |
| 2026-09-25 | 6dbeba0 | w1-old-en-server | 4 | 5.444 | 5.407 | 5.458 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/en-server.json TF_PREFILL_ADVISE_LAYER=0 TF_PREFILL_ATTN_FULL=qblock TF_PREFILL_ATTN_SWA=mpp TF_PREFILL_MOE_NAX=0 TF_PREFILL_ROUTER=gemm |
| 2026-09-25 | 6dbeba0 | w1-old-ja-m6readme | 4 | 4.878 | 4.787 | 5.018 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/ja-m6readme.json TF_PREFILL_ADVISE_LAYER=0 TF_PREFILL_ATTN_FULL=qblock TF_PREFILL_ATTN_SWA=mpp TF_PREFILL_MOE_NAX=0 TF_PREFILL_ROUTER=gemm |
| 2026-09-25 | 6dbeba0 | w1-old-ja-twomachine | 4 | 3.859 | 3.790 | 3.951 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/ja-twomachine.json TF_PREFILL_ADVISE_LAYER=0 TF_PREFILL_ATTN_FULL=qblock TF_PREFILL_ATTN_SWA=mpp TF_PREFILL_MOE_NAX=0 TF_PREFILL_ROUTER=gemm |
| 2026-09-25 | 6dbeba0 | w1-old-l | 4 | 2.869 | 2.855 | 2.907 | 0 | TF_BENCH_MESSAGES=/Users/mh/m6-warm-prompts/l.json TF_PREFILL_ADVISE_LAYER=0 TF_PREFILL_ATTN_FULL=qblock TF_PREFILL_ATTN_SWA=mpp TF_PREFILL_MOE_NAX=0 TF_PREFILL_ROUTER=gemm |
