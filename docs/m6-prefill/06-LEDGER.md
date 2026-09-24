# 06. 台帳 — 進捗は既定行の秒だけ

**この文書の表は `bench/m6/ledger.sh` の出力を貼ったもので、手で行を書かない。**
prefill について「速くなった」と言うときは、必ず §1 の行を指す。指せない主張は進捗ではない。
2026-09-24 にこう決めた理由は [18](18-RESET-ABSOLUTE-FLOOR.md)。

## 0. 規則

1. **測る量は 1 つ。**M6、既定のバイナリ (`TF_` 環境変数なし、追加引数なし)、`prefill_bench.sh` の固定コマンド、新しいプロセスの 1 回目の prefill 壁時計 (秒)。
   ラベルは `ledger-<何>`。`prefill_bench.sh` は `ledger-*` に `TF_` か追加引数が付いていると測らずに止まる。
2. **比の列は無い。**倍率・pp・%・「下限の何倍か」は、この文書にも README の現在地にも書かない。変更の効果は秒の差で書く。
3. **行が増えるのは証拠ディレクトリができたときだけ。**同じ commit で n ≥ 6。表は `bench/m6/ledger.sh` で作り直して貼る (§2 は `--all`)。
4. **`TF_` 付きの構成は候補であって進捗ではない** (§2)。検査を通って秒が縮んだ候補は既定に入れる ([04 原則 2](04-GATES.md))。既定に入ったら、次の `ledger-*` の行が動いたかどうかでだけ確かめる。
5. **数字の中身**: 層ファイルの SHA-256 検証は含まない (`--verification trusted-install`、[35](35-PREFILL-WITHOUT-VERIFY.md))。空の expert キャッシュと、その run のページキャッシュの状態 (M6 では 1 run あたり SSD から 3.0〜3.7 GiB 読む) は含む。[04 B0](04-GATES.md) の (c)。同じプロセスの 2 回目 (b) の列は、測る口ができてから足す。それまで (b) の数字は存在しない。

## 1. 既定 (進捗はここだけ)

| 日付 | commit | ラベル | n | prefill 壁時計 中央値 s | 最小 | 最大 | 失敗 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-09-24 | c08e2de | ledger-full-attn-mpp | 6 | 2.740 | 2.722 | 3.110 | 0 |
| 2026-09-24 | ef29110 | ledger-trusted | 6 | 2.897 | 2.880 | 3.123 | 0 |
| 2026-09-25 | 084de6b | ledger-full-attn-unfused | 6 | 2.626 | 2.617 | 3.030 | 0 |
| 2026-09-25 | 122e144 | ledger-router-nax | 6 | 2.267 | 2.252 | 2.744 | 0 |
| 2026-09-25 | 994a0ef | ledger-swa-unfused | 6 | 2.482 | 2.471 | 2.984 | 0 |

**読み (2026-09-24)**: 表は `--verification trusted-install` で取った行だけ (ef29110 から、[35](35-PREFILL-WITHOUT-VERIFY.md))。ef29110 より前の行は層ファイルの SHA-256 検証を含む量で、`ledger.sh` は載せない。ef29110 の 2.897 s が、この量の最初の行。
c08e2de (2.740 s) と 084de6b (2.626 s) は全注意の 2 回の既定の変更 ([36](36-FULL-ATTN-UNFUSED.md))、994a0ef (2.482 s) は SWA の変更 ([37](37-SWA-ATTN-UNFUSED.md))、122e144 (2.267 s) は router の変更 ([38](38-ROUTER-NAX.md))。この測定環境では同じ構成の n = 6 の中央値が 0.05 s 程度揺れる (36 §2)。

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
| 2026-09-25 | 084de6b | ledger-full-attn-unfused | 6 | 2.626 | 2.617 | 3.030 | 0 | 既定 |
| 2026-09-25 | 122e144 | ledger-router-nax | 6 | 2.267 | 2.252 | 2.744 | 0 | 既定 |
| 2026-09-25 | 994a0ef | ledger-swa-unfused | 6 | 2.482 | 2.471 | 2.984 | 0 | 既定 |
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
