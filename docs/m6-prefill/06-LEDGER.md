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
| 2026-09-24 | ef29110 | ledger-trusted | 6 | 2.897 | 2.880 | 3.123 | 0 |

**読み (2026-09-24)**: 表は `--verification trusted-install` で取った行だけ (ef29110 から、[35](35-PREFILL-WITHOUT-VERIFY.md))。ef29110 より前の行は層ファイルの SHA-256 検証を含む量で、`ledger.sh` は載せない。ef29110 の 2.897 s が、この量の最初の行。

## 2. 既定でない構成 (候補。進捗ではない)

`prefill_bench.sh` で取った全ディレクトリ。profile 付き (`TF_PREFILL_*_PROFILE`) は計測で膨らむ。greedy (`--max-new 64`) と失敗 (exit ≠ 0) も残す。

| 日付 | commit | ラベル | n | prefill 壁時計 中央値 s | 最小 | 最大 | 失敗 | TF_ (無ければ既定) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-24 | 973a9b1 | f2-idle-trusted | 0 | - | - | - | 3 | TF_PREFILL_HOST_PROFILE=1 |
| 2026-09-24 | ef29110 | ledger-trusted | 6 | 2.897 | 2.880 | 3.123 | 0 | 既定 |
| 2026-09-24 | 29948fd | s2-trusted-mpp64 | 0 | - | - | - | 3 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 29948fd | s2-trusted-simdgroup | 0 | - | - | - | 3 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=simdgroup |
| 2026-09-24 | ef29110 | trusted-host | 3 | 2.880 | 2.879 | 2.992 | 0 | TF_PREFILL_HOST_PROFILE=1 |
