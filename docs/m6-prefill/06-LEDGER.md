# 06. 台帳 — 進捗は既定行の秒だけ

**この文書の表は `bench/m6/ledger.sh` の出力を貼ったもので、手で行を書かない。**
prefill について「速くなった」と言うときは、必ず §1 の行を指す。指せない主張は進捗ではない。
2026-09-24 にこう決めた理由は [18](18-RESET-ABSOLUTE-FLOOR.md)。

## 0. 規則

1. **測る量は 1 つ。**M6、既定のバイナリ (`TF_` 環境変数なし、追加引数なし)、`prefill_bench.sh` の固定コマンド、新しいプロセスの 1 回目の prefill 壁時計 (秒)。
   ラベルは `ledger-<何>`。`prefill_bench.sh` は `ledger-*` に `TF_` か追加引数が付いていると測らずに止まる。
2. **比の列は無い。**倍率・pp・%・「下限の何倍か」は、この文書にも README の現在地にも書かない。変更の効果は秒の差で書く。
3. **行が増えるのは証拠ディレクトリができたときだけ。**同じ commit で n ≥ 6。表は `bench/m6/ledger.sh` で作り直して貼る (§2 は `--all`)。
4. **`TF_` 付きの構成は候補であって進捗ではない** (§2)。既定に入れる判断はユーザーがする ([04 B3](04-GATES.md))。既定に入ったら、次の `ledger-*` の行が動いたかどうかでだけ確かめる。
5. **数字の中身**: 起動時の層ファイルの SHA-256 検証 (CPU 約 4.5 s、一部は GPU と重なる) と空の expert キャッシュを含む。[04 B0](04-GATES.md) の (c)。同じプロセスの 2 回目 (b) の列は、測る口ができてから足す。それまで (b) の数字は存在しない。

## 1. 既定 (進捗はここだけ)

| 日付 | commit | ラベル | n | prefill 壁時計 中央値 s | 最小 | 最大 | 失敗 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-09-23 | 837586c | g0-baseline | 6 | 9.506 | 9.188 | 9.597 | 0 |
| 2026-09-24 | 22d7629 | ledger-head | 6 | 8.682 | 8.633 | 8.801 | 0 |

**読み (2026-09-24)**: HEAD (22d7629) の既定は 8.682 s。基準線 (837586c) の 9.506 s から差 −0.824 s。どの commit で動いたかは未確認 (837586c..22d7629 の Sources の変更はすべて `TF_` の裏のはず)。

## 2. 既定でない構成 (候補。進捗ではない)

`prefill_bench.sh` で取った全ディレクトリ。profile 付き (`TF_PREFILL_*_PROFILE`) は計測で膨らむ。greedy (`--max-new 64`) と失敗 (exit ≠ 0) も残す。

| 日付 | commit | ラベル | n | prefill 壁時計 中央値 s | 最小 | 最大 | 失敗 | TF_ (無ければ既定) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-23 | 837586c | g0-baseline | 6 | 9.506 | 9.188 | 9.597 | 0 | 既定 |
| 2026-09-23 | 837586c | g0-profile-detail | 3 | 9.622 | 9.434 | 9.742 | 0 | TF_PREFILL_GPU_PROFILE=2 |
| 2026-09-23 | 837586c | g0-profile | 3 | 9.593 | 9.158 | 9.630 | 0 | TF_PREFILL_GPU_PROFILE=1 |
| 2026-09-23 | 906eae2 | g1-greedy-mpp | 2 | 9.657 | 9.587 | 9.728 | 0 | TF_PREFILL_GPU_PROFILE=1 TF_PREFILL_QMM=mpp 追加引数: --max-new 64 --temperature 0 |
| 2026-09-23 | 906eae2 | g1-greedy-simdgroup | 2 | 9.624 | 9.503 | 9.745 | 0 | TF_PREFILL_GPU_PROFILE=1 TF_PREFILL_QMM=simdgroup 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | 67a087a | a0-profile-detail | 3 | 4.624 | 4.590 | 5.151 | 0 | TF_PREFILL_GPU_PROFILE=2 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 3f042e7 | best4-a | 3 | 4.578 | 4.545 | 5.181 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 3f042e7 | best4-b | 3 | 4.551 | 4.473 | 4.582 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 3f042e7 | best4-sgmm-a | 3 | 4.629 | 4.227 | 5.203 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 3f042e7 | best4-sgmm-b | 3 | 4.146 | 4.110 | 4.353 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 7c7ad7b | g3-greedy-mpp | 2 | 7.596 | 7.552 | 7.641 | 0 | TF_PREFILL_MOE=mpp 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | 7c7ad7b | g3-greedy-simdgroup | 2 | 9.034 | 8.787 | 9.280 | 0 | TF_PREFILL_MOE=simdgroup 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | 7c7ad7b | g3-mpp-a | 3 | 7.345 | 7.317 | 7.360 | 0 | TF_PREFILL_MOE=mpp |
| 2026-09-24 | 7c7ad7b | g3-mpp-b | 3 | 7.450 | 7.407 | 7.621 | 0 | TF_PREFILL_MOE=mpp |
| 2026-09-24 | 7c7ad7b | g3-profile-mpp | 3 | 7.271 | 7.207 | 7.300 | 0 | TF_PREFILL_GPU_PROFILE=1 TF_PREFILL_MOE=mpp |
| 2026-09-24 | 7c7ad7b | g3-profile-simdgroup | 3 | 8.712 | 8.648 | 8.896 | 0 | TF_PREFILL_GPU_PROFILE=1 TF_PREFILL_MOE=simdgroup |
| 2026-09-24 | 7c7ad7b | g3-simdgroup-a | 3 | 8.978 | 8.746 | 9.067 | 0 | TF_PREFILL_MOE=simdgroup |
| 2026-09-24 | 7c7ad7b | g3-simdgroup-b | 3 | 9.032 | 8.986 | 9.082 | 0 | TF_PREFILL_MOE=simdgroup |
| 2026-09-24 | 8dee853 | g5-ahead0-a | 3 | 7.259 | 7.106 | 7.264 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD=0 |
| 2026-09-24 | 8dee853 | g5-ahead0-b | 3 | 7.245 | 7.214 | 7.444 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD=0 |
| 2026-09-24 | 8dee853 | g5-ahead2-a | 3 | 5.948 | 5.947 | 6.007 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 8dee853 | g5-ahead2-b | 3 | 5.945 | 5.841 | 6.026 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 8dee853 | g5-greedy-ahead2 | 2 | 6.094 | 6.024 | 6.163 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD=2 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | 8dee853 | g5-host-ahead0 | 3 | 7.253 | 7.073 | 7.341 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD=0 |
| 2026-09-24 | 8dee853 | g5-host-ahead1 | 3 | 6.055 | 5.982 | 6.139 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD=1 |
| 2026-09-24 | 8dee853 | g5-host-ahead2 | 3 | 5.901 | 5.872 | 6.023 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 8dee853 | g5-host-ahead4 | 3 | 5.918 | 5.915 | 5.923 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD=4 |
| 2026-09-24 | 3c45f83 | g5-host-par-ahead2 | 3 | 4.654 | 4.626 | 5.341 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 3c45f83 | g5-host-par-ahead4 | 3 | 4.525 | 4.396 | 5.017 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=4 |
| 2026-09-24 | 3c45f83 | g5c-ahead0-a | 3 | 6.980 | 6.935 | 7.006 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=0 TF_PREFILL_VERIFY_AHEAD=0 |
| 2026-09-24 | 3c45f83 | g5c-ahead0-b | 3 | 6.984 | 6.927 | 6.995 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=0 TF_PREFILL_VERIFY_AHEAD=0 |
| 2026-09-24 | 3c45f83 | g5c-greedy-par2 | 2 | 4.765 | 4.608 | 4.922 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | 3c45f83 | g5c-par2-a | 3 | 4.443 | 4.421 | 4.577 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 3c45f83 | g5c-par2-b | 3 | 4.407 | 4.397 | 4.972 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 |
| 2026-09-24 | 3c45f83 | g5c-par4-a | 3 | 4.420 | 4.358 | 4.833 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=4 |
| 2026-09-24 | 3c45f83 | g5c-par4-b | 3 | 4.463 | 4.418 | 4.527 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=4 |
| 2026-09-24 | 22d7629 | h0-profile-detail | 3 | 8.754 | 8.646 | 8.842 | 0 | TF_PREFILL_GPU_PROFILE=2 |
| 2026-09-24 | 22d7629 | ledger-head | 6 | 8.682 | 8.633 | 8.801 | 0 | 既定 |
| 2026-09-24 | 29948fd | s2-greedy-mpp64 | 2 | 7.409 | 7.153 | 7.664 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | 29948fd | s2-host-mpp64 | 3 | 7.342 | 7.228 | 7.357 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 29948fd | s2-host-simdgroup | 3 | 7.298 | 7.101 | 7.319 | 0 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=simdgroup |
| 2026-09-24 | 29948fd | s2-mpp64-a | 3 | 7.243 | 7.115 | 7.330 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 29948fd | s2-mpp64-b | 3 | 7.211 | 7.143 | 7.262 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 29948fd | s2-profile-mpp64 | 3 | 7.329 | 7.304 | 7.426 | 0 | TF_PREFILL_GPU_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 29948fd | s2-profile-simdgroup | 3 | 7.407 | 7.243 | 7.430 | 0 | TF_PREFILL_GPU_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=simdgroup |
| 2026-09-24 | 29948fd | s2-simdgroup-a | 3 | 7.275 | 7.198 | 7.279 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=simdgroup |
| 2026-09-24 | 29948fd | s2-simdgroup-b | 3 | 7.336 | 7.322 | 7.397 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=simdgroup |
| 2026-09-24 | 29948fd | s2-trusted-mpp64 | 0 | - | - | - | 3 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 追加引数: --verification trusted-install |
| 2026-09-24 | 29948fd | s2-trusted-simdgroup | 0 | - | - | - | 3 | TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_MOE=mpp TF_PREFILL_QMM=simdgroup 追加引数: --verification trusted-install |
| 2026-09-24 | 3f042e7 | sgmm-a | 6 | 8.829 | 8.626 | 8.957 | 0 | TF_PREFILL_ATTN=sgmm |
| 2026-09-24 | 3f042e7 | sgmm-profile-detail-2 | 3 | 8.742 | 8.739 | 8.951 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_GPU_PROFILE=2 |
| 2026-09-24 | 71786c5 | sgmm-profile-detail | 3 | 8.626 | 8.563 | 8.810 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_GPU_PROFILE=2 |
