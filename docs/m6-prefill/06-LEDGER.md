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
| 2026-09-24 | 79eea8f | ledger-verify-ahead | 6 | 7.184 | 7.107 | 7.275 | 0 |
| 2026-09-24 | 5decec4 | ledger-gpu3-default | 6 | 4.584 | 4.545 | 6.110 | 0 |

**読み (2026-09-24)**: 既定は 5decec4 で 7.184 → 4.584 s (差 −2.600 s、導出)。変えたのは GPU 側の候補 3 つ (`mpp` / `mpp64` / `sgmm`) を既定にしたことだけ ([23](23-GPU-CANDIDATES-DEFAULT.md))。その前は 79eea8f で 8.682 → 7.184 s (層ファイルの検証の並行先読み、[20](20-VERIFY-AHEAD-ALONE.md))。基準線 (837586c) の 9.506 s から 22d7629 の 8.682 s までの −0.824 s は、どの commit で動いたか未確認。

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
| 2026-09-24 | 6791e58 | b1-host | 3 | 4.512 | 4.486 | 5.368 | 0 | TF_PREFILL_HOST_PROFILE=1 |
| 2026-09-24 | 6791e58 | b1-profile-detail | 3 | 4.523 | 4.443 | 5.523 | 0 | TF_PREFILL_GPU_PROFILE=2 |
| 2026-09-24 | 6791e58 | b1-profile | 3 | 4.563 | 4.409 | 5.402 | 0 | TF_PREFILL_GPU_PROFILE=1 |
| 2026-09-24 | 61619a9 | b1s-profile-detail | 3 | 4.518 | 4.467 | 5.515 | 0 | TF_PREFILL_GPU_PROFILE=2 |
| 2026-09-24 | 61619a9 | b1s-profile | 3 | 4.545 | 4.448 | 5.493 | 0 | TF_PREFILL_GPU_PROFILE=1 |
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
| 2026-09-24 | 3075b56 | gpu3-all3-a | 3 | 4.162 | 4.151 | 5.111 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 3075b56 | gpu3-all3-b | 3 | 4.171 | 4.067 | 4.935 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 3075b56 | gpu3-attn-a | 3 | 6.779 | 6.744 | 6.797 | 0 | TF_PREFILL_ATTN=sgmm |
| 2026-09-24 | 3075b56 | gpu3-attn-b | 3 | 6.772 | 6.611 | 6.808 | 0 | TF_PREFILL_ATTN=sgmm |
| 2026-09-24 | 3075b56 | gpu3-base-a | 3 | 7.190 | 7.153 | 7.239 | 0 | 既定 |
| 2026-09-24 | 3075b56 | gpu3-base-b | 3 | 7.120 | 7.078 | 7.260 | 0 | 既定 |
| 2026-09-24 | 3075b56 | gpu3-moe-a | 3 | 5.640 | 5.553 | 5.836 | 0 | TF_PREFILL_MOE=mpp |
| 2026-09-24 | 3075b56 | gpu3-moe-b | 3 | 5.745 | 5.603 | 5.802 | 0 | TF_PREFILL_MOE=mpp |
| 2026-09-24 | 3075b56 | gpu3-moeqmm-a | 3 | 4.559 | 4.482 | 5.194 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 3075b56 | gpu3-moeqmm-b | 3 | 4.688 | 4.584 | 5.297 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 3075b56 | gpu3-qmm-a | 3 | 6.140 | 6.020 | 6.174 | 0 | TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 3075b56 | gpu3-qmm-b | 3 | 6.064 | 5.945 | 6.145 | 0 | TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 585817a | gpu3d-def-a | 3 | 4.580 | 4.529 | 5.549 | 0 | 既定 |
| 2026-09-24 | 585817a | gpu3d-def-b | 3 | 4.527 | 4.516 | 5.771 | 0 | 既定 |
| 2026-09-24 | 585817a | gpu3d-env-a | 3 | 4.732 | 4.707 | 5.703 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 585817a | gpu3d-env-b | 3 | 4.600 | 4.556 | 5.757 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 5decec4 | gpu3d-greedy | 2 | 4.737 | 4.723 | 4.752 | 0 | 既定 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | ae4cf42 | gpu3g-all3 | 2 | 4.307 | 4.275 | 4.339 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | ae4cf42 | gpu3g-attn | 2 | 6.793 | 6.711 | 6.874 | 0 | TF_PREFILL_ATTN=sgmm 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | 60c3d33 | gpu3g-base | 2 | 7.290 | 7.186 | 7.394 | 0 | 既定 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | ae4cf42 | gpu3g-moe | 2 | 5.845 | 5.827 | 5.864 | 0 | TF_PREFILL_MOE=mpp 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | ae4cf42 | gpu3g-qmm | 2 | 6.106 | 6.085 | 6.127 | 0 | TF_PREFILL_QMM=mpp64 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | ae4cf42 | gpu3r-all3-a | 6 | 4.183 | 4.063 | 5.064 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | ae4cf42 | gpu3r-all3-b | 3 | 4.143 | 4.089 | 5.143 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | ae4cf42 | gpu3r-all3-c | 3 | 4.124 | 4.122 | 4.173 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | ae4cf42 | gpu3r-base-gap60-102553 | 1 | 6.999 | 6.999 | 6.999 | 0 | 既定 |
| 2026-09-24 | ae4cf42 | gpu3r-base-gap60-102702 | 1 | 6.944 | 6.944 | 6.944 | 0 | 既定 |
| 2026-09-24 | ae4cf42 | gpu3r-gap5-102321 | 1 | 4.214 | 4.214 | 4.214 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | ae4cf42 | gpu3r-gap5-102439 | 1 | 4.078 | 4.078 | 4.078 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | ae4cf42 | gpu3r-gap60-102309 | 1 | 4.876 | 4.876 | 4.876 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | ae4cf42 | gpu3r-gap60-102427 | 1 | 5.008 | 5.008 | 5.008 | 0 | TF_PREFILL_ATTN=sgmm TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | ae4cf42 | gpu3r-moeqmm-a | 6 | 4.588 | 4.466 | 5.307 | 0 | TF_PREFILL_MOE=mpp TF_PREFILL_QMM=mpp64 |
| 2026-09-24 | 22d7629 | h0-profile-detail | 3 | 8.754 | 8.646 | 8.842 | 0 | TF_PREFILL_GPU_PROFILE=2 |
| 2026-09-24 | 5decec4 | ledger-gpu3-default | 6 | 4.584 | 4.545 | 6.110 | 0 | 既定 |
| 2026-09-24 | 22d7629 | ledger-head | 6 | 8.682 | 8.633 | 8.801 | 0 | 既定 |
| 2026-09-24 | 79eea8f | ledger-verify-ahead | 6 | 7.184 | 7.107 | 7.275 | 0 | 既定 |
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
| 2026-09-24 | f603ade | va0-dec64-a | 3 | 8.910 | 8.582 | 8.910 | 0 | 既定 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | f603ade | va0-dec64-b | 3 | 8.957 | 8.661 | 9.352 | 0 | 既定 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | f603ade | va2p-dec64-a | 3 | 7.323 | 7.203 | 7.374 | 0 | TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 追加引数: --max-new 64 --temperature 0 |
| 2026-09-24 | f603ade | va2p-dec64-b | 3 | 7.216 | 7.197 | 7.242 | 0 | TF_PREFILL_VERIFY_AHEAD_PARALLEL=1 TF_PREFILL_VERIFY_AHEAD=2 追加引数: --max-new 64 --temperature 0 |
