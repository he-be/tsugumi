# 20. 層ファイルの検証の並行先読みを単独で — prefill 壁時計 8.910 → 7.229 s。Swapouts 0、decode と greedy の本文は同じ (実測、2026-09-24)

証拠 (M6):
[`va0-dec64-a`](../../bench/m6/results/2026-09-24-Mac18-5-va0-dec64-a/)、
[`va2p-dec64-a`](../../bench/m6/results/2026-09-24-Mac18-5-va2p-dec64-a/)、
[`va0-dec64-b`](../../bench/m6/results/2026-09-24-Mac18-5-va0-dec64-b/)、
[`va2p-dec64-b`](../../bench/m6/results/2026-09-24-Mac18-5-va2p-dec64-b/) (この順に取った)。
ホスト札 (4 ディレクトリ共通):

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit f603ade / minos 15.0
```

コマンド (4 ディレクトリ共通、差は `env.txt` の `TF_` だけ)。反復は各 3、腕あたり n = 6:

```
/Users/mh/dev/tsugumi/.build/release/TsugumiCLI --model /Users/mh/dev/tsugumi/scratch/gemma4-qat-sym.gturbo --messages-file /Users/mh/dev/tsugumi/bench/l.json --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification full-sha256 --max-new 64 --temperature 0
```

変更後の腕は `TF_PREFILL_VERIFY_AHEAD=2 TF_PREFILL_VERIFY_AHEAD_PARALLEL=1` だけ。他の候補 (`mpp`、`mpp64`、`sgmm`) は付けていない。

**台帳** ([06](06-LEDGER.md)): 候補。台帳 §1 は動かない。追加引数 (`--max-new 64 --temperature 0`) があるので、変更前の腕も §1 の行にはならない。

## 0. 結論

1. **既定の経路で律速していたのは、層ファイルの SHA-256 を 1 本ずつ直列に取る時間だった** ([14 §1](14-G5-VERIFY-AHEAD.md) の内訳と同じ)。そのため [19](19-SWA-ATTN-SGMM-GQA.md) で GPU を縮めても壁時計が動かなかった。
2. **並行先読みを単独で入れると、prefill 壁時計は 8.910 → 7.229 s (差 −1.681 s、導出)。**
3. [14](14-G5-VERIFY-AHEAD.md) が未確認として残した 3 点は、この測定の範囲で出なかった。run 中の Swapouts は 12 本すべて 0。decode は 32 tok/s 前後で同じ。greedy の本文は 12 本すべて同一。1 本目だけ遅くなる現象も出ていない。

## 1. 数字

| | 変更前 (既定) | 変更後 (並行先読み) | 差 s (**導出**) |
| --- | ---: | ---: | ---: |
| prefill 壁時計 中央値 (レンジ)、(c) | 8.910 (8.582〜9.352) | 7.229 (7.197〜7.374) | −1.681 |
| prefill io 中央値 | 0.980 | 0.765 | −0.215 |
| layerVerify 中央値 (層ごとの合計なので壁時計の成分ではない) | 4.614 | 4.629 | — |
| decode tok/s (64 トークン、中央値) | 32.50 | 32.25 | — |
| decode hit | 83.1% (全 run) | 83.1% (全 run) | — |
| run 中の Swapouts / Swapins (`run-N.post.txt` − `run-N.pre.txt`) | 0 / 0 (全 run) | 0 / 0 (全 run) | — |
| peak_gb | 1.21〜1.22 | 1.21〜1.23 | — |
| greedy 64 トークンの本文 | 12 本すべて同一 | | |

- run 中の Swapouts を読むため、`prefill_bench.sh` は f603ade から、run の前後に `vm_stat` の全行を残す。
- decode tok/s の中央値は、各 run のフッタの `tok/s` 6 本から取った。

## 2. 分かったこと / 外れたこと

- 変更後の 7.229 s は、候補 4 つを全部入れた 4.4 s ([14](14-G5-VERIFY-AHEAD.md)、[19](19-SWA-ATTN-SGMM-GQA.md)) まで下がらない。検証を裏に回すと GPU 側が律速になり、既定のカーネルの GPU 秒がそのまま出る。GPU 側の候補が壁時計に効くのは、この先読みが入った後である。
- 並行の検証が同時に何本走るか (深さ 2 なら最大 2〜3 本) は数えていない。16 GB 機でのスワップは、この長さ (2478 トークン、decode 64) では出なかった。長い文脈やアプリの常駐と重なった場合は測っていない。
- アプリとサーバは常駐プロセスなので、検証の支払いは起動後の最初の要求 1 回だけ。この変更は、2 回目以降の prefill ([04 B0](04-GATES.md) の (b)) には効かない。

## 3. 次に渡すもの

- 既定に入れると、台帳 §1 の既定は 7.2 s 前後になる見込み (**未確認**。入れた commit で `ledger-*` を n = 6 で取って確かめる)。
- GPU 側の候補 (`mpp`、`mpp64`、`sgmm`) は、その後の既定の上で効き方を測る。
