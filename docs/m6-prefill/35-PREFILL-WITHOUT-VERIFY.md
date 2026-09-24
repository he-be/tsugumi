# 35. prefill の計測から層ファイルの検証を外す — `trusted-install` で 2.897 s。全層を先に開く形はページキャッシュを追い出すので使わない (実測、2026-09-24)

**決まり (ユーザー、2026-09-24)**: prefill の秒に、層ファイルの SHA-256 検証を含めない。検証を含む計測は使わない。
[06](06-LEDGER.md) の行はすべて検証を含む量 (新しいプロセスの 1 回目の prefill。層ファイルは初めて触るときに開いてハッシュする) だったので、進捗の物差しから取り下げた。
34 の 3.142 → 2.885 s と 20 の 8.682 → 7.184 s は、検証の待ちを prefill の区間から減らした分である。

証拠 (M6): [`ledger-no-verify`](../../bench/m6/results/2026-09-24-Mac18-5-ledger-no-verify/) (n = 6)、[`nv-host`](../../bench/m6/results/2026-09-24-Mac18-5-nv-host/) (`TF_PREFILL_HOST_PROFILE=1`、n = 3)。比べた相手は [34](34-VERIFY-AHEAD-DEPTH.md) の `f4-default-a`。

```
測定: 2026-09-24 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit b0b4de2 / minos 15.0
```

コマンドは `prefill_bench.sh` の固定形に `--open-layers-before-prompt` を足したもの (b0b4de2)。load の後、prompt の前に全層を開いて SHA-256 を済ませ、その秒はフッタの `openLayers=` に出る。全 run exit 0、Swapouts 0。

## 0. 結論

1. **全層を先に開くと、prefill は 3.818 s (3.784〜3.921、n = 6)。**検証を prefill の中でしていた 34 の既定 (3.159 s) より長い。
2. **理由は SSD からの読み直し。**M6 の 16 GB では run の前のファイルのページキャッシュが約 5 GB で、エキスパート 12.9 GB は収まらない。どの run も SSD から読んでいる。層ごとの検証は、その層を prefill の直前に読み込む先読みの役もしていた。全層を先にハッシュすると前の層が追い出され、prefill が読み直す。
3. **この形は prefill の計測として使わない。**検証は外れたが、ページキャッシュの追い出しという別の歪みが入る。

## 1. 数字

| | 34 の既定 (`f4-default-a`、検証は prefill の中) | 全層を先に開く (`ledger-no-verify`) |
| --- | --- | --- |
| prefill s (run 1〜3) | 4.436、3.161、3.129 | 3.921、3.858、3.784 |
| run 中の Pageins (MiB、導出) | 12883、6441、6334 | 10226、9150、8831 |
| run の前のファイルのページ (MiB) | 11968、5085、5584 | 11930、4850、5360 |

`nv-host` の 3 本目: `fetch` 1.292 s、GPU の空き 0.831 s のうち `fetch` の間 0.727 s。residency の同期が 1.255 s (34 では 0.18 s 前後)。

## 2. `trusted-install` で測った (ef29110)

M6 の受領書を `TsugumiRepack --verify-install --input-moepack scratch/gemma4-qat-sym.gturbo` で M6 の置き場所に作り直した (ユーザーの了承の上。39 ファイル 15,681,261,432 バイトが一致、manifest のハッシュは前と同じ。元の受領書は M6 の `~/verified-install.json.mbp-path.bak`)。
`prefill_bench.sh` は `--verification trusted-install` で測る。`--open-layers-before-prompt` は CLI から消した。

証拠: [`ledger-trusted`](../../bench/m6/results/2026-09-24-Mac18-5-ledger-trusted/) (n = 6)、[`trusted-host`](../../bench/m6/results/2026-09-24-Mac18-5-trusted-host/) (`TF_PREFILL_HOST_PROFILE=1`、n = 3)。commit ef29110。全 run exit 0、Swapouts 0。

| | 値 |
| --- | --- |
| prefill s (run 順) | 3.123、2.993、2.884、2.889、2.905、2.880。**中央値 2.897** (台帳 §1 の最初の行) |
| run 中の Pageins (MiB、導出) | 3633、3659、3095、2961、3031、3129 |
| GPU busy / idle (`trusted-host` の 2・3 本目) | 2.451 / 0.335、2.457 / 0.328 s |
| GPU の空きのうち `fetch` の間 | 0.270、0.272 s |
| residency の同期 (`[expert mmap]`) | 0.726、0.712 s |

- 検証が無いので、エキスパートのページは prefill の中で SSD から引かれる。その分が residency の同期 (0.18 → 0.71 s) と、fetch の間の GPU の空きに出ている。
- 到達目標 mlx_lm 1.823 s (検証なし、暖まったプロセス) と並べる量はこれ。差は 1.07 s (導出)。GPU busy 2.45 s が MLX の全体 1.823 s より 0.63 s 長い。

## 3. 次に渡すもの

- 残る差は GPU busy (2.45 s) と、SSD からの読みで GPU が空く分 (約 0.3 s)。
