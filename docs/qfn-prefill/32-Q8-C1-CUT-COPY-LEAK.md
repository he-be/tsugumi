# 32. Q8 C1 — 31 §3 の測定、C1 の既定化、M6 で切れ目の写しが解放されなかったバグの修正 (実測、2026-09-26)

証拠:
- 31 §3 の手順: 正しさ [`2026-09-26-pair-q8-resume-31/`](../../bench/qfn/results/2026-09-26-pair-q8-resume-31/)、交互の秒 [`-pair-q8-split-31/`](../../bench/qfn/results/2026-09-26-pair-q8-split-31/)・[`-pair-q8-inchunk-31/`](../../bench/qfn/results/2026-09-26-pair-q8-inchunk-31/)。
- 既定化の後の台帳 (バグあり、commit 8e7b058): [`2026-09-26-Mac15-6-ledger-c1/`](../../bench/qfn/results/2026-09-26-Mac15-6-ledger-c1/)・[`-pair-ledger-c1/`](../../bench/qfn/results/2026-09-26-pair-ledger-c1/)。
- 修正の後の台帳 (commit 9b983d0): [`2026-09-26-Mac15-6-ledger-c1b/`](../../bench/qfn/results/2026-09-26-Mac15-6-ledger-c1b/)・[`-pair-ledger-c1b/`](../../bench/qfn/results/2026-09-26-pair-ledger-c1b/)。

```
MBP: 測定: 2026-09-26 / Mac15,6 Apple M3 Pro 18GB / macOS 15.7.5 (24G624) / Swift 6.3.3 / SDK 26.5 / Xcode 26.6 / APPLE SSD AP1024Z / commit 3c8026f (31 §3), 8e7b058 (ledger-c1), 9b983d0 (ledger-c1b) / minos 15.0
M6:  測定: 2026-09-26 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / 同じ commit / minos 15.0
iogpu.wired_limit_mb 14336 (M6)、SSD256 マウント済み
link: bridge0 169.254.218.152 → 169.254.13.135 / 分け方 A k = 46
```

M6 は CLT だけなので `swift build -c release` 全体は Mac アプリ (SwiftUI のマクロ) で落ちる。2 機で使う `TsugumiKernelCheck` と `TsugumiServer` を `--product` で組んだ。

**台帳** ([06](06-LEDGER.md)): `ledger-c1b` (単体・2 機、n = 6) を既定の行に足す。`ledger-c1` は修正前で、2 機の本文 2 が 6 本中 5 本失敗した記録として残す。

## 0. 結論

1. **31 §3 の手順 1〜3 は通った。**2 機の本文 2 の 1 ターンは split 104.365 s (104.198〜105.202) → inchunk 82.492 s (82.474〜83.230)、交互 n = 3、6 本とも exit 0。差 21.873 s (導出) は閾値 3.06 s を超える。
   M6 のラウンド後の fixed per-batch は inchunk でも全ラウンド 0.02 GB (31 の直しは効いた)。04 の規則で `Q38_CUT_INCHUNK` を既定にした (8e7b058、`=0` で切る)。
2. **既定化の後の台帳で 2 機が落ちた。**M6 のラウンド後の metal allocated が本文 2 のラウンドごとに 0.11〜0.23 GB 増え、serve をまたいで持ち越され (10.64 → 12.13 GB)、2 本目の 5 ラウンド目で見張りが serve を止めた (Swapouts +32,988 ページ / 60 s)。
   31 §3 の交互の run は 1 回ごとに M6 の serve を起動し直していたので、この増え方が表に出なかった。
3. **原因は切れ目の写しの解放漏れ。**`Qwen38Runner.forEachCutRegion` で写しを読むと、写しのバッファが autorelease プールに入る。M6 の pair serve のループはプールを空にしないので、`releaseCutBuffers` の後もラウンドごとに写し 1 本分 (109.5 MiB) が残った。
   **直し: `forEachCutRegion` の中身を `autoreleasepool` で包む** (9b983d0)。写しを読む経路 (M6 の送り、MBP の受け取り、MBP のチェックポイント) はすべてここを通る。
4. **修正の後、2 機の台帳は 6 本とも完走した。**M6 のラウンド後の metal allocated は 72 ラウンドすべて 10.64 GB、M6 の Swapouts 0。本文 2 の 1 ターンは 82.587 s (81.967〜86.046)。
   単体は 141.854 s (141.718〜142.091)、ledger-q2 の 156.233 s から 14.379 s 短い (導出)。本文 1 は単体 28.416〜30.530 s、2 機 14.821〜15.323 s。

## 1. 再現と切り分け (`TsugumiKernelCheck --qwen38-cut-leak`)

M6 の serve と同じ形 (1 つの runner、層 0〜45、`trunkOnly`) で、tool.tokens (468 トークン) を切れ目 1 本 (234) で forward → `forEachCutRegion` で読む → `releaseCutBuffers` → `shrinkBatchAfterRound` を 8 回。
値は metal allocated から重みのビュー (MBP ではエキスパートを初めて使ったときに作られる) を引いたもの。MBP で取った (M6 には KernelCheck の既定の置き場にモデルが無い)。

| | ラウンド 1 → 8 (GB) | 増分 |
| --- | --- | --- |
| 修正前 | 0.29 → 1.10 (毎ラウンド +0.11〜0.12) | 0.81 GB、FAIL |
| 修正前、写しを読まない | 0.18 → 0.18 | 0 |
| 修正前、読みを呼び出し側で `autoreleasepool` に入れる | 0.18 → 0.18 | 0 |
| 修正後 | 0.18 → 0.18 | 0 |

- 検査は最後のラウンドの値が最初より写し 1 本分を超えて増えたら exit 1。修正前は落ち、修正後は通る。回帰検査として残す。
- forward のコマンドバッファは既に `autoreleasepool` の中 (`Qwen38Runner.forward`) で、増分はそこではなかった。

## 2. 数字 (中央値 (最小〜最大)、秒)

| | 本文 2 の 1 ターン | 失敗 | M6 の Swapouts (ページ) |
| --- | ---: | ---: | --- |
| 31 §3 交互 split (3c8026f、n = 3) | 104.365 (104.198〜105.202) | 0 | 0・0・0 |
| 31 §3 交互 inchunk (3c8026f、n = 3) | 82.492 (82.474〜83.230) | 0 | 21,432・17,644・15,856 |
| ledger-c1 2 機 (8e7b058、n = 6) | 82.689 (完走 1 本) | 5 | 9,680・32,988 (2 本目で serve 停止) |
| **ledger-c1b 2 機 (9b983d0、n = 6)** | **82.587 (81.967〜86.046)** | 0 | 0 × 6 |
| ledger-c1 単体 (8e7b058、n = 6) | 143.402 (143.079〜143.608) | 0 | - |
| **ledger-c1b 単体 (9b983d0、n = 6)** | **141.854 (141.718〜142.091)** | 0 | - |

- 31 §3 の inchunk の Swapouts は、1 run の中の 7 ラウンド分の漏れ (約 0.8 GB) と見られる (修正後の交互の run は取っていない)。
- 2 機の本文 1 (5 本) は ledger-c1b で 14.821〜15.323 s、失敗 0。
- 単体の本文 2 は ledger-c1 (8e7b058) と ledger-c1b (9b983d0) で範囲が重ならない (差 1.548 s、導出)。閾値 3.06 s の内側で、理由は調べていない。

## 3. 正しさ

- `QFN_PAIR_K=46 bench/qfn/q8_resume.sh q8-resume-31` (3c8026f): チャンク 2048 の 4 本 (split / inchunk × MTP off / spec) は 30 §1.1 と同じ (L→B が 40 トークン目から違い、負例が FAIL、他は一致)。
  2 機でチャンク 12 も流れ、両腕とも負例と「call == whole」が FAIL (30 の 2 機ではチャンク 12 を流していないので比べられない)。
- 修正 (9b983d0) は写しを読む間のプールだけで、読み書きする値は変わらない。修正後に `--qwen38-resume` は流していない。
- 前例 14 の「greedy 64 の一致」は、サーバが公式値でサンプリングする (temperature 0.7 / top_k 20 / top_p 0.8) ので greedy ではない。変更前どうしでも本文が 1 文字目から違い、比較にならない。
  取ったものは `-pair-q8-inchunk-31/sampled64/` に置いた。

## 4. 変更

- `Qwen38Completion.cutInChunk`: 既定 on、`Q38_CUT_INCHUNK=0` で切る (8e7b058)。
- `bench/qfn/q8_resume.sh`: split 腕は `Q38_CUT_INCHUNK=0` を明示 (8e7b058)。
- `Qwen38Runner.forEachCutRegion`: 中身を `autoreleasepool` で包む (9b983d0)。
- `TsugumiKernelCheck --qwen38-cut-leak <tokens> [--q38-rounds N] [--q38-layers k]` (`Qwen38CutLeakCheck.swift`、9b983d0)。
