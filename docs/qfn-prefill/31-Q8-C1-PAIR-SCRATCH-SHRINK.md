# 31. Q8 C1 の 2 機 — M6 が溢れたのは写しではなく、縮まなくなったバッチのスクラッチ。serve がラウンドの終わりで縮める (変更のみ、未測定、2026-09-26)

証拠 (新しい run は無い。[30](30-Q8-C1-CUT-IN-CHUNK.md) の 2 機の記録を読み直した):
[`2026-09-26-pair-q8-split/`](../../bench/qfn/results/2026-09-26-pair-q8-split/)・[`-pair-q8-inchunk/`](../../bench/qfn/results/2026-09-26-pair-q8-inchunk/) の
`serve.m6.log` (ラウンド後の `memory GB after the round`)、`serve.m6.guard.txt` (M6 の wired / file / compressor GB と Swapouts、2 秒ごと)、`b2-rN.rounds.jsonl`。

**この文書は変更の理由と形だけ。秒は別セッションで測る (§3)。**
**→ 測定は [32](32-Q8-C1-CUT-COPY-LEAK.md)。スクラッチは縮んだが、M6 には切れ目の写しの解放漏れ (autorelease) も残っていて、それも直した。§0 の 4 (既定に入れていない) は 32 で既定に入れた。**

## 0. 結論

1. **30 §0 の 3「切れ目 1 本あたり約 112 MB の写しで溢れた」は外れ。**inchunk のとき M6 の Metal 割当はラウンド後に split より約 2.8 GB 多く、写し 112 MB では説明がつかない (§1)。
2. **原因は `Qwen38Runner.resizeBatch`。**バッチ用スクラッチは T ≤ 32 のチャンクが来たときだけ 32 行に縮む (`shrinkBatch`、既定 on)。
   split では毎ラウンドの末尾にチェックポイントの切れ目で切れる 1〜8 トークンのチャンクがあり、M6 はそこで毎回スクラッチを返していた。
   inchunk はその末尾チャンクを消すので、M6 は maxBatch 分のスクラッチ (fixed 1.47 GB + 文脈で育つ attention / gemm など) をラウンド間も持ち続ける。
   M6 には decode が無いので、他に縮める機会が無い。遊休のページが圧縮 (compressor 1.8 GB) → スワップされ、見張りが serve を止めた。
3. **直し: `Qwen38Pair.serveRound` がラウンドの終わりで `runner.shrinkBatchAfterRound()` を呼ぶ** (`resizeBatch(T: 1)` の package 版)。
   MBP 側は decode の最初のステップ (T ≤ 32) が同じことをするので変えない。split では末尾チャンクが既に縮めているので no-op で、split の秒も RAM も変わらない。
4. **既定には入れていない。**`Q38_CUT_INCHUNK` は候補のまま。2 機の検査と交互の秒 (§3) を取ってから、04 の条件で判断する。

## 1. 記録で数えたこと

M6 の `memory GB after the round` (ラウンド後、`serve.m6.log` の r3):

| | split | inchunk |
| --- | ---: | ---: |
| metal allocated (ラウンド 1 → 7) | 10.56 → 10.84 | 12.59 → 13.68 |
| fixed per-batch | 0.02 | 1.47 |
| attnScores+selection / attn QG..VG / attnSelScores / idxDots.. / gemm | 0 | 0.07 / 0.05〜0.07 / 0.01〜0.03 / 0.01〜0.08 / 0.21 |
| dense views + expert views (常駐 10 層) | 8.09 + 8.09 (同じ) | 8.09 + 8.09 |
| GDN hist + state / KV + indexer keys | 0.11 / 0.51 (同じ) | 0.11 / 0.51 |

- fixed per-batch 0.02 GB は 32 行、1.47 GB は maxBatch (2048 行) の値。split は「ラウンド後に縮んでいる」、inchunk は「縮んでいない」。
- M6 の `serve.m6.guard.txt` (r3): wired の最大は split 13.45、inchunk 13.44 GB で同じ。違うのは compressor で、split は最大 1.08 GB、inchunk は 1.8 GB に育ち、Swapouts はラウンド 5〜6 (文脈 7.7K〜9.6K、育ったスクラッチが最大) から出る (7,896 → 26,384 ページ、16 KB/ページ)。
  r1・r2 は 50,184・76,954 ページで serve が止まった (30 §0 の 3)。
- 写し (cutStates) は M6 では `releaseCutBuffers` がチャンクごとに返しており (`Qwen38Pair.serveRound`)、`pair_begin` の `state_bytes` 114,769,920 は split でも同じ (チェックポイントの状態は split でも M6 から MBP へ送る)。C1 固有の RAM ではない。
- 単体 (MBP、18 GB) の inchunk の Swapouts 36〜76 ページ (30 §1) も同じ機構と見られるが、MBP は decode の最初のステップで縮むので、ラウンドの間だけ持つ。直しの対象にしない。

## 2. 変更

- `Qwen38Runner.shrinkBatchAfterRound()`: `resizeBatch(T: 1)`。`shrinkBatch` が off (`Q38_SHRINK_BATCH=0`) なら何もしない。
- `Qwen38Pair.serveRound`: チャンクのループの後、`memoryReport` の前に呼ぶ。ログの `memory GB after the round` は縮めた後の値になる (split と同じ読み方)。
- 単体の経路 (`Qwen38Completion`) は触らない。プロトコルは v3 のまま。

## 3. 別セッションで測ること (順に)

1. 両機のビルド: MBP `swift build -c release`、`git push m6 HEAD`、`ssh m6 'cd ~/dev/tsugumi && swift build -c release'`。M6 の `iogpu.wired_limit_mb` 14336 と SSD256 のマウントを確かめる。
2. 正しさ: `QFN_PAIR_K=46 bench/qfn/q8_resume.sh q8-resume-31`。30 §1.1 の 2 機の結果 (チャンク 2048 で L→B と負例が FAIL、split と同じ) と同じであること。
3. 秒: `for r in 1 2 3; do QFN_BODIES=2 QFN_ROUND_LOG=1 QFN_APPEND=1 bench/qfn/pair_bench.sh q8-split-31 1 --pair 46; Q38_CUT_INCHUNK=1 QFN_BODIES=2 QFN_ROUND_LOG=1 QFN_APPEND=1 bench/qfn/pair_bench.sh q8-inchunk-31 1 --pair 46; done`。
   見るもの: 3 本とも完走 (exit 0)、M6 の Swapouts (`summary-b2.tsv` の `m6_swapouts`)、`serve.m6.log` の fixed per-batch が 0.02 に戻っていること、本文 2 の 1 ターンの中央値。
   30 の 2 機は split 104.473 s、inchunk 84.178 s (n = 1)。閾値 3.06 s。
4. 単体の split / inchunk は 30 で取ってあり、この変更は単体の経路を通らないので取り直さない。
5. 通れば 04 Q8 の条件 (検査を通り、交互の壁時計が閾値以上縮む) で `Q38_CUT_INCHUNK` を既定に入れ、`ledger-*` の行 (n ≥ 6) で確かめる。
