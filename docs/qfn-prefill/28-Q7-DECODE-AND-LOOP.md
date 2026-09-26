# 28. Q7 decode と会話の機能 — チェックは 2 機で通る (progress を除く)。decode の 1 ステップは 2 機の 4 セットとも単体より長く、1 ステップの SSD の読みも多い (実測、2026-09-26)

証拠: 単体 [`bench/qfn/results/2026-09-26-Mac15-6-q7-mtp-on/`](../../bench/qfn/results/2026-09-26-Mac15-6-q7-mtp-on/)・[`-q7-mtp-off/`](../../bench/qfn/results/2026-09-26-Mac15-6-q7-mtp-off/)、
2 機 [`2026-09-26-pair-q7-mtp-on/`](../../bench/qfn/results/2026-09-26-pair-q7-mtp-on/)・[`2026-09-26-pair-q7-mtp-off/`](../../bench/qfn/results/2026-09-26-pair-q7-mtp-off/)。
各 `summary-b3.tsv`、`b3-rN.log` (ToolLoopCheck)、`b3-rN.rounds.jsonl` (`Q38_ROUND_LOG`)、`b3-rN/` (rounds.jsonl・turns.jsonl)。

```
MBP: 測定: 2026-09-26 / Mac15,6 Apple M3 Pro 18GB / macOS 15.7.5 (24G624) / Swift 6.3.3 / SDK 26.5 / Xcode 26.6 / APPLE SSD AP1024Z / commit 63b8fbb / minos 15.0
M6:  測定: 2026-09-26 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 63b8fbb2 / minos 15.0
iogpu.wired_limit_mb 14336 (M6)
link: bridge0 169.254.218.152 → 169.254.13.135 / rtt 0.418〜0.745 ms (link.txt) / 分け方 A k = 46
```

コマンド (`command.txt` の本文 3): `QFN_BODIES=3 QFN_MTP=<on|off> bench/qfn/pair_bench.sh q7-mtp-<on|off> 1 --solo` と `… --pair 46` →

```
Q38_ROUND_LOG=<b3-rN.rounds.jsonl> TsugumiToolLoopCheck --model ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2
  --conversations docs/experiments/knowledge-sources/questions-v2-stage2.json --network offline --max-rounds 6
  --thinking off --context 32768 --repeats 1 --mtp <on|off> --out <b3-rN>          (guarded.sh と caffeinate -i の下)
```

2 機は MBP の各プロセスに `Q38_PAIR=169.254.13.135 Q38_PAIR_K=46`、M6 は `pair_serve.sh` の既定 (P1・常駐 10 層・置き場 2・送り 0.59)。
順は 単体 on → 2 機 on → 単体 off → 2 機 off を 2 巡 (間 20 s、同じバイナリ)。反復 n = 2 (1 セット = Offline 5 問 × 1 ターン)。**反復 3 未満なので解釈は書かない。**

**逸脱**: (a) M6 のバイナリは faa3b11 で組んだ。63b8fbb との差は `bench/` だけ。(b) M6 は CLT しか無く、`TsugumiKernelCheck` だけを組んだ (アプリは組めない)。
(c) 本文 2 (再生) は 1 ラウンド 1 トークンしか生成しないので、04 Q7 の「本文 2 の decode」は本文 3 (実際に生成する) に置き換えた。答えが単体と違うので、ステップ数とラウンド数は腕ごとに違う。
(d) `Qwen38PromptCacheTests` はエンジンに触れない (場面表のロジックだけ) ので 2 機では流していない。2 機での prompt cache とチェックポイントの検査は ToolLoopCheck の live・restore が受け持つ。

**台帳** ([06](06-LEDGER.md)): 道具 (round log の `step_s`・`pair_begin`、ToolLoopCheck `--mtp`) を足しただけで、既定は変えていない。台帳 §1 は動かない。

## 0. 結論

1. **会話の機能は 2 機で通った。**progress 以外のチェックが通ったターンは 20 中 19。残り 1 は live で、描き直しの改行の食い違い (§2 の 2、単体でも出る既知の現象) が原因。このとき prompt cache はチェックポイントから restore し、2 機のまま最後まで回った。
   progress は 2 機の全ターン (20) で落ちる。M6 の層の間は進捗の報告が無いため。**prefill のカウンタは表示だけなので放置する** (ユーザー、2026-09-26)。
2. **MTP の受理率は単体と同じ幅。**単体 825/995・968/1153、2 機 836/998・1059/1280 (0.83〜0.84、導出)。qwen38/10 の 0.77〜0.97 に入る。
3. **decode の 1 ステップの中央値は、2 機の 4 セットがすべて単体の 4 セットより長い。**MTP on: 単体 221.3・230.4 ms、2 機 232.2・234.3 ms。MTP off: 単体 177.3・179.2 ms、2 機 186.5・189.0 ms。
   04 Q7 の合格 (単体より遅くない) を満たしていない。同じ run で decode の 1 ステップの SSD の読みも 2 機の方が多い (§1)。

## 1. 数字 (セットごと、r1 / r2)

| | 単体 on | 2 機 on | 単体 off | 2 機 off |
| --- | --- | --- | --- | --- |
| progress 以外が通ったターン / 5 | 5 / 5 | 5 / 4 | 5 / 5 | 5 / 5 |
| decode ステップ中央値 ms | 221.3 / 230.4 | 232.2 / 234.3 | 177.3 / 179.2 | 186.5 / 189.0 |
| decode ステップ p90 ms | 295.3 / 296.5 | 323.8 / 313.0 | 238.5 / 238.8 | 248.3 / 252.4 |
| ステップ数 / 生成トークン | 995 / 1840、1153 / 2144 | 998 / 1857、1280 / 2364 | 2156 / 2179、2000 / 2020 | 2106 / 2130、2026 / 2047 |
| decode の読み MiB/ステップ (導出) | 350.3 / 371.1 | 452.5 / 454.4 | 179.6 / 185.1 | 239.5 / 244.1 |
| decode 合計 s | 231.610 / 275.157 | 289.221 / 364.647 | 394.480 / 366.204 | 443.469 / 430.976 |
| prefill 合計 s | 381.607 / 452.408 | 300.738 / 350.314 | 413.483 / 350.599 | 306.124 / 277.879 |
| 5 ターンの壁時計の和 s (ログの秒) | 616 / 730 | 592 / 718 | 811 / 721 | 752 / 711 |
| 受け渡し MBP→M6: 回数、合計 s、最大 s | – | 23・0.932・0.043、25・0.998・0.042 | – | 24・0.986・0.054、21・0.850・0.045 |
| MBP Pageins MB | 1,048,089 / 1,236,539 | 1,356,741 / 1,591,101 | 1,145,375 / 1,030,191 | 1,375,818 / 1,277,411 |
| MBP Swapouts (ページ) | 11,224 / 272 | 15,560 / 32,012 | 36 / 176 | 356 / 1,144 |
| M6 内蔵 / 外付け MB、Swapouts | – | 590,826 / 3,600・0、659,084 / 5,533・0 | – | 613,531 / 3,782・0、545,401 / 3,784・0 |

- decode の読みは、`rounds.jsonl` の各 `run` 行の `disk_read` から直前の行 (最後のチャンク) の値を引いた和 ÷ ステップ数。MTP の head・checkpoint の書き込みの読みを含む。
- 受け渡しの 1 回は 0.040〜0.041 s (合計 ÷ 回数、導出)、状態 114.8 MB。22 §0-3 の見込み 0.03〜0.04 s の上端。最大 0.054 s は見込みを 1 割以上超える (長い向き)。
- 壁時計の和はターンの長さ (答えが違うので生成量もラウンド数も違う) を含み、腕の比較には使わない。

## 2. 分かったこと / 外れたこと

1. **decode の経路は単体と同じ (P6) だが、1 ステップが長い。**同じセットで decode の 1 ステップの SSD の読みが、MTP on で約 +100 MiB、off で約 +60 MiB 多い。
2. **live の失敗 1 件 (2 機 on r2、N2-easternmost 3 ラウンド目)**: モデルは `</think>\n\n` の直後に `<tool_call>` を書き、次の要求の描き直しは改行を 1 つ多く置いた (トークン 271 → 1358)。
   session は `diverged at=2852` で生きている状態を使わず、チェックポイント 2852 から restore した (`restore_ms=34`)。qwen38/21・26 と同じ種類で、2 機に固有ではない。**2 機で restore から prefill し直した経路が通った初めての記録。**
3. **progress は 2 機で必ず落ちる。**M6 の層を待つ間 (チャンクあたり約 7 s) は報告が無い。直さない (ユーザー)。
4. MBP の Swapouts は 2 機の方が多い (最大 32,012 ページ)。有効条件ではない (24、ユーザーの判断)。M6 は全セットで 0。

## 3. 次の Gate に渡すもの

- 04 Q7 の decode の条件は満たしていない。ステップの中央値の差は MTP on で 1.8〜13.0 ms、off で 7.3〜11.7 ms (2 機と単体の全組、導出)。n = 2 なので、原因の判別の前に反復を増やすかは判断待ち。
- 開いたまま: decode の読みの差の原因 (page cache の中身)、受け渡しの最大 0.054 s。
