# 26. M6 の T = 2 の幹が長い理由の判別 — 単体では M6 の I/O 以外は MBP より短い (実測、2026-09-25)

証拠: [`bench/qfn/results/2026-09-25-Mac15-6-t2-trunk/`](../../bench/qfn/results/2026-09-25-Mac15-6-t2-trunk/) と [`2026-09-25-Mac18-5-t2-trunk/`](../../bench/qfn/results/2026-09-25-Mac18-5-t2-trunk/) (各 `run-<i>.log`・`run-<i>.guard.txt`・vm_stat の前後、`generate.out`、`hostinfo.txt`、`command.txt`、`env.txt`、`driver.out`)。
計画は [25](25-M6-T2-TRUNK-PLAN.md)。集計は [`bench/qfn/t2_trunk_summary.py`](../../bench/qfn/t2_trunk_summary.py) (ステップ行の 50〜200 本目の中央値)。

```
MBP: 測定: 2026-09-25 / Mac15,6 Apple M3 Pro 18GB / macOS 15.7.5 (24G624) / Swift 6.3.3 / SDK 26.5 / Xcode 26.6 / APPLE SSD AP1024Z / commit 94f3bf2+dirty / minos 15.0
     iogpu.wired_limit_mb 15360
M6:  測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit 94f3bf2 / minos 15.0
     iogpu.wired_limit_mb 14336、SSD256 マウント済み
```

`+dirty` は docs の未コミットの変更だけ (ソースは 94f3bf2 のまま)。

コマンド (両機、同じ時間帯に並行):

```
MBP: caffeinate -i zsh bench/qfn/p6_decode.sh t2-trunk 3 400                 # env QFN_MTP=spec
M6:  caffeinate -i zsh bench/qfn/p6_decode.sh t2-trunk 3 400                 # env QFN_MTP=spec QFN_MODEL=$HOME/dev/tsugumi/scratch/Qwen3.8-Flash-Next-DS4-IQ2
各 run: TsugumiKernelCheck --qwen38-generate scratch/qwen38/prompt-code.tokens --q38-new 400 --q38-mtp spec --q38-out <dir>/generate.out
          --q38-gguf <model>/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf --q38-ple <model>/ple-bf16/Qwen3.8-Flash-Next-PLE-BF16.gguf
```

今のコードの既定 (KV Q8_0、チャンク 2048、MTP spec 連鎖 1、instruct seed 1)。prompt 8,192 トークン、decode 400 トークン。
6 本とも exit 0・guarded.sh は止めていない・Swapouts 0 ページで、25 §0-5 の条件で全部有効。EOS で止まった run は無い (両機とも 399 トークン)。

**道具の変更**: `p6_decode.sh` の MTP を `QFN_MTP` (既定 off) で選ぶようにした (25 §2 の 1、94f3bf2)。

**逸脱**
1. `p6_decode.sh` に run ごとの `GUARD_LOG` (wired / file-backed を 2 秒ごと) も足した。25 §3 の表の「wired 最大」を取るため。計画の「それ以外は変えない」からの逸脱。
2. **Pageins はステップあたりにできていない。**vm_stat は run の前後だけなので、prefill を含む run 全体の値を書く (§1)。
3. **25 §0-3 の「待ち」(pre 壁 − pre GPU) には I/O が入っている。**MBP で待ち 111 ms に対し I/O 114 ms で、差は −3.0 ms (M6 は −8.6 ms)。routed の `kernel>gpu` と `advise` が verify の pre 壁の中で待たれているため。判定には定義どおりの待ちを使い (§2)、待ち − I/O を添えた。

**台帳** ([06](06-LEDGER.md)): 計測だけで、ランタイムの既定は変えていない。

## 0. 結論

1. **単体の MTP on (T = 2) では、M6 の verify の I/O 以外は 96.8 ms で MBP の 103.4 ms より短い。**24 の割り直し (25 §0-1) にあった「T = 2 で M6 が層あたり約 2 倍」は、単体では出ない。
   pre GPU (48 対 56 ms) も routed GPU (13 対 15 ms) も M6 が短い。
2. **M6 の step が 28 ms 長い (286 対 258 ms) のは I/O の差 (148.6 対 114.0 ms、+34.6 ms)。**`kernel>gpu` が 114.5 対 77.3 ms で、SSD の差 (25 §0-1) の側。
3. **同じ M6・同じ 48 層で、単体の I/O 以外 96.8 ms に対し、24 の S48 (split-decode serve) の I/O 以外は 229 ms (25 §0-1)。**MBP 側は単体 103.4 ms と 24 の S0 102〜103 ms で一致する。差は split-decode の M6 側 (serve) にある。
4. host は M6 が少し長い: `views` 3.0 対 1.1 ms、`topk` 2.0 対 1.7 ms (和で +2.2 ms)。

## 1. 数字

ステップ行の 50〜200 本目 (151 ステップ) の中央値 (ms)。腕の値は 3 本の中央値、レンジは max − min。

| 量 | MBP run 1 / 2 / 3 | MBP 中央値 (レンジ) | M6 run 1 / 2 / 3 | M6 中央値 (レンジ) |
| --- | --- | ---: | --- | ---: |
| step | 257 / 258 / 264 | 258 (7) | 285 / 286 / 289 | 286 (4) |
| draft | 27 / 27 / 28 | 27 (1) | 28 / 29 / 28 | 28 (1) |
| verify | 222 / 220 / 224 | 222 (4) | 246 / 247 / 249 | 247 (3) |
| verify pre 壁 | 168 / 169 / 169 | 169 (1) | 189 / 191 / 193 | 191 (4) |
| verify pre GPU | 56 / 56 / 56 | 56 (0) | 48 / 48 / 48 | 48 (0) |
| verify route | 39 / 39 / 40 | 39 (1) | 39 / 39 / 39 | 39 (0) |
| verify routed (壁) | 2 / 2 / 2 | 2 (0) | 3 / 3 / 3 | 3 (0) |
| verify routed GPU | 15 / 15 / 15 | 15 (0) | 13 / 13 / 13 | 13 (0) |
| verify head | 5 / 6 / 5 | 5 (1) | 5 / 5 / 5 | 5 (0) |
| rollback | 0 | 0 | 0 | 0 |
| topk | 1.8 / 1.7 / 1.7 | 1.7 (0.1) | 2.0 / 2.0 / 2.0 | 2.0 (0.0) |
| views | 1.0 / 1.1 / 1.1 | 1.1 (0.1) | 2.8 / 3.0 / 3.0 | 3.0 (0.2) |
| advise | 35.3 / 35.1 / 36.3 | 35.3 (1.2) | 33.0 / 33.8 / 32.9 | 33.0 (0.9) |
| kernel>gpu | 76.2 / 81.8 / 77.3 | 77.3 (5.6) | 113.2 / 114.5 / 115.4 | 114.5 (2.2) |
| routed gpu (行の後半) | 15.4 / 15.4 / 15.4 | 15.4 (0.0) | 12.6 / 12.6 / 12.7 | 12.6 (0.1) |
| **I/O** (kernel>gpu + advise) | 114.0 / 112.7 / 114.8 | 114.0 (2.1) | 148.6 / 149.6 / 148.4 | 148.6 (1.2) |
| **I/O 以外** (verify − I/O) | 103.4 / 103.2 / 103.6 | **103.4** (0.4) | 96.7 / 97.2 / 96.8 | **96.8** (0.5) |
| 待ち (pre 壁 − pre GPU) | 111 / 113 / 110 | 111 (3) | 140 / 141 / 142 | 141 (2) |
| 待ち − I/O | −3.0 / 0.3 / −4.8 | −3.0 (5.1) | −8.6 / −8.6 / −6.4 | −8.6 (2.2) |
| experts | 853 | 853 | 855 | 855 |

| run 全体 | MBP | M6 |
| --- | --- | --- |
| ステップ / トークン | 203 / 399 | 201 / 399 |
| 受理 (位置 1) | 196 = 0.966 (3 本同じ) | 198 = 0.985 (3 本同じ) |
| steady tok/s | 7.77 / 7.67 / 7.70 | 6.95 / 6.88 / 6.92 |
| Pageins (prefill を含む run 全体) | 236.6 / 236.8 / 236.7 GB | 204.5 / 211.2 / 210.1 GB |
| Swapouts | 0 / 0 / 0 | 0 / 0 / 0 |
| wired 最大 (GUARD_LOG) | 14.1 / 13.8 / 14.1 GB | 12.5 / 12.0 / 12.5 GB |
| file-backed 最大 | 11.2 / 11.0 / 11.5 GB | 13.0 / 13.8 / 13.7 GB |

両機で列は同じではない (受理数が違う) が、比べるのは行の ms の中央値 (25 §0-5)。

### 1-1. 24 と並べる (25 §0-1 の値)

| | I/O 以外 | I/O |
| --- | ---: | ---: |
| MBP 単体 (この文書) | 103.4 | 114.0 |
| MBP 24 S0 (`mbp_trunk_ms` の割り直し) | 102〜103 | 111〜114 |
| M6 単体 (この文書、48 層) | 96.8 | 148.6 |
| M6 24 S48 (serve の `trunk_ms` の割り直し、48 層) | 229 | 163〜164 |

M6 の 24 S48 の幹 (389〜391 ms) と単体の verify (247 ms) の差 142〜144 ms のうち、I/O 以外が 132 ms、I/O が 14〜15 ms。
I/O の差は `advise` (単体 33 ms、S48 の serve は 62〜63 ms、この文書で割り直した値) が大きい。

## 2. 決め方 (25 §0-4) の当てはめ

閾値は両機のレンジの大きい方。

| 量 | M6 − MBP | 閾値 | 「長い」 |
| --- | ---: | ---: | --- |
| verify pre GPU | −8 | 0 | いいえ (M6 が短い) |
| verify routed GPU | −2 | 0 | いいえ (M6 が短い) |
| 待ち (pre 壁 − pre GPU) | +30 | 3 | はい |
| topk + views | +2.2 | 0.2 | はい |
| I/O 以外 | −6.6 | 0.5 | いいえ (M6 が短い) |

- **(a) GPU は当たらない。**
- **(b) は条件の字面では当たる**が、待ちの +30 ms は I/O の差 (+34.6 ms) で、待ち − I/O は M6 の方が短い (逸脱 3)。host の `topk` + `views` は +2.2 ms。
- **(c) の条件「M6 の I/O 以外が長いに当たらない」は当たる。**
- 25 の表は (b) と (c) が同時に当たる場合を決めていない。**0-2 の問い (24 の 2 倍の理由) への答えは 4 (split-decode の道具の側)**: 単体の M6 の I/O 以外は 96.8 ms で、24 の S48 の 229 ms は単体では出ない (§1-1)。host の +2.2 ms は M6 と MBP の差として実在するが、その 132 ms を説明する大きさではない。
- (c) の添え書き「単体の M6 の step が MBP の step より短いか等しければ」は当たらない (286 対 258 ms)。step の差は I/O の差 (SSD) で、I/O 以外は M6 が短い。

## 3. 次 (25 §4、2026-09-26 に決めた)

- P6 (22) は動かさない。Q6 の k の格子 (44・46・48) も変えない。
- **split-decode の serve 側は直さない。**25 §4 の (c) の行き先は「serve 側を直して S24 と S48 を測り直す」だが、(c) の添え書き (M6 の step が MBP 以下) は当たっておらず、P6 が確定した後は split-decode の道具を Q6 でも Q7 でも使わない。
  直しても台帳の秒は動かず、直す価値の上限は I/O の差の約 30 ms/ステップ (25 §4 (a) 行) で、単体の M6 の step 286 ms が MBP の 258 ms より長い (SSD) 事実は変わらない。
- serve 側で何が 132 ms を足しているか (受信、`snapshotRows = 1`、rollback の準備、`advise` の 2 倍) は分けていない。このうち `advise` の 2 倍 (単体 33 ms、S48 の serve 62〜63 ms) だけは、Q6 の M6 のプロセスが同じ `Qwen38Runner` の `layerRange` を使うので、**Q6 の最初の run で層ごとの `advise` を単体の値と並べて見る** (22 §4 の 3)。別の実験は立てない。
- **次は Q6 の本体** (22 §4 の 2〜4)。
