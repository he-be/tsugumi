# 22. Q6 の前提 P6 — (decode を分けるかは 23 で判別中) decode は MBP が単体の経路で回し、M6 は prefill の分の状態をラウンドごとに受け渡す (設計 + 実測、2026-09-25)

証拠: [`bench/qfn/results/2026-09-25-Mac15-6-p6-decode/`](../../bench/qfn/results/2026-09-25-Mac15-6-p6-decode/) (MBP 単体)、
[`2026-09-25-Mac18-5-p6-decode/`](../../bench/qfn/results/2026-09-25-Mac18-5-p6-decode/) (M6 単体)。各 `run-<i>.log`、`vm_stat` の前後、`generate.out`。

```
MBP: 測定: 2026-09-25 / Mac15,6 Apple M3 Pro 18GB / macOS 15.7.5 (24G624) / Swift 6.3.3 / SDK 26.5 / Xcode 26.6 / APPLE SSD AP1024Z / commit c54b71b+dirty / minos 15.0
     iogpu.wired_limit_mb 15360
M6:  測定: 2026-09-25 / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit c54b71b / minos 15.0
     iogpu.wired_limit_mb 14336
link: 使っていない (各機が単独で回した。2 機は同じ時間帯に走ったが共有するものは無い)
```

コマンド (`command.txt`、両機同じ形、モデルの置き場だけ違う): `caffeinate -i bench/qfn/p6_decode.sh p6-decode 3 200` →

```
TsugumiKernelCheck --qwen38-generate scratch/qwen38/prompt-code.tokens --q38-new 200 --q38-mtp off --q38-out <dir>/generate.out
  --q38-gguf <model>/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf --q38-ple <model>/ple-bf16/Qwen3.8-Flash-Next-PLE-BF16.gguf   (guarded.sh の下)
```

今のコードの既定 (KV Q8_0、チャンク 2048、instruct seed 1)。prompt 8,192 トークン、decode 200 (steady は 199 ステップの中央値)。反復 n = 3。全 6 本 exit 0。

**逸脱**: MBP の札は `+dirty` (21 の docs の編集が未コミットだった。ランタイムとスクリプトは c54b71b と同じ)。M6 の run 1 は Swapouts 76 ページ (guard の線の下、run 2・3 は 0)。

**台帳** ([06](06-LEDGER.md)): 計測と設計だけで、ランタイムは変えていない。台帳 §1 は動かない。

**状態 (2026-09-25 の追記): P6 は確定。**MTP on で MBP 単体・層で分けた 2 機 (j 16・24・32)・M6 が幹の全層を同じトークン列で測り、MBP 単体が最短だった ([24](24-DECODE-SPLIT-RESULT.md): 窓の s/token 中央値 S0 0.1265、最短の分けた腕 S16 0.1421)。§2 の状態の受け渡しがそのまま設計になる。

## 0. 結論

1. **P6: decode の k は 0 (MBP が全 48 層を単体と同じ経路で回す)。prefill の k は Q6 の格子で決める (出発点 48)。**
   prefill の後、M6 は自分が回した層の状態 (その回の新しいトークンの KV 行と、GDN の状態・conv 履歴) を MBP に送る。次の prefill の前に、MBP は decode で進んだ状態を M6 に返す。
2. **M6 に全層の decode を任せる形 (k 48 のまま decode) は、今のコードで MBP 単体より遅い。**steady の中央値は MBP 192 ms (191〜203)、M6 207 ms (205〜208)。
   Q7 の合格 (2 機の decode が単体より遅くない) を、この形は今の数字では満たさない。P6 の形なら decode は単体と同じ経路なので、Q7 の decode の条件は形の上で満たす (状態の受け渡しは prefill 側の秒に入る)。
3. **受け渡しは 1 ラウンド 0.03〜0.04 s の見込み (導出)。**KV + インデクサ鍵は 12 注意層で 16.5 KB/トークン (32K で 541 MB、[qwen38/23 §0](../qwen38/23-KV-Q8-32K.md))、GDN 36 層の状態・conv 履歴・PLE 履歴は 117.6 MiB/本で文脈長に依らない ([qwen38/18](../qwen38/18-PROMPT-CACHE-CHECKPOINTS.md))。
   新規 2,650 トークン (アプリの平均、03) で KV 44 MB + 状態 123 MB = 167 MB、link 4.7 GB/s で 0.036 s。KV の行はチャンクごとに次のチャンクの裏で送れるので、壁時計に乗るのは最後の状態の 0.026 s。返す向きは状態 123 MB + decode の分の KV 行で 0.027 s。

## 1. 数字 (n = 3)

| | prefill 8,192 (trunk) s | decode steady 中央値 ms (run ごと) | 中央値 | Swapouts |
| --- | --- | --- | ---: | --- |
| MBP 単体 | 83.1 / 82.8 / 82.3 | 192 / 191 / 203 | 192 | 0 / 0 / 0 |
| M6 単体 | 111.2 / 110.6 / 110.7 | 205 / 208 / 207 | 207 | 76 / 0 / 0 |

- M6 の decode は今のコード (fp32 の decode 経路、重みは page cache)。A の M6 (常駐 13 層を自前のバッファ、staging 2 層分) の decode は測っていない。常駐 13 層は読みが無くなるが、page cache に残る余地は小さくなる。
- M6 の prefill は PLE を外付けから読む (03 §5)。この表の prefill は A の形ではなく、比較に使わない。

## 2. 設計 (P6)

### 2-1. 状態の持ち主

| 状態 | 持ち主 | M6 の扱い |
| --- | --- | --- |
| KV・インデクサ鍵 (12 注意層) | MBP (decode・prompt cache・チェックポイントと同じ) | 自分が回す層の分を写しとして持つ。prefill で書いた行を MBP に送る。decode で MBP が書いた行を次の prefill の前に受ける |
| GDN 36 層の状態・conv 履歴・PLE 履歴 | MBP | prefill の前に MBP から受け、prefill の後に MBP へ返す (117.6 MiB) |
| MTP 層の KV、head、サンプラ | MBP | 持たない。M6 は各チャンクの最後の層の残差 (84 MB) を MBP に送り、MBP が head と MTP の KV を回す |
| チェックポイント・prompt cache の場面表 | MBP だけ | 持たない。restore は MBP で行い、次の prefill の前の受け渡しで M6 の写しを揃える (KV は位置で決まるので、M6 は長さを戻すだけ) |

- これで 08 §1 の「チェックポイントは両機が同じ id で取る・戻す・捨てる」は要らなくなる。M6 は各ラウンドの頭で状態を受け、終わりに返すだけで、ラウンドをまたいで持つのは KV の写しだけ。
- M6 が落ちたら MBP は単体に戻れる (状態は全部 MBP にある)。README の前提「MBP 単体で動く」と同じ向き。

### 2-2. 1 ラウンドの流れ (k = prefill の k)

1. MBP → M6: 新しいトークン列、GDN 状態・conv 履歴・PLE 履歴 (117.6 MiB)、M6 の写しに無い KV 行 (前のラウンドの decode の分)。restore があれば M6 の KV の長さ。
2. M6: 層 0〜k−1 をチャンクごとに回す。チャンク c の層の KV 行を、c+1 を回す裏で MBP に送る。チャンクの残差を送る (今の A と同じ)。
3. MBP: 層 k〜47 (k 48 なら無し) → head → MTP の KV。M6 から最後の GDN 状態を受ける。
4. MBP: decode (単体と同じ)。M6 は待つ。

### 2-3. RAM (M6)

M6 は KV の写しを文脈の長さだけ持つ (32K で 541 MB)。21 の模擬の run の終わりの free は 0.76〜0.84 GB で、KV も本番の作業領域 (attention の T = 2048 の一時領域) も入っていない。
**Q6 の最初の run で Swapouts を見る。**出たら常駐を 12 層にする (−0.81 GB、17)、または KV の写しを持たずに毎ラウンド MBP から全部受ける (32K で 541 MB = 0.115 s、導出) のどちらかを測って秒で選ぶ。

## 3. 次の Gate に渡すもの

- **Q6 の k の格子は prefill だけの k で測る** (44・46・48、送り 0.59、`QFN_SLAB=2`)。decode は MBP で固定なので、decode の k は格子に入らない。
- Q7 の測り方は「本文 2 の decode の 1 ステップ中央値が単体と同じ幅」に加え、受け渡しの秒 (§2-2 の 1 と 3) をラウンドごとにログに出す。見込み (0.03〜0.04 s) と 1 割以上ずれたら、どちらの向きかを書く。
- 開いたまま: A の M6 (常駐 + staging) の decode は測っていない (§1)。M6 の decode が MBP より速くなる根拠が出たら、decode を M6 に任せる形を別の候補として測る。

## 4. 再開手順 (2026-09-25 の区切り、最新 commit はこの節を足したもの)

ここまでで済んだこと: Q0・Q2・Q1 (A)・Q4 腕 b (送り 0.59、20)・Q5 の M6 側 (`QFN_SLAB=2` 採用、21)・P6 の下書き (この文書)。
**次は [23](23-DECODE-SPLIT-PLAN.md) の decode の判別 (道具 `--qwen38-split-decode` を作って S0・S16・S24・S32 を n = 3)。その結果で P6 を確定してから Q6 の本体 (下の 2〜4)。**

1. **始める前の確認** (AGENTS.md と 01 の 1 サイクル):
   - 両機で `pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiKernelCheck|mlx_lm|llama'` が空、`memory_pressure -Q`、`vm_stat | grep Swapouts` を控える。
   - M6: `sysctl -n iogpu.wired_limit_mb` が 14336 (再起動で戻る。`sudo sysctl iogpu.wired_limit_mb=14336`)、`mount | grep SSD256` (PLE BF16 は外付け)。
   - link: 両機の `ifconfig bridge0` に 169.254.x.x (link-local で変わる)。`ping -c 3` で rtt 0.5 ms 前後。
   - `git push m6 HEAD` → `ssh m6 'cd dev/tsugumi && swift build -c release'`。MBP も `swift build -c release`。
2. **作るもの (04 Q6、この文書 §2)**。手本は模擬の `Sources/TsugumiKernelCheck/QFNPairSim.swift` (送り `QFN_FEED`、置き場 `QFN_SLAB`、残差の送り)。
   - M6 側のプロセス: `Qwen38Runner` (`Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift`) を層 0〜k−1 の範囲で開く。常駐 13 層は自前のバッファ + `MTLResidencySet` (17)、非常駐層は staging 2 層分に `F_NOCACHE` で読み、後ろ 0.59 を MBP から受ける。M6 の層は tensor ops (P1、15)。
   - MBP 側: 送り手 (`F_NOCACHE`、16 MB ブロック × 4 本)、層 k〜47 (k 48 なら無し)、head、MTP の KV、decode は単体のまま。
   - 状態の受け渡し (§2-2): 状態の出入り口は `forEachRecurrentRegion` (GDN・conv・PLE 履歴、:1818) と KV の行。restore は `rollback(keep:)` (:1845) の長さを M6 に伝える。
   - 候補の名前: `Q38_EXPERT_FEED=0.59`、`Q38_EXPERT_SLAB=2`、分け方 `Q38_PAIR=<M6 の bridge0 IP>` と k (名前は作るときに決める)。link が無ければ単体に戻る。
3. **最初の run で見るもの**: 正しさ (04 Q6: 単体との logits の差、top-1 の一致、入れ替わりが相手の top-5 に入る、greedy 64 トークンの本文) → M6 の Swapouts (§2-3。出たら常駐 12 層か KV の写しなしを測る) → 受け渡しの秒 (見込み 0.03〜0.04 s)。
4. **測るもの**: k 44・46・48 を本文 1・本文 2 で n = 3 (`bench/qfn/pair_bench.sh`・`ledger.sh` の 2 機の列)。台帳 06 に 2 機の列を足す。
5. **開いたままのもの**: 03 §1 の o_X (模擬に入っていない仕事) が M6 でどれだけ長いか (20 §2)、A の M6 の decode (22 §1)、3 口 (内蔵 + 外付け + TB4)、送りを要求より先に始めること (20 §3)、常駐層を散らす並べ方 (21 §2)。
   README の判断待ち 1〜3 はそのまま (3 は k 48 が通れば要らない)。
