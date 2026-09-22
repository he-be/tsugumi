# 05. 罠

## 1. 16 GB のページキャッシュ

- `sudo purge` が使えず冷やせない。rsync 直後の 1 本目に io 3.33 s (他は 0.77〜1.08) が出た例がある (旧 §7-4a、証拠なし)。
- **対策**: `prefill_bench.sh` が run ごとに `vm_stat` / `memory_pressure -Q` を残す。外れ値は捨てずに `pre.txt` で説明する。中央値で語る。
- `F_NOCACHE` は新規のキャッシュを止めるだけで、常駐済みのページは使われる ([M6_SSD_BANDWIDTH §1](../investigations/M6_SSD_BANDWIDTH.md))。

## 2. 単体ベンチと本番で符号が反転する

mflux で `mlp.down` の K 分割が単体 −30 ms / 本番 +42 ms だった例がある (別プロジェクト、数字は引かない)。
**G3 / G4 の判定は必ず `prefill_bench.sh` の本番ループで取る。**G2 の単体プローブは「線を開くか閉じるか」だけに使う。

## 3. 主スレッドが I/O で離れると GPU が飢える

`executeExpertCachePlan` が `concurrentPerform` でミスを読む間、コマンドバッファを積む人がいない。
GPU カーネルを速くするほど、この露出が相対的に効く。**M6 は SSD が M3 Pro の半分なので離脱時間が長い。**G5 の的。

## 4. 熱とクロック

連続実行で GPU クロックが落ちる例が M3 Pro にある ([qwen35moe/24](../qwen35moe/24-PREFILL-MOE-PATH.md))。
A/B は **A B A B と交互**に取る。6 本連続で単調に遅くなっていたら熱を疑い、間を空けて取り直す。mini の筐体は MBP と違う。

## 5. M6 に Xcode が無い

- Metal System Trace / GPU フレームキャプチャは M6 で取れない。MBP には tensor 経路が存在しない。
  **つまり tensor カーネルの GPU トレースはどちらの機械でも取れない。**
- `MTLCaptureManager` でプログラムから `.gputrace` を書き、MBP の Xcode 26.6 で開く手は**未確認** (macOS 27 のトレースを 26.6 が読めるか)。
- 代替は throughput プローブ (G2) と、カウンタ (`MTLCounterSampleBuffer`) の直読み。**「Neural Accelerator が使われた」はキャプチャでは示せない前提で計画する。**

## 6. MBP では tensor 経路の正しさを一切検証できない

MSL 4.0 はコンパイラに拒否される ([証拠](../../bench/m6/results/2026-09-22-Mac15-6-gpu_family_probe/output.txt))。
`swift test` が緑でも tensor カーネルは走っていない。M6 で `TsugumiKernelCheck` を回す ([04 G1](04-GATES.md))。

## 7. family でゲートしない

`supportsFamily(.apple10)` は M6 で true だが、それは今日の結果であって次の機械の保証ではない。
ゲートは**PSO 生成の成否**に置く (G1)。static な family 判定は残さない。

## 8. ベンチのモデルは group 32

`gemma4-qat-sym.gturbo` の attention は `groupSize 32 / sym`。`MPPPrefillInt4QMM` は `affineGroupSize == 64` のガードで nil になる。
機械が正しくても経路は閉じる。G1 で 32 対応を入れるまで、MPP は 1 度も走らない。

## 9. 機械を混ぜない

`Mac15-6` と `Mac18-5` の数字を 1 つの表に並べるのはよい。**比を 1 つの数にして「M6 で N 倍」と書くのは同一 commit・同一コマンドの対だけ。**
機械をまたぐ比は世代差と実装差が混ざる。分けたいなら両機で同じ A/B を取る。

## 10. 名前の罠

Metal キャプチャに出る `_nax_` 系のカーネル名は GPU 内の tensor unit のもので **ANE ではない**。
「Neural Accelerator」(GPU 内) と「Neural Engine」(ANE) を混同した記述を書かない。

## 11. 既定を動かさない

`allowedExpertCacheSlots` / チャンク幅 / 検証モードの既定は変えない。ベンチは 32 スロット・chunk 2048・`full-sha256` に固定
(`prefill_bench.sh` が焼いてある)。変えるときは別ラベルで取る。
