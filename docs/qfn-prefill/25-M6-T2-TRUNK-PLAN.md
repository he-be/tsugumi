# 25. M6 の T = 2 の幹が長い理由を判別する実験 — 単体・MTP on で両機の検証 1 行の内訳を取る (計画、2026-09-25)

**状態: 測った ([26](26-M6-T2-TRUNK-RESULT.md))。単体の M6 の I/O 以外は MBP より短く、24 の 2 倍は split-decode の道具の側。**以下は計画のまま。§0 の数字は [24](24-DECODE-SPLIT-RESULT.md) と [22](22-Q6-P6-DECODE-PLACEMENT.md) の証拠ディレクトリのログを割り直したもので、出所はその 2 つ。
24 は「分けた腕はどの j でも遅い」で P6 (decode は MBP 単体) を確定し、§3 で「T の違いとページの読みの違いのどちらが効いたかは分けていない」と残した。
この文書は、その残りを**単体の道具で**判別する手順を固定する。P6 は動かさない。

言葉: **I/O** = routed の `kernel>gpu` (ページインを含む) と `advise` の和。**I/O 以外** = 幹の ms から I/O を引いた残り (GPU の計算、host の top-10 と views、隠れた待ち)。**検証** = MTP spec の T = 2 の forward。

## 0. 問いと決め方

### 0-1. 記録が言っていること (24・22 のログの割り直し)

24 の各 run のステップ 50〜200 の中央値 (ms)。M6 は `.m6.log` の `trunk_ms − kernel_gpu_ms − advise_ms`、MBP は `.log` の `mbp_trunk_ms − mbp_kernel_gpu_ms − mbp_advise_ms`。

| 腕 | I/O の和 (両機) | I/O 以外 MBP | I/O 以外 M6 | M6 の I/O 以外 ÷ 層 | MBP の I/O 以外 ÷ 層 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | 111〜114 | 102〜103 | — | — | 2.1 |
| S16 | 88〜92 | 72 | 81〜84 | 5.1〜5.3 | 2.3 |
| S24 | 82〜84 | 57 | 109〜112 | 4.5〜4.7 | 2.4 |
| S32 | 91〜94 | 41 | 142〜146 | 4.4〜4.6 | 2.6 |
| S48 | 163〜164 | 7〜12 (head だけ) | 229 | 4.8 | — |

- **I/O は分けると減る** (S0 111〜114 → S24 82〜84)。減った分を M6 の I/O 以外が食っている。
- **S48 の M6 は、I/O を 0 にしても 229 ms で S0 の幹 (212〜219) より長い。**送り (Q4 の腕 b の逆向き) や RAM の置き場は、この数字が MBP 並みになるまで乗らない。
- 22 (MTP off、T = 1、`--qwen38-generate`) を同じ割り方にすると、MBP 192 − 40 − 19 = 133、M6 207 − 52 − 22 = 133 で**同じ**。pre GPU は MBP 80 対 M6 72〜75、routed GPU は 18 対 17〜18。
- 22 と 24 は道具が違う (幹に何を含めるかが違う) ので、133 と 102 を跨いで比べない。**同じ道具の中で、T = 1 は等しく、T = 2 は M6 が層あたり約 2 倍**、が記録の言うこと。
- I/O 側は説明がつく。ページインは MBP S0 520 MB/ステップ、M6 S48 352 MB/ステップで、`kernel>gpu` ÷ バイトはそれぞれの内蔵 SSD (6.68 / 3.34 GB/s、[03 §4](03-FACTS.md)) に一致する。

### 0-2. 問い

**M6 の検証 (T = 2) の I/O 以外が MBP の層あたり 2 倍なのは、次のどれか。**

1. GPU の計算 (pre-router の群か、routed の群か)
2. host (top-10 の Swift ループ、views の対応付け。22 の T = 1 で M6 の `views` は 2.4 ms 対 MBP 0.6 ms)
3. 隠れたページフォルト (I/O に数えられない読み。dense の重み約 5 GB/パスが page cache から押し出される等)
4. split-decode の道具の側 (受信、`snapshotRows = 1`、rollback の準備。単体では出ない)

### 0-3. 物差し

両機で**単体**の `--qwen38-generate --q38-mtp spec` を回す。各ステップの行に既に出るもの ([qwen38/11 §1](../qwen38/11-VERIFY-PASS-SPLIT.md)、`Qwen38MTPCheck.swift` の `speculativeLoop`):

```
step = draft (T=1 pre route routed head) + verify (pre [gpu] route routed [gpu] head) + rollback
  | route topk views advise (miss) calls | routed commit>kernel kernel>gpu (behind shared, shared gpu) gpu after | experts N ...
```

run ごとに、ステップ 50〜200 の中央値を取る量: verify の pre 壁・pre GPU・route・routed GPU・head、`topk`・`views`・`advise`・`kernel>gpu`・routed `gpu`、`experts` (和集合の本数)、draft の合計、step。
派生: **I/O** = `kernel>gpu` + `advise`、**I/O 以外** = verify − I/O、**待ち** = pre 壁 − pre GPU。腕の値は n = 3 の中央値、レンジは max − min。

### 0-4. 決め方 (測る前に置く)

閾値は 23 と同じ形: **M6 の中央値が MBP の中央値より、両機のレンジの大きい方を超えて長ければ「長い」。**量ごとに判定する。

| 判定 | 条件 | 読み |
| --- | --- | --- |
| (a) GPU | verify の pre GPU か routed GPU が「長い」 | 0-2 の 1。次に区間 (§2 の 2) で群を特定する |
| (b) host か隠れたフォルト | GPU は「長い」に当たらず、待ち (pre 壁 − pre GPU) か `topk` + `views` が「長い」 | 0-2 の 2 か 3。`views`・`topk` が長ければ host、待ちだけ長ければフォルト (Pageins と GUARD_LOG の file-backed を添える) |
| (c) 道具 | M6 の I/O 以外が MBP と同じ幅 (「長い」に当たらない) | 0-2 の 4。24 の S48 の M6 の幹 (389〜391 ms) と単体の verify を並べて、差を道具の側に書く |
| (d) 混在 | (a) と (b) の両方に当たる | 両方を書く。次の手は大きい方から |

- I/O (`kernel>gpu` + `advise`) が M6 で長いのは 0-1 のとおり SSD の差で説明がつくので、この実験の判定には使わない (数字は書く)。
- (c) のとき: 単体の M6 の step が MBP の step より短いか等しければ、24 の結論 (どの腕も遅い) の原因は道具にあり、道具を直して S24・S48 を測り直す (§4)。

### 0-5. 前提 (P 表、測る前に固定。動かしたら結果文書に書く)

| 項目 | 固定する値 |
| --- | --- |
| M6 の経路 | 今のコード (fp32、重みは page cache、常駐なし)。22・24 と同じ |
| RAM | M6 `iogpu.wired_limit_mb` 14336、MBP 15360。Swapouts は両機とも run ごとにバイトで記録する |
| 無効な run | guarded.sh が止めた run と exit が 0 でない run だけ (24 で改めた条件)。無効は巡回の最後に 1 本足す (1 腕 2 本まで) |
| 置き場 | 両機とも GGUF は内蔵 SSD、M6 の PLE BF16 は外付け SSD256。22・24 と同じ |
| MTP | spec、連鎖 1 (`Q38_MTP_CHAIN` 既定)。両機とも自分で MTP ヘッドも回す (単体) |
| サンプラ | instruct seed 1 (22 と同じ)。両機で列が途中から分かれうるが、比べるのは行の ms の中央値で、受理数と `experts` を添える |
| prompt / decode | `scratch/qwen38/prompt-code.tokens` 8,192 トークン、チャンク 2048、decode 400 トークン (23 と同じ長さ)。EOS で止まったら seed を 1 上げて run を足す |
| 窓 | ステップ 50〜200 (0-1 と同じ切り方。ステップ数で切る。1 ステップで 1〜2 トークン) |
| 閾値 | 0-4 の決め方。合否を後から動かさない |

## 1. 腕

| 腕 | 機 | コマンド | 比べる相手 |
| --- | --- | --- | --- |
| M-spec | MBP 単体 | `--qwen38-generate ... --q38-mtp spec --q38-new 400` | 6-spec、22 の MBP (off)、24 の S0 の `mbp_trunk_ms` |
| 6-spec | M6 単体 | 同じ (M6 の置き場) | M-spec、22 の M6 (off)、24 の S48 の `trunk_ms` |

- MTP off の腕は測らない (22 にある)。層で分けた腕も測らない (24 にある)。
- 両機は同じ時間帯に走ってよい (22 と同じ。共有するものは無い)。link は使わない。

## 2. 道具

1. **既存で足りる。**`bench/qfn/p6_decode.sh` は `--q38-mtp off` を固定で書いているので、MTP を引数 (`QFN_MTP`、既定 off) にする。それ以外は変えない。ランタイムの既定は変えない。
2. **(a) が出たときだけ作る**: `runQwen38Generate` に `Q38_SPLIT_PRE` を通す (`splitPreRouter`、既定 off。今は `--qwen38-decode-check` と prefill bench だけが設定している) と、verify の行に区間 (attention / GDN 入力 / GDN / hc) の GPU ms を出す。
   [qwen38/14 §2-1](../qwen38/14-MTP-NMAX2.md) の「幅の順で伸びる区間が入れ替わる」があるので、区間の比較は 1 起動 1 幅 (spec のループそのまま) で取り、両機で同じ形にする。
3. 集計は awk で (0-1 と同じ切り方)。スクリプトに入れるなら `bench/qfn/` に置く。

## 3. 測り方

- 両機の確認は 22 §4-1 と同じ (他のモデルのプロセスが無い、`memory_pressure -Q`、M6 の wired limit 14336、SSD256 のマウント)。link は要らない。
- `caffeinate -i QFN_MTP=spec bench/qfn/p6_decode.sh t2-trunk 3 400` を両機で (1 run 1 プロセス、`guarded.sh`、間 20 秒)。
- 1 run は prefill (MBP 83 s、M6 111 s) + decode 400 トークン (MBP 約 50 s、M6 約 1.5 分) で、両機 3 本ずつ 20 分前後。
- 証拠: `bench/qfn/results/2026-MM-DD-Mac15-6-t2-trunk/` と `2026-MM-DD-Mac18-5-t2-trunk/` (各 `run-<i>.log`、`vm_stat` の前後、`generate.out`、`hostinfo.txt`、`command.txt`)。結果文書は 26。
- 結果文書に書く表: 腕 × (step、draft、verify、pre 壁、pre GPU、route、routed GPU、head、topk、views、advise、kernel>gpu、routed gpu、I/O、I/O 以外、待ち、experts、受理、Pageins/ステップ、Swapouts、wired 最大) と、22 の T = 1 の同じ量。

## 4. 結果の行き先

| 判定 | P6 (22) | 次 |
| --- | --- | --- |
| (a) GPU | そのまま | 区間を特定 (§2 の 2)。直すなら decode の T = 2 の経路を M6 向け (P1 の MPS fp16 / routed のタイル、[15](15-Q1-F1-LAYER.md)) にする案になるが、**作る前に上限で判断を仰ぐ**: 分けた腕が S0 に勝てる幅は、測った I/O の差 (S0 111〜114 − S24 82〜84 = 約 30 ms/ステップ、24) を超えない |
| (b) host / フォルト | そのまま | 原因の側を直す (views の作り直しは [qwen38/13 §0-6](../qwen38/13-KERNEL-TO-GPU-WAIT.md)、フォルトなら wired と page cache の配分)。直したら S24 を 1 腕だけ n = 3 で測り直す |
| (c) 道具 | そのまま | split-decode の serve 側を直し、S24 と S48 を n = 3 で測り直す (23 の決め方のまま) |
| (d) 混在 | そのまま | 大きい方から |

- **どの結果でも P6 は動かさず、Q6 の k の格子 (44・46・48) も変えない。**この実験は Q6 と並行できる。
- 検証の行を 2 機に分ける形 (行 y を MBP、行 d を M6) は、この実験の結果に依らず作らない。上限は T = 2 が T = 1 より長い分 (qwen38/11 で約 40 ms/ステップ、24 の S0 では 219 − 22 の 192 = 27 ms) で、行 d は各層で行 y の GDN 状態と KV を要るので層ごとに往復 48 回が要る。作るなら、往復 48 回の秒を先に測る別の文書にする。
- 22 §3 の「A の M6 (常駐 + staging) の decode は測っていない」はそのまま開いている。

## 5. 測らないもの

- 送り (Q4 の腕 b の逆向き、MBP → M6 に expert を送る)。0-1 のとおり I/O を 0 にしても S48 は S0 に届かないので、この実験の前には測らない。
- M6 の RAM を MBP の expert の置き場にする形。同じ理由。
- 層またぎの先読み prev10 ([qwen38/12](../qwen38/12-PIPELINE-PREVIEW.md))。RAM の理由で閉じた経緯 ([qwen38/13 §0-1](../qwen38/13-KERNEL-TO-GPU-WAIT.md)) があり、この文書では扱わない。
- MTP off の単体 (22 にある)、連鎖 2、層で分けた腕 (24 にある)。
