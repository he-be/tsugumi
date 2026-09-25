# 07. 設計を決める実験 — 先に測り、式に入れ、それから作る

**前提 (2026-09-25 ユーザー): M6 では必ず tensor ops を使う (MPS で同じ速さなら MPS でよい)。**M6 の秒は tensor ops の経路で取る。fp32 のコードで取った E1 の M6 の値は、それだけで分け方を決めない ([11](11-Q1-REOPEN-TENSOR-OPS.md))。

2 機の分け方は 2 つある ([08](08-CANDIDATES.md): **A** 層で分ける / **B** expert で分ける)。どちらが速いかは、
今は 1 つも測っていない量 (M6 で QFN の各段が何秒か、link が何秒か) で決まる。**推論で選ばない。**
ここに置く E0〜E4 を先に取り、[08 §3](08-CANDIDATES.md) の式に入れて、記録の 1 ターンで秒の小さい方を作る。

実験はどれも既存の道具で取れる (E4-b だけカーネルを 1 本書く)。証拠は `bench/qfn/results/YYYY-MM-DD-<Mac15-6|Mac18-5|pair>-e<N>/`、
規則は [02](02-EVIDENCE.md)。**n ≥ 3、同じ入力、run ごとに「まだ読んでいない層」か新しいプロセス。**

## E0. link — 生 TCP の速さと往復

| 項目 | 内容 |
| --- | --- |
| 問い | `bridge0` の生 TCP で、84 MB を 1 回送る秒 (A の 1 チャンク) と、21 MB を送って 21 MB を受ける往復を 48 回続ける秒 (B の 1 チャンク) |
| 道具 | 小さな送受信プログラム (Swift か C、`bench/qfn/link_probe`)。`ping -c 20` の rtt 中央値も取る |
| 出す量 | **l_1** (84 MB 片道、秒)、**l_48** (21 MB 往復 × 48、秒)、rtt (ms)、1 GiB の連続の GB/s |
| 決めること | l_48 が 1 チャンクの数秒に対して大きければ B は不利。l_1 が大きければ A も link が効く |
| 注意 | TWO_MACHINE_DEV §3 の 37.8 Gb/s は証拠なし。ssh は使わない (暗号化で 1.5 GB/s に落ちる、[03 §5](03-FACTS.md))。両端の IP は起動のたびに `ifconfig bridge0` |

## E1. M6 で QFN の prefill を今のコードのまま動かす (段の秒)

| 項目 | 内容 |
| --- | --- |
| 問い | M6 で dense (pre-router)、routed の GPU、pread、top-10、PLE がそれぞれ何秒か。MBP の同じ run と並べる |
| 道具 | `Scripts/qwen38/guarded.sh out.txt env Q38_SPLIT_PRE=1 .build/release/TsugumiKernelCheck --qwen38-prefill-bench <tokens> --q38-tokens 4096 --q38-chunk 2048 --q38-gguf <M6 のパス> --q38-ple <外付けの BF16>`。**両機で同じコマンド、同じトークン列** (MBP の `scratch/qwen38/` のトークン列を M6 へ写す)。M6 は PLE を外付けから読むので、PLE の秒は別に見る |
| 出す量 | 層あたりの **d_X** (pre-router GPU ÷ 48)、**r_X** (routed GPU ÷ 48)、**t_X** (top-10 ÷ 48)、**p_X** (PLE、チャンクあたり)、pread の秒、wired の最大と Swapouts (X ∈ {MBP, M6}) |
| 決めること | r_M6 / r_MBP と d_M6 / d_MBP。fp32 sgemm のままの M6 が MBP と比べてどうか。**M6 の 16 GB に 4K 文脈の 1 チャンクが乗るか** (乗らなければ A の M6 側は文脈長で詰まる) |
| 注意 | 4096 トークンは 2 チャンク。1 本目は PLE と dense が冷えている。両方の値を残す。M6 の `iogpu.wired_limit_mb` は既定 (0) のまま |

## E2. M6 の SSD を QFN の expert の大きさで読む (内蔵 / 外付け / 同時)

| 項目 | 内容 |
| --- | --- |
| 問い | M6 が 1 層の expert (762,839,040 B、down サイドカー後は約 710 MB) を読むのに何秒か。外付けを足すと何秒か |
| 道具 | `Scripts/qwen38/readbench.c` (`expert_ranges.py` で本体と down サイドカーの範囲を出す) を M6 でビルド。`pread 4 16` を内蔵だけ / 外付けの写しだけ / 2 プロセス同時 (`bench/m6/dual_ssd_probe.sh` の形)。外付けの写しは E2 のために本体の expert 範囲を `/Volumes/SSD256/qfn/` に置く (Q4 の写しの先取り) |
| 出す量 | **s_M6** (1 層、内蔵だけ)、**s_M6+** (内蔵 + 外付けを同時に読んだときの 1 層)、MBP の s_MBP は qwen38/05 の 6.2〜6.6 GB/s から導出し、同じ道具で取り直す |
| 決めること | M6 の読みの秒。A なら M6 の層数 k を、B なら M6 の担当分 σ を、この秒が押し下げる |
| 注意 | 「まだ読んでいない層」で取る。同じ層を 2 回読むとキャッシュの速さになる (qwen38/05 §5) |

## E3. link の往復を含めた B の 1 層の形 (机上ではなく、模擬で)

| 項目 | 内容 |
| --- | --- |
| 問い | B の 1 層で、MBP が `mixed` (T × 2560 fp32) と割り当てを送り、M6 が担当の expert を計算して部分和を返す往復が、GPU の秒とどう重なるか |
| 道具 | E0 の送受信プログラムに、M6 側で `--q2-gemm-bench` 相当の 1 層の GEMM (E4 のカーネル) を挟む。MBP 側は送信 → 自分の担当の GEMM → 受信の順。48 層ぶん繰り返す |
| 出す量 | B の 1 チャンクの routed 部分の秒 (実測)。[08 §3](08-CANDIDATES.md) の式の max(…) の項が実測に置き換わる |
| 決めること | 式の B の値が実測と合うか。合わなければ式を直す (link の重なり方の仮定が外れている) |

## E4. M6 の expert GEMM — fp32 sgemm (今) と fp16 tensor ops (候補)

| 項目 | 内容 |
| --- | --- |
| 問い | (a) 今の routed の GEMM (dequant → float32 → MPS sgemm) は M6 で 1 層何秒か。(b) fp16 のタイルに展開して tensor ops (Gemma の [m6-prefill/40](../m6-prefill/40-MOE-GEMM-NAX.md) の形) にすると何秒か |
| 道具 | (a) `TsugumiKernelCheck --q2-gemm-bench <gguf> <route.bin> 4 <iterations> 2048` を両機で。`route.bin` は MBP で `Q38_DUMP_ROUTE` を付けて 1 チャンク流して作り、M6 へ写す。(b) 同じベンチに fp16 tensor ops の腕を足す (**カーネル 1 本を書く。ランタイムには入れない**) |
| 出す量 | **r_M6** (a)、**r_M6,nax** (b)、(b) の出力と (a) の出力の相対差 (fp16 の丸め) |
| 決めること | M6 の GEMM が MBP の何秒に対して何秒か。M6 の r には (b) か (c) の速い方を使う。精度は ds4-metal と同じく、腕 A との差 ≤ 2e-3 × scale を合格とする (fp16 の行列積は qwen38/01 の 1.7e-6 を原理的に満たせないので、その幅は使わない、[11 §0](11-Q1-REOPEN-TENSOR-OPS.md)) |
| (c) | ds4-metal `origin/qwen2-nax-tiles` の mid / down の tensor ops カーネル ([11 §1](11-Q1-REOPEN-TENSOR-OPS.md)) を同じベンチの腕 D に移植する。M6 だけ。(b) (MPS fp16) と並べて「MPS で同じ速さか」に答える |
| 注意 | 単体ベンチの勝ちは本番ループで確かめるまで勝ちではない (m6-prefill/05 §2)。ここで得るのは式に入れる秒だけ |

## E5. M6 の dense を tensor ops で

| 項目 | 内容 |
| --- | --- |
| 問い | M6 で dense (pre-router の射影、shared expert) を fp16 の tensor ops で回すと 1 層何秒か (d_M6,nax)。今の d_M6 (fp32) は 0.0948 s/層 ([10 §2](10-Q1-SPLIT-DECISION.md)) |
| 道具 | 未定。pre-router の各射影の形 (T = 2048) を KernelCheck のベンチで、MPS fp16 と、必要なら ds4-metal の dense の tensor ops の形で |
| 出す量 | **d_M6,nax**、fp32 の出力との相対差 |
| 決めること | A (M6 が層ごと dense も持つ) の M6 の 1 層の秒。B では MBP に残る dense が効くので、B の形そのものを見直す根拠になる |

## 決め方 (計測の前に置く。後から動かさない)

1. E0〜E4 の量を [08 §3](08-CANDIDATES.md) の式に入れ、A と B の「1 チャンクのラウンド」「2 チャンク以上の定常」の秒を出す。
2. 記録の 1 ターン ([qwen38/34](../qwen38/34-ROUND-BREAKDOWN.md) のアプリの記録、ラウンドごとの新規トークン数) を式に通し、**1 ターンの prefill 合計秒**を A と B で出す。
3. 小さい方を作る。差が Q0 の揺れの幅の 2 倍より小さければ **B** (M6 に状態が無く、decode と prompt cache に手を入れない方) を作る。
4. M6 側は tensor ops の経路で作る (前提)。E4 の (b)・(c) と E5 の速い方を M6 の d・r に入れて 1〜3 をやる。
5. 式と実測が 1 割以上ずれたら (E3)、設計の前に式を直して 1〜3 をやり直す。

決めた結果は結果文書 (10 番台) に、式に入れた数字と証拠のリンク付きで書く。README の Gate の表はそこで更新する。
