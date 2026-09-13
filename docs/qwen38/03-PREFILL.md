# 03. Q2 ランナーの prefill — T トークン一括の forward、2048 トークンで 59 tok/s、8K で 41 tok/s

実測: 2026-09-13、M3 Pro 18 GB、macOS 15。対象は [01](01-Q2-FIRST-LIGHT.md)・[02](02-DECODE-SPEED.md) と同じ GGUF とランナー。
表記は 01 と同じ (実測 / 導出 / 未確認)。**速度の行はどれも n=1 (1 走行) で、数字だけを書く。**

## 0. 結論

1. **`Qwen38Runner.forward(tokens:startPos:)` が T ≥ 1 トークンを一度に流す。**decode は T=1 で同じ経路 (旧 decode 専用の経路は消した)。
   PLE も GPU に移した (host は n-gram の行の取り出しだけ)。
2. **正しさ**: 参照と、チャンク境界をまたいでも一致。France をチャンク 4 で logits ≤ 8.5e-7、Fuji 予算 16 (QSA の選択がチャンクの途中で始まる) をチャンク 16 / 53 で 53/53・≤ 2.5e-6、
   選択なしの Fuji 53 を 1 チャンクで top-1 53/53 (§1)。decode は T=1 で従来どおり PASS。
3. **速度** (ソースコード 8,192 トークンのプロンプト、§3):

   | | tok/s |
   | --- | ---: |
   | 1 トークンずつ (02 の decode) | 約 6〜10 |
   | 512 トークン、チャンク 128 (一括化しただけ) | 16.3 |
   | 2,048 トークン、チャンク 2048 (§2 の変更後) | **59.3〜64.3** |
   | 8,192 トークン、チャンク 2048 (2 チャンク目から QSA 選択) | **41.0** (選択ありのチャンクは 36.4〜37.0) |

   32K をこのまま流すと約 15 分 (導出)。**まだ実用ではない。**
4. 効いた変更は 4 つ: dense を「float32 に逆量子化 → MPS sgemm」(T ≥ 32)、選択の無いバッチの注意を sgemm 2 本、dense の residency set 常駐、GDN step を SIMD レーンに (§2)。
5. 8K の 1 チャンク (56 s) の内訳: **routed expert の読み待ち 14〜15 s、routed の GPU 13.9 s、選択ありの注意 10.6 s**、GDN 5 s、advise 3.9 s (§3)。次はこの 3 つ (§4)。

## 1. 形と正しさ

- 1 トークン分のバッファを T 行並べ、ディスパッチの格子の最後の軸をトークンにした。dense GEMV (`ggml_dense.metal`)、`qwen38.metal` の全カーネル、
  routed expert の 2 カーネル (`moe_ggml.metal`: 引数バッファのスロットを 512 に広げ、(トークン, k) の組を `pair_slot` でスロットに引く) が T 行を取る。
- 状態の持ち越し: GDN の conv 履歴 (`q38_gdn_conv_hist`) と再帰状態、PLE の dilated conv 履歴 (`q38_ple_hist`、GPU バッファ)、KV とインデクサの鍵。
- QSA: クエリごとに選ぶ (decode と同じ意味)。ブロック鍵はバッチ内で 4 トークン目が入ったブロックぶんを一度に作る。
- 検査: `--qwen38-prefill <ref log> [--q38-chunk N]` が参照の列 (プロンプト + greedy 続き) を N ずつ流し、全位置の logits を比べる。

| 検査 | top-1 | logits 最悪 |
| --- | ---: | ---: |
| decode France (T=1) | 10/10 | 8.50e-7 |
| decode Fuji 予算 16 (T=1) | 53/53 | 6.95e-7 |
| prefill France、チャンク 4 | 10/10 | 8.50e-7 |
| prefill Fuji 予算 16、チャンク 16 | 53/53 | 6.95e-7 |
| 同、sgemm を T ≥ 16 に下げて | 53/53 | 1.78e-6 |
| prefill Fuji 予算 16、チャンク 53 | 53/53 | 2.48e-6 |
| prefill Fuji 選択なし、チャンク 53 (注意も sgemm) | 53/53 | (logits の参照なし) |

## 2. 速くしたもの

### 2-1. dense: 逆量子化 + MPS sgemm (`GGMLDenseGEMV`、T ≥ 32)

T 行の GEMM を Q8_0 のまま回すと、トークンごとに重みを読み直す計算律速になる。テンソルを GPU で float32 に展開して MPS に渡す (`--q8-gemm-bench`、GPU ms、5 回の中央値):

| T | attn_qkv 10240×2560: Q8_0 GEMM | 展開 + sgemm | ssm_out 2560×6144: Q8_0 | 展開 + sgemm |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4 | 1.3 + 0.9 | 0.2 | 0.8 + 0.6 |
| 4 | 1.4 | 1.3 + 2.0 | 0.8 | 0.8 + 1.4 |
| 16 | 2.9 | 1.3 + 0.9 | 0.9 | 0.7 + 0.5 |
| 64 | 5.5 | 1.3 + 0.9 | 3.1 | 0.7 + 0.6 |
| 512 | 45.1 | 1.3 + 5.4 | 26.8 | 0.8 + 3.4 |

出力差 ≤ 3e-6。閾値は T ≥ 32 (`Q38_MPS_MIN_T`)。LM head (6.4 億重み) は展開しない。512 トークンのチャンクで pre-router の GPU が 8.6 → 2.9 s。

### 2-2. 注意: 選択の無いバッチは sgemm 2 本 (`attentionSgemm`、T ≥ 32)

02 の 5 パスは (トークン × ヘッド × キー) の 3 次元の格子で、2048 × 24 × 2048 × 32 レーンのスレッド数そのものが重い。
KV グループごとに Q・K・V を連続行に集め、`scores = Q Kᵀ/√D` と `out = W V` を sgemm にし、max・Σexp・重み (因果の先は 0) は既存のレーンのカーネル。
2,048 トークン 1 チャンクで注意の GPU が **7,152 → 1,022 ms**。QSA の選択が要るバッチは 5 パスのまま。

### 2-3. dense の常駐 (residency set)

expert を読むと dense (Q8_0/F16/F32 のビュー、約 4.7 GB) がページキャッシュから追い出され、pre-router が GPU の外で待っていた
(512 トークンの後の decode 1 トークンが pre 1,309 ms / GPU 107 ms)。全ビューを 1 つの `MTLResidencySet` に入れてキューに付けると **1.52 → 0.19 s**。
`Q38_DENSE_RESIDENT=0` で外せる。

### 2-4. GDN step をレーンに

T 回の再帰を 1 スレッドが 128 次元ずつ回していたのを、(dv, hv) ごとに 32 レーンで次元を分け、各ステップの k·S と q·S を `simd_sum` にした。
2,048 トークン 1 チャンクの全体が 43.4 → 54.6 tok/s (§2-2 の前、各 n=1)。減衰と β の前計算 (`q38_gdn_gates`) は gdn_step 3,022 → 3,022 ms で効かなかった (残した、計算は同じ)。

### 2-5. 効かなかったもの

- **advise のまとめ**: 隣接 expert を 1 回にまとめて 28K → 17K 回にしても時間は同じ (1.45〜1.63 s / 128 トークン)。コストは回数でなくバイト量側。
- **テンソル丸ごと advise** (層の expert の半分以上が選ばれたら、`Q38_ADVISE_WHOLE=0.5`): advise 3.5 → 4.6 s、読み待ちも増えて 34.4 → 36.4 s。既定は off。

## 3. 内訳 (実測、`--qwen38-prefill-bench`、`Q38_SPLIT_PRE=1`)

ソースコード (`MmapExpertMapping.swift` ほか 4 本を `<file>` で包み、上流のトークナイザで 8,192 トークン)。チャンク 2048。

| 段 (1 チャンク) | 0..<2048 | 2048..<4096 (QSA 選択あり) |
| --- | ---: | ---: |
| 合計 | 31.8 s (64.3 tok/s) | 55.3 s (37.0 tok/s) |
| PLE の行 (host) | 0.1 s | 2.5 s |
| pre-router wall / GPU | 9.7 / 7.5 s | 20.1 / 17.2 s |
| うち注意 | 1.0 s | **10.6 s** |
| うち GDN (入力 GEMM / step / 出力) | 1.36 / 3.01 / 0.63 s | 1.36 / 3.06 / 0.64 s |
| うち hc 2 本 / 共有 expert + router | 1.17 / 0.28 s | 1.17 / 0.28 s |
| route (top-k / advise) | 4.4 s (0.67 / 3.5) | 4.6 s (0.69 / 3.8) |
| routed wall / GPU | 15.9 / 13.8 s | **28.2 / 13.9 s** |
| 選ばれた expert (層ごとの異なり数の和) | 19,183 (= 48×512 の 78 %) | 20,077 |

8,192 トークン全体: 199.7 s (41.0 tok/s)、peak footprint 3.04 GB、Swapouts 増分 0。その直後の decode 1 トークン 0.43 s。

decode (T=1) は同日に France 位置 1〜9 で 0.09〜0.11 s (3 本中 2 本。残る 1 本は GPU 時間が全体に倍で 0.19〜0.28 s、02 §4 と同じく原因は特定していない)。

## 4. 次

1. **routed expert の読み**: 2 チャンク目から読み待ちが 14 s (1 チャンク 30 GB 前後、dense 常駐と KV でページキャッシュが減った)。
   チャンクを 4096 に上げる (異なり数の割合が上がり 1 トークンあたりの読みが減る見込み、導出) か、読みの並べ方を変える。
2. **routed expert の GPU** (13.9 s / 2048 トークン、全体の 1/4〜2/5): 組ごとに IQ2_XXS の行を展開し直している。expert ごとにトークンを束ねて 1 回だけ展開する形を試す。
3. **QSA 選択ありの注意** (10.6 s): host のクエリごとのソートと 5 パスをやめ、GPU の top-k + sgemm (選択をマスクで) にする。32K の運用点ではこれが全チャンクに効く。
4. GDN step (3 s): 時間方向は直列のまま。チャンク内の並列形 (delta rule の WY 表現) は大きいので後。
5. PLE の行の読み (冷えていると 2.5 s): 行を先に advise するか並列に読む。

## 5. 足したもの・消したもの

| 種類 | パス |
| --- | --- |
| ランナー | `Qwen38Runner.forward(tokens:startPos:allLogits:)`、`maxBatch`、`attentionSgemm`、dense の residency set、`StepProfile` に `distinctExperts`・`routeTopK/Views/Advise`・`adviseCalls` |
| カーネル | `qwen38.metal` を全面的に T 行化 (+ `q38_gdn_conv_hist`・`q38_gdn_gates`・`q38_attn_gather_q/kv`・`q38_attn_scatter_out`・`q38_ple_*`)、`ggml_dense.metal` に token 軸と `ggml_q8_0_dequant_f32`・`ggml_f16_dequant_f32`、`moe_ggml.metal` の 2 本を T 行 + 512 スロットに |
| 検査 | `--qwen38-prefill`、`--qwen38-prefill-bench <tokens> [--q38-tokens N] [--q38-chunk C]`、`--q8-gemm-bench <gguf> [T]` |
| 環境変数 | `Q38_MPS_MIN_T` (32)、`Q38_ATTN_MPS_MIN_T` (32)、`Q38_DENSE_RESIDENT` (on)、`Q38_ADVISE_GAP` (0)、`Q38_ADVISE_WHOLE` (off) |
| 消した | `--q38-small-bench` (カーネルの引数が T 行化で変わり、そのままでは誤った束縛で走るため。02 §2-4 の数字は旧カーネルのもの) |

再現:

```bash
B=.build/release/TsugumiKernelCheck
$B --qwen38-prefill scratch/qwen38/ref-france-dump.log --q38-ref-logits scratch/qwen38/ref-france.logits --q38-chunk 4
$B --qwen38-prefill scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 16
Q38_SPLIT_PRE=1 $B --qwen38-prefill-bench scratch/qwen38/prompt-code.tokens --q38-tokens 8192 --q38-chunk 2048
```

`scratch/qwen38/prompt-code.tokens` は §3 の 4 ファイルを上流の `tokenizer.json` で符号化した先頭 8,192 個 (カンマ区切り)。

## 6. 再開手順 (新しいセッションで続けるとき)

### 6-1. 状態

- コードはこの文書と同じコミットまで入っている (prefill 本体は `e0f7ebe`)。ランナーは `Sources/Tsugumi/Runtime/Qwen38/Qwen38Runner.swift` 1 本 (`forward` が T ≥ 1、`step` は T=1)。
- ウェイト・トークナイザ・ds4 の資料の在処は [01 §7-1](01-Q2-FIRST-LIGHT.md) のまま。
- `scratch/qwen38/` (git 管理外) にあるもの: `ref-france-dump.log` + `.logits` (10 位置)、`ref-fuji-idx16.log` + `.logits` (予算 16、53 位置)、
  `ref-fuji-ple.log` (選択なし 53 位置、logits 無し)、`expert-fixture-l24/`、`prompt-code.tokens` (8,192 トークン)。
  消えていたら参照は 01 §7-3、`prompt-code.tokens` は §3 の 4 ファイルを `~/LLM/venv/bin/python` + `tokenizers` で符号化し直す。
- 運用点は thinking 無効・32K・エージェント主体・英語主体 (メモリ `qwen38-operating-point`)。32K の prefill はいま約 15 分。

### 6-2. 検査を一通り (どれも数分以内、GPU は 1 本ずつ、間に 20 秒)

```bash
swift build -c release --product TsugumiKernelCheck
B=.build/release/TsugumiKernelCheck
G=~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf
$B --q2-expert scratch/qwen38/expert-fixture-l24
$B --ggml-dense $G
$B --qwen38-decode scratch/qwen38/ref-france-dump.log --q38-ref-logits scratch/qwen38/ref-france.logits
$B --qwen38-decode scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16
$B --qwen38-prefill scratch/qwen38/ref-france-dump.log --q38-ref-logits scratch/qwen38/ref-france.logits --q38-chunk 4
$B --qwen38-prefill scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 16
Q38_MPS_MIN_T=16 Q38_ATTN_MPS_MIN_T=16 $B --qwen38-prefill scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 16
$B --qwen38-prefill scratch/qwen38/ref-fuji-idx16.log --q38-ref-logits scratch/qwen38/ref-fuji-idx16.logits --q38-indexer-top-k 16 --q38-chunk 53
$B --qwen38-prefill scratch/qwen38/ref-fuji-ple.log --q38-chunk 53
```

期待値は §1 の表 (全部 PASS、logits ≤ 2.5e-6)。速度は `Q38_SPLIT_PRE=1 $B --qwen38-prefill-bench scratch/qwen38/prompt-code.tokens --q38-tokens 8192 --q38-chunk 2048` (約 200 s) で §3 と比べる。
走らせる前に `vm_stat` の Swapouts を控え、終わったら増分を見る。

### 6-3. 次にやること (§4 の順を、32K に効く順に並べ直したもの)

**1 番は済んだ ([04](04-QSA-GPU.md)、8K で 46.5 tok/s)。続きは 04 §4。**

1. **QSA 選択ありの注意を GPU に** (1 チャンク 10.6 s + host のソート)。場所は `attention(_:il:pos0:T:)` の
   「Per query: top kBlocks」のループ (host で `sorted()`) と、その後の 5 パス。案:
   - ブロックのスコア `idxScores [T][nBlocks]` から、クエリごとの上位 `kBlocks` を GPU で選ぶ (しきい値を二分探索で求めて数える形なら並べ替え不要。同点は小さいブロック番号が勝つ規則を守る)。
   - 注意は `attentionSgemm` と同じく KV グループごとに `Q Kᵀ` を全 n 列で sgemm し、重みのパスで「選ばれていない列を 0」にする (選択をビットマスクか列ごとのフラグで渡す)。
     メモリは `T × H/Hkv × n` 浮動小数 (T=2048・n=32K で 1.6 億 = 630 MB/グループ) なので、長文脈ではチャンクを小さくするか行を分割する。
   - **正解の作り方**: 32K の CPU 参照は回せない (19 s/トークン)。予算 16 の Fuji (§1) で参照と照合し、長文脈は**今の host 選択 + 5 パスの経路をオラクル**にして
     8K プロンプトの最後のチャンクの logits を新旧で比べる (`allLogits` を使う小さな比較モードを検査に足す)。
2. **routed expert の読み待ち** (1 チャンク 14 s)。まず `--q38-chunk 4096` で 8K を回し、1 トークンあたりの読みと footprint を見る
   (`attnScores` が 5 パス時 `maxBatch × 24 × 2051 × 4` B = 4096 で 790 MB になる点に注意)。advise の既定は「選んだ expert、隣接だけまとめる」。
3. **routed expert の GPU** (13.9 s / 2048)。`moeRouted` と `moe_ggml.metal` の 2 本。組ごとに IQ2_XXS / Q2_K の行を展開し直している。
   expert ごとにトークンを集め、その expert の gate/up/down を 1 回だけ float32 に展開して sgemm にする形を、まず 1 層の A/B (`--q8-gemm-bench` と同じ作り) で見積もってから入れる。
4. GDN step (3 s)、PLE の行の冷えた読み (最大 2.5 s)。

### 6-4. 落とし穴 (このセッションで踏んだもの)

- **GPU の 1 スレッドに長いループを書かない。**4 スレッド × 2560 要素の RMS が 340 µs、注意 n=2048 が 1 層 120 ms、GDN step も同じ理由で遅かった。32 レーン + `simd_sum` か要素ごとのスレッドにする。
- **F16/F32 の行を 1 要素飛びで読むと 3〜9 倍遅い**。32 要素のチャンクで読む (Q8_0 のブロックと同じ形)。
- **T ≥ 32 の行列積は float32 に展開して MPS sgemm**。Q8_0 のまま T 行回すのは重みの読み直しで計算律速になる。T=1 は逆 (直接読みが速い)。
- **dense は residency set に入れておかないと expert の読みに追い出される** (pre-router の wall が GPU の数倍になったらこれ)。
- **wired_limit を上げても速くならない**。効いているのは RAM 18 GB のページキャッシュで、GPU の wired 上限には当たっていない。
- `swift build` は Metal をコンパイルしない (実行時にコンパイル)。**カーネルの引数を変えたら、そのカーネルを呼ぶ検査を必ず 1 本走らせる。**
  引数のずれは落ちずに静かに間違う (`--q38-small-bench` を消したのはこのため)。
- **zsh では `env $VARS cmd` が単語分割されない** (`${=VARS}`)。腕の片方が素通しになり、A と同じ数字が出る。
- 走行の 1 本目だけ GPU 時間が全体に倍になることが何度かあった (原因は未特定)。**速度は 1 本で判断しない。**別プロセスのテストが GPU を使っていたこともある (ユーザーが止めた)。
- `git rm --cached` で先に 1 ファイルだけステージしたまま `git add` が失敗すると、コミットがそれだけになる。コミット後に `git show --stat HEAD` を見る。
