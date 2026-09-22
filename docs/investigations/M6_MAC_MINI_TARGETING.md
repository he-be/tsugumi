# 調査: M6 Mac mini を対象機にしたら何が改善できるか

作成: 2026-09-01
対象機: **Mac mini (M6, 2026)** 16 GB / 256 GB、¥149,800、2026-09-22 発売
現行機: **Apple M3 Pro 18 GB / macOS 15.7.5**（`tsugumi-m3pro-status.md`）

表記は他の調査文書と同じ:

- **実測** = 手元で数字を取ったもの。条件を併記する。
- **導出** = 実測または公称値から計算したもの。誤差要因を併記する。
- **公称** = Apple / 論文 / 報道が言っているだけで、手元では取っていないもの。
- **未確認** = 根拠を持っていないもの。断定しない。

**この文書の実測はすべて M3 Pro のものである。M6 の数字は 1 つも手元にない。**
「M6 で N 倍」と書ける段階ではない。§7 の 5 つを実機で取るまでは §2-4 の期待値は導出のままである。

> **復元 (2026-09-13)**: 本書は 2026-09-01 に書かれ、同日の `ee0c6ef` で調査ログ整理の巻き添えで削除された。
> 到着日 (9/22) が近いので戻した。書かれた時点の対象は Gemma 経路で、Qwen 経路
> ([docs/qwen35moe](../qwen35moe/README.md)) はまだ無い。**Qwen 経路から見た M6 と、
> Perplexity Lily 記事との対応は [qwen35moe/43-LILY-IDEAS.md](../qwen35moe/43-LILY-IDEAS.md)** に書いた。
> §10 に、その後の実測で古くなった箇所を挙げる。

---

## 0. 結論を 3 行で

1. **M6 で効くのは prefill / TTFT だけで、しかも自動では効かない。**
   このリポジトリには MPP tensor-ops のカーネルが 2 本入っているが、M3 Pro（apple9 / macOS 15）では
   `__HAVE_TENSOR__` が立たず**コンパイル時に丸ごと落ちている死んだ経路**である（実測、§2-3）。
   M6 + macOS 27 で初めて生きる。そこを書き直せば prefill が伸びる。**載せ替えるだけでは伸びない。**
2. **decode は M6 でもほぼ伸びない。** Neural Accelerator は decode に効かない（公称、§2-2）。
   そして **16 GB 構成の DRAM 帯域は 153 GB/s で、M3 Pro の 150 GB/s と実質同じ**（+2%）。
   170 GB/s は 24/32 GB 構成だけの値である（§5-1）。ここで効きうるのは公称「ストレージ 2 倍」の方で、
   decode の 12.5 ms/token の I/O に直接当たる（§5-2）。**ただし 256 GB 構成の実速度は未確認。**
3. **16 GB は 18 GB からの後退だが、本体経路には効かない。**
   Gemma 経路の peak は 2〜7 GB なので問題ない。効くのは**同一ホストでの比較ベースライン**の方で、
   MLX（peak GPU alloc 14.7〜15.3 GB）と llama.cpp（15 GB）が 16 GB 機ではまともに回らなくなる（§5-3）。
   24 GB は 512 GB 抱き合わせで **+¥72,000**。この額を払う理由は今のところない（§6）。

---

## 1. M6 / Mac mini の確定スペック（すべて公称）

| 項目 | M6 Mac mini | 手元の M3 Pro | 差 |
| --- | --- | --- | --- |
| プロセス | TSMC 2 nm（Apple 初） | 3 nm | |
| CPU | 12 コア = **super 2 + P 4 + E 6** | 12 コア（P 6 + E 6） | 大コアは 6 本で同数 |
| GPU | **12 コア、全コアに Neural Accelerator** | 18 コア、行列ユニットなし | **コア数は減る** |
| GPU の AI 演算ピーク | M5 比 +30%、M1 比 8 倍超 | — | 絶対値は未公表 |
| Neural Engine | **デュアル 16 コア**、ピーク 2 倍 | 16 コア × 1 | |
| メモリ | 16 GB / **153 GB/s**（24・32 GB のみ 170 GB/s） | 18 GB、150 GB/s | **帯域 +2%**、容量 −2 GB |
| SSD | 256 GB〜、**前世代 mini 比 2 倍速** | ランダム 3.72 MB pread で 4.19 GB/s（実測） | §5-2 |
| ポート | Thunderbolt **4** ×3、2.5GbE（10GbE 選択可） | — | TB5 は M5 Pro 側のみ |
| OS | macOS 27 Golden Gate 同梱 | 15.7.5 | §2-3 |

Apple の宣伝値（公称、条件不明）:

- LM Studio の**プロンプト処理**が M4 mini 比 **4.8 倍**、M1 mini 比 13.5 倍
- CPU は「世界最速のシングルスレッド」、マルチスレッドは M5 比 1.2 倍
- グラフィックスとストレージが M4 mini 比 2 倍

宣伝値がすべて**プロンプト処理（= prefill）**で語られていることに注意する。
decode（token generation）の倍率は Apple 自身が出していない。これは §2-2 と整合する。

**「最大 170 GB/s」の "最大" は構成依存である**（Apple 技術仕様、実質公称）:

| 構成 | 帯域 |
| --- | ---: |
| 16 GB（16/256、16/512） | **153 GB/s** |
| 24 GB / 32 GB | 170 GB/s |
| 参考: M5 Pro（24 GB〜） | 307 GB/s |
| 参考: 手元の M3 Pro | 150 GB/s |

**本命の 16/256 は 153 GB/s** で、M5 と同値、M3 Pro に対して +2% でしかない。§5-1 で効く。

---

## 2. GPU と Neural Accelerator — ここが本命

### 2-1. いまの GPU 時間の内訳（実測、M3 Pro）

`docs/investigations/PREFILL_THROUGHPUT.md` §7-9。2478 トークン、QAT、48 スロット / chunk 2048。
**pp 231 tok/s、壁時計の 97% が GPU busy。**

| 区分 | scalar | tiled（現行） |
| --- | ---: | ---: |
| routed MoE | 8.58 s | **8.59 s（GPU 時間の 64%、手つかず）** |
| attention（SWA + full） | 2.02 s | 2.02 s（手つかず） |
| q/k/v 射影 | 6.35 s | 1.01 s |
| o 射影 | 3.06 s | 0.57 s |
| 共有 MLP | 4.15 s | 0.74 s |
| **GPU 合計** | 24.78 s | **13.44 s** |

300 pp tok/s に必要な実効演算性能は 2.0 TFLOP/s（導出、同文書 §0）。現在 0.47。
**残っている的は routed MoE 一つに集中している。**

### 2-2. Neural Accelerator に到達する唯一の経路

M5 世代から GPU の各コアに行列積和ユニット（Neural Accelerator）が入り、
**`mpp::tensor_ops::matmul2d`（Metal 4 / MSL 4.0 の Metal Performance Primitives）からしか叩けない**（公称）。

ここから 2 つの帰結が出る:

1. **いま書いてある `simdgroup_matrix` のタイル化カーネルは Neural Accelerator を使わない。**
   通常のシェーダコアで回る。つまり **M6 に載せ替えただけでは prefill は速くならない。**
   むしろ GPU 18 → 12 コアなので、**素の fp16 スループットは横ばいか微減の可能性がある**（導出、§7-3/7-4 で要実測）。
2. **decode には効かない。** BaseRT（arXiv 2607.19438, M5）は
   「token generation はメモリ帯域律速のままで、単トークン decode に tensor core の利得はない」と明言。
   Apple の MLX 計測でも M5 の TTFT は M4 比 3.19〜3.97 倍に対し、**decode は 1.19〜1.27 倍**（= 帯域 +28% とほぼ同じ）。

BaseRT が prefill GEMM / **MoE expert GEMM** / prefill attention を `matmul2d` に通して得た値（公称）:
llama.cpp 比で prefill 最大 6.4 倍、MLX 比 3.9 倍。**MoE モデルで差が最大**（llama.cpp 比 29〜120%）。
これは §2-1 の「残りは routed MoE 8.59 s」にそのまま重なる。

### 2-3. コード上の事実 3 つ（実測、この checkout）

1. **tensor 経路は macOS 27 なら自動で生き返る。**
   `MetalContext.shaderLanguageVersion`（`Infrastructure/Metal/MetalContext.swift:249-259`）は
   macOS 26+ で `.version4_0` を返す。macOS 15 では `.version3_2` なので `__HAVE_TENSOR__` が未定義になり、
   `Metal/TensorCore/tensorops.metal` のカーネルはライブラリから落ち、呼び出し側が非 tensor 経路に落ちる。
   **移植作業ゼロで復活する。** ただし復活するのは以下の 2 本だけ:
   - `mpp_prefill_affine_threadgroup_f16`（`Kernels/TensorCore/MPPPrefillInt4QMM.swift`）
   - TensorOps 2D prefill attention（`.fullTensorOps2DValidityV2`、`Kernels/Attention/PrefillAttention.swift`）

2. **`supportsFamily(.apple10)` のゲートが危ない**（`PrefillAttention.swift:74`）。
   **M5 は apple9 を名乗ったままで、Neural Accelerator は feature flag で引けない**という報告がある
   （Rigel / BaseRT、**未確認** — 一次資料で裏を取っていない）。M6 が apple9 / apple10 のどちらを返すかも未確認。
   このままだと **M6 で黙って非 tensor 経路に落ちる**。
   ゲートを family 判定ではなく**「パイプライン生成が成功するか」**に変えるべき。

3. **`MPPPrefillInt4QMM` は `affineGroupSize == 64` 専用で、group-32 の QAT チェックポイントでは死んでいる**
   （`MPPPrefillInt4QMM.swift:22-35`、`PREFILL_THROUGHPUT.md` §7-9）。
   simdgroup 版で「scale/bias をタイル番号ではなくグローバル K 位置 `(k0 + kk) / kPrefillGroupSize` から索く」
   ことで 32/64 両対応にした修正が既にある。**同じ手を MPP 版に移すのが最初の一手。**

### 2-4. 期待値（導出。誤差は大きい）

MoE 8.59 → 2 s、attention 2.02 → 0.7 s に落ちたとして GPU 13.44 → 5 s 台、
2478 トークンで **pp 350〜400**（現行 231）。誤差要因:

- M6 の 12 コア GPU の絶対性能が未確認（§7-3）。BaseRT の倍率は M5 の**中での**比であって、
  M3 Pro 18 コアとの絶対比較ではない。**比を家族の外に持ち出さない。**
- int4 デクォンタイズを挟んだときの `matmul2d` 実効効率が未確認。
  BaseRT は 4bit/8bit をタイル単位で展開して `matmul2d` に渡す形を取っており、
  現行 `mpp_prefill_affine_threadgroup_f16` の「bounded FP16 staging に展開してから MPP」と同じ構造。
- prefill の壁時計は GPU busy 97%（実測）なので、GPU が縮むと**残りの 3% と I/O が相対的に効いてくる**。
  GPU を半分にしても壁時計は半分にならない。

---

## 3. ANE のデュアル化

### 3-1. 分かっていること

- M6 は**デュアル 16 コア Neural Engine、ピーク 2 倍**（公称）。
- Apple は「システムフレームワークが自動で両エンジンに振る」と言っており、
  **開発者が 2 基を明示的に叩く API は出ていない**（公称）。Core ML 経由という状況は変わらない。

### 3-2. 手元の ANE 実測（`~/dev/Irodori-TTS` docs/experiments/13, 15。同じ M3 Pro）

| 項目 | 値 |
| --- | ---: |
| ANE 16 コアの fp16 実効（12 層 Transformer スタック 248M） | **7.3〜8.7 TFLOPS** |
| 同じ形の GPU（MPS fp16 eager） | 4.0 TFLOPS |
| 比 | **約 2 倍** |

**M3 Pro の時点で ANE は GPU の 2 倍出る。** M6 ならこれが最大 2 基ぶん。数字だけ見れば魅力的である。

> **2026-09-22 追記: M6 でこの比は逆転した。**M6 12 コア GPU の密 fp16/bf16 は **19.0 TFLOPS**
> （MLX 19.2 / PyTorch MPS 18.7〜19.0 の 2 系統一致、`~/dev/mflux` の
> `docs/16gb/measurements/2026-09-22-m9-ceiling-probe.md`。**実測**、別プロジェクト）。
> 上の 4.0 は M3 Pro の MPS eager 値で silicon ピークではないので割り引く必要はあるが、
> それでも **ANE 単基 7.3〜8.7 は M6 GPU の 40% 前後、2 基でも 85%** にしかならない。
> 「数字だけ見れば魅力的」は M6 では成立しない。§3-5 の判定（GPU tensor ops が先）は強化される。

### 3-3. それでもこの設計には直接入らない（理由 3 つ）

1. **重みが焼き込みになる。** Core ML は重みを mlpackage に埋める。
   14.3 GB の expert を毎トークン差し替える設計と根本的に噛み合わない。
   入力で渡す手は実測済みで、**KV 47 MB を入力に渡すだけで 36.8 → 58.6 ms**（Irodori 13 §2-2）。
   1 層 476 MB では論外。
2. **ANE は fp16。** int4 の重みを fp16 に展開して渡すことになり、
   「2 GB で 26B を回す」という設計の一番の売りが消える。
3. **軸長の上限**（Irodori 15 §4-1 の実測で境界は 15360〜21600 の間。既知の 16384 と整合、上限値そのものは未確認）と、
   **fp16 オーバーフロー**（残差 |h| が block 11 で 2300 に達し、`x*x` が 65504 を超える。
   norm 内部で `x/64` に事前スケールして回避）はそのまま持ち込まれる。

一方、Irodori で最大の運用制約だった「`predict()` が GIL を握るので ANE は別プロセス」は
**Swift では効かない**。ここだけは楽になる。

### 3-4. 乗る候補（いずれも「常駐している dense 部分」）

| 候補 | 理由 | 優先 |
| --- | --- | --- |
| **vision encoder** | 純粋な dense 演算ブロック、重みも固定。ANE と tensor ops の両方に最も素直に乗る | 高 |
| **MTP ドラフトヘッド** | 小さく固定。デュアル ANE なら本体と同時に回せる余地がある | ~~中~~ **取り下げ（§3-6）** |
| 共有 core（1.35 GB）の射影 / 共有 MLP / 出力ヘッド | 固定重み。**ANE に出して GPU を expert GEMM に専念させる**ヘテロ分割 | 中（§2 の後） |

ただし §2-1 のとおり、射影 + 共有 MLP はタイル化後で合計 2.3 s しかない。
**ANE に出して浮くのは最大 2.3 s、routed MoE の 8.59 s には届かない。**
ヘテロ分割が成立するのは「ANE と GPU を実際に重ねられたとき」だけで、
Irodori 15 §2 と同じく**重ねられる形かどうかを先に構造で確かめる**必要がある
（prefill はチャンク単位なので、チャンク N の MoE と N+1 の射影を重ねる形なら成立しうる。**未確認**）。

> **2026-09-22 追記: 重なりの形は構成できる（依存関係まで詰めた版）。**
> **チャンク N+1 を、チャンク N より 1 層だけ遅らせて流す。**チャンク N+1 の層 L の attention が要るのは
> 「チャンク N の層 L の KV」で、それは N が層 L を終えた時点で揃っている。だから GPU がチャンク N の
> 層 L+1 の routed MoE を回している間、ANE はチャンク N+1 の層 L の射影 / 共有 MLP を回せる。
> 活性を 2 チャンク分持つだけ（2048 × 2816 × fp16 ≈ 11.5 MB/本）なのでメモリでは詰まらない。
> **decode には入れない**（§3-6 の呼び出し粒度の話がそのまま当たる）。prefill だけが ANE と形が合う。
>
> ただし **M6 では代金が上がった。**ANE は fp16 なので共有 core の複製を mlpackage 側に持つことになり
> （§3-6）、16 GB 機ではそれが expert キャッシュのスロットから引かれる。そのミス 1 回を
> **M6 は 3.3 GB/s で払う**（M3 Pro の 6.7 の半分、[M6_SSD_BANDWIDTH](M6_SSD_BANDWIDTH.md)）。
> つまり §3-6 が「未確認」とした「GPU 側 1.15 GB 常駐を ANE 側に付け替えて wired を相殺できるか」は、
> **相殺できなければ不採用**という判定基準になる。回避策は Core ML の blockwise 量子化（macOS 15+）で
> int4 のまま持たせることだが**未確認**。
>
> 取り分の勘定（**導出**）: M6-2 / M6-4 が予測どおり通ると prefill GPU は 13.44 → 約 7 s になり、
> ANE に出せるのは射影 + 共有 MLP の 0.8 s = **完全に重なっても 11%**。
> 一方 vision tower（M6-5）は TTFT 4.16 s のうち 1.4 s = **34%**。優先順位は変わらない。

### 3-5. 判定

**ANE のデュアル化は、この設計にとって GPU の Neural Accelerator ほどのリターンはない。**
投資順は GPU tensor ops が先。ANE は vision に限った第二弾とする（MTP は §3-6 で取り下げ）。

### 3-6. 補遺（2026-09-01）: M3 Pro のままでの ANE の余地

§3-4 の候補を「M6 を待たずに、いまの M3 Pro で成立するか」で引き直した。
結論: **vision tower の 1 点だけ成立する。MTP ドラフターは候補から取り下げる。メモリ節約は逆効果。**

**vision tower — M6 依存ゼロで成立する（優先: 高のまま）**

- tower は画像 1 枚 3.5 TFLOP（導出、`PLAN_VISION.md` §3-2）で、**実測 1.9 s**。
  画像あり TTFT 4.16 s のほぼ半分を占める（`RESULTS_VISION.md` §4、S=280 / `trusted-install`）。
- ANE 16 コア単基の fp16 実効 7.3〜8.7 TFLOPS（§3-2、同じ M3 Pro の実測）が出れば
  **1.9 → 0.5 s 前後、TTFT 4.16 → 2.8 s 程度**（導出）。
- §3-3 の三大制約がすべて外れる唯一の大物である:
  重みは 1.15 GB の**固定**（焼き込み可）、もともと fp16 系（int4 の売りを侵さない）、
  Swift なので GIL も効かない。持ち込まれるのは軸長上限と fp16 オーバーフロー（§3-3-3）のみで、
  S=280 の系列長なら上限には遠い。
- Core ML 経路は macOS 15 で動くので **M6-5 は M6 を待つ理由がない**。
  検証は `TsugumiKernelCheck --vision-tower` の fixture（max 8e-2 / rms 2e-3）がそのまま使える。

**MTP ドラフター — 取り下げ（M3 Pro でも M6 でも効かない）**

1. **小さすぎる。** ドラフター 1 ステップは壁時計 **2.86〜4.64 ms**（実測、
   `docs/mtp/14-M3.5-RESULTS.md`）。ブロックの残りは moe 27 + ホスト往復 14 + post 5 ms
   （`RESULTS_MTP.md` §7）なので、**タダにしても縮むのは 1 割弱**。
2. **呼び出し粒度が ANE に最悪。** Irodori 13 §2-1 の実測は 180 トークンまとめて 1 呼び出し
   10.3 ms、B=1 の実ラッパーで 21 ms。ドラフターは **seq=1 の呼び出しを毎ステップ**やる形で、
   演算ほぼゼロのまま固定ディスパッチコストだけ払う。GPU の 3 ms より速くなる見込みがない。
   tower（1 呼び出し 1.9 s でオーバーヘッドを完全償却）と正反対の形である。
3. **直列依存で並走の形が作れない。** 次のドラフトは verify で確定した受理トークンに依存する。
   単基 ANE では GPU が遊ぶだけで、§3-4 の「デュアルなら同時に回せる余地」も未確認の余地止まり。
4. **本当のボトルネックは受理長。** 日本語散文 a=1.06（M3.5 以来未着手、`RESULTS_MTP.md` §7）は
   ドラフターの品質の問題で、ハードでは解けない。しかも MTP は既定オフ。

**メモリ節約 — 期待できない**

ANE は fp16 なので int4 を展開して渡すことになり、メモリはむしろ増える方向にしか働かない。
tower の ANE 化も mlpackage 側に fp16 の複製を持つぶんディスクとロードは微増する
（GPU 側 1.15 GB 常駐を ANE 側に付け替えて wired を相殺できるかは**未確認**）。

---

## 4. CPU（super core）

super 2 + P 4 = **大コア 6 本で M3 Pro の P 6 と同数**、1 本あたりが速い（公称）。
Apple 自身が「コード補完・ファイル索引・エージェント的ワークフローの一部は 1 コア速度で決まる」と説明している。

効く場所:

| 場所 | 根拠 |
| --- | --- |
| **decode のホスト側オーバーヘッド** | M2 の診断（`docs/BENCHMARKS.md`）で 1 step 162.8 ms 中「コマンドバッファ待ち 55.6 ms」「その他ランタイム 9.9 ms」。ここはシングルスレッド律速 |
| **repack / インストール** | 15 GB のストリーミングと SHA-256 検証。`full-sha256` を諦めて `trusted-install` にした判断（`RESULTS.md` §0 で TTFT −2.95 s）は、super core なら再検討の余地がある（**未確認**） |
| トークナイザ / grammar / ツールループ | エージェント経路の JSON 処理 |

decode 31 tok/s の内訳のうち、GPU でも I/O でもない部分がここに当たる。**倍率は未確認。**

---

## 5. メモリ・帯域・SSD

### 5-1. DRAM 帯域: 16 GB 構成では +2%、つまり無い

**16 GB 構成は 153 GB/s**（Apple 技術仕様）。170 GB/s は 24/32 GB を選んだときだけの値である。
手元の M3 Pro が 150 GB/s なので、**16/256 を買った場合の帯域向上は +2%**。

したがって decode 31 tok/s（実測、64 スロット / `trusted-install`）は
**帯域由来ではほぼ 1 tok/s も動かない**（導出）。24 GB を選んだ場合でも 170 GB/s = +13% で、
かつ decode は純粋な帯域律速ではない（expert I/O + 出力ヘッド + コマンドバッファ待ちの混合、
`docs/BENCHMARKS.md` の M2 内訳）ので、上限でも +13% には届かない。

上流が 24 GB M5 Pro（153 GB/s）/ macOS 26.5.1 で取った decode 34.7〜35.2 tok/s（`docs/BENCHMARKS.md`）は
**帯域が M3 Pro とほぼ同じ機械での値**であり、その差 3〜4 tok/s は帯域ではなく世代・OS・GPU 側の差である（導出）。
**16 GB M6 の decode の見込みは、この 34〜35 tok/s 近辺が上限**と見るのが妥当。

この 1 点だけで、**M6 に移る価値は prefill 側にしか無い**ことがほぼ確定する。

### 5-2. SSD「2 倍」は本命になりうる（が 256 GB 構成が未確認）

現状 decode の expert I/O は **12.5 ms/token**（Mac アプリ実測、`tsugumi-m3pro-status.md` §2-1）。
M3 Pro の SSD 実力（`F_NOCACHE` でページキャッシュを迂回、実測）:

| パターン | スループット |
| --- | ---: |
| 連続読み（1 層 476 MB） | 4.74 GB/s |
| ランダム 3.72 MB pread（30 層またぎ） | **4.19 GB/s**（0.89 ms/expert） |

M6 mini がここを超えるなら、**このリポジトリの decode が最も直接に得をする**。
ただし **256 GB 構成は NAND ダイ数が減って遅い前科がある**:
M2 mini のベースモデルは 30〜50% 低下、M4 mini は 2×128 GB 構成にして解消した。
**M6 の 256 GB がどちらかは未確認。買ったら最初にやるのは §7-2 の実測。**

なお `PREFILL_THROUGHPUT.md` §0 の結論「SSD は律速ではない（chunk 2048 なら expert I/O 23.9 → 1.85 s）」は
**prefill の話**である。decode 側は依然 12.5 ms/token を払っている。混同しない。

### 5-3. 16 GB の本当のコスト — 比較ベースラインが動かせなくなる

本体経路は困らない。**chunk 2048 なら 16 スロット（peak 3.1 GB）と 80 スロット（peak 10.3 GB）が同速**と実測済みで
（`PREFILL_THROUGHPUT.md` §0）、Gemma の運用点は peak 2〜7 GB に収まる。
「モデルを RAM に置かない」という設計そのものが 16 GB と整合している。

困るのは横の 3 つ:

1. **同一ホスト比較ができなくなる。** `docs/BENCHMARKS.md` の MLX 比較は
   peak GPU alloc 14.66〜15.31 GB / peak RSS 8.27〜9.79 GB。llama.cpp は 15 GB。
   **16 GB 機ではどちらもまともに回らない。** 「半分以下のメモリで同圏」という主張の根拠を
   同じ機械で取り直せなくなる。→ 比較は M3 Pro に残す運用にする。
2. **Qwen3.5-MoE の参照器。** 既存ランタイムは 18 GB に載らないと記録済みなので、16 GB でも当然載らない。
   もっとも対策は層ストリーミングで自前に書く方針が既に決まっているので、**16 GB でも方針は変わらない**。
3. **48 スロット運用の wired limit。** 18 GB でも `sudo sysctl` で `iogpu.wired_limit_mb` を上げる必要があった
   （再起動で 8192 に戻る）。16 GB ではさらに窮屈。ただし §5-3 の 1 行目のとおり
   スロット数は chunk 2048 では効かないので、**既定を下げる方向で辻褄が合う。**

---

## 6. 構成と値段

選べる構成（公称、日本価格）:

| 構成 | 帯域 | 価格 | 差額 |
| --- | ---: | ---: | ---: |
| 16 GB / 256 GB | 153 GB/s | ¥149,800 | — |
| 16 GB / 512 GB | 153 GB/s | ¥185,800 | +¥36,000 |
| 24 GB / 512 GB | **170 GB/s** | ¥221,800 | **+¥72,000** |

**24 GB は 512 GB との抱き合わせなので、実質 +¥72,000 になる。**
そして 24 GB を選んだときだけ帯域が 153 → 170 GB/s に上がる（§5-1）。
つまり ¥72,000 で買えるのは「RAM +8 GB」「SSD +256 GB」「帯域 +11%」の 3 点セットである。

判定:

- **16 GB / 256 GB（素）が第一候補。** モデル 15 GB + macOS で 256 GB は足りる。
  RAM は §5-3 のとおり本体経路では問題にならない。
- **16 GB / 512 GB（+¥36,000）は、§7-2 の実測で 256 GB の SSD が遅いと判明した場合の保険。**
  買う前に判定できないのが難点だが、この設計では SSD 帯域が decode に直結するので、
  ¥36,000 の中では最も筋のよい追加投資である。
- **24 GB（+¥72,000）は現時点で理由がない。** 24 GB でも MLX / llama.cpp の 15 GB ベースラインは
  快適には回らないし、Qwen3.5-MoE は 24 GB でも載らない。比較ベースラインは M3 Pro に残す方が安い。
  抱き合わせで付く帯域 +11%（153 → 170 GB/s）も、§5-1 のとおり decode では上限でも +11%、
  実際にはそれ以下にしかならない。**¥72,000 の対価としては薄い。**
- **M5 Pro（TB5、64 GB まで）はこの設計には不要。** mini クラスタは本リポジトリの射程外。

---

## 7. 実機で最初に取る 5 つの数字

**この 5 つが揃うまで「M6 で N 倍」と書かない。** 特に 3 と 4 は土俵の絶対値そのものである。

| # | 測るもの | 方法 | 効く判断 |
| --- | --- | --- | --- |
| 1 | ~~`supportsFamily(.apple10)` が M6 で true か~~ **済 (2026-09-22): apple10 / apple11 / metal4 すべて true**（M3 Pro は apple9 止まり・metal4 false） | `MTLDevice.supportsFamily` を直接叩いた | §2-3-2。**ゲートは通る**（§7-1a も見ること） |
| 2 | ~~`F_NOCACHE` での SSD 帯域（256 GB 構成）~~ **済 (2026-09-22): 3.34 GB/s = M3 Pro 1TB の半分** | [M6_SSD_BANDWIDTH.md](M6_SSD_BANDWIDTH.md) | §5-2 / §6。512 GB を買うべきかの唯一の根拠 → **買う意味はある (同 §3)** |
| 3 | ~~M6 12 コア GPU の素の fp16 ピーク~~ **済 (2026-09-22): 約 19 TFLOPS** | mflux M9 の天井プローブ（MLX 19.2 / MPS 18.7〜19.0 の 2 系統） | §2-4 の誤差の主因。**MPS（Apple 自身のカーネル、neural accelerator 込み）でも 19 で頭打ちなので、silicon に隠れた性能は無い。**M6-4 の合格条件「実効 2.0 TFLOP/s 以上」はこの天井の 10% という意味になる |
| 4 | **tensor ops を使わないまま移植したときの pp** | 現行コードを macOS 27 でそのままビルド、2478 トークン | §2-2-1。「12 コアで遅くなる」かどうか。**比の分母** |
| 5 | macOS 27 で deployment target の引き下げ（`103bfbc`）を戻せるか | ビルドのみ | 上流想定（macOS 26 / Metal 4）に戻れる = 上流ベンチと直接比較できる |

### 7-1a. #1 の続き: M6 で tensor 経路が動かない理由は機械ではなくモデルだった（2026-09-22、**実測**）

`supportsFamily(.apple10)` は true、`metal4` も true。そのうえで M6 の実機で確かめた:

- **`tensorops.metal` は MSL 4.0 でコンパイルが通り、`mpp_prefill_affine_threadgroup_f16` の
  PSO も生成できる**（maxTotalThreadsPerThreadgroup 1024）。単体プローブで確認。
- にもかかわらず `MPPPrefillInt4QMM` は本番で nil になる。理由は
  **`init` の `guard context.affineGroupSize == 64`** で、
  ベンチで使っている `gemma4-qat-sym.gturbo` の attention は
  **`groupSize 32 / scheme sym`**（manifest）だからである。MPP が使われるのは
  `tokenCount >= 32` の Q / KV / O 射影だけ（`RealForwardRunner` 1308 行目）なので、
  **この QAT モデルでは経路そのものが構造的に閉じている。機械の問題ではない。**
- つまり **M6 で tensor 経路を開ける鍵は M6-0 ではなく M6-1**（MPP の group 32/64 両対応）である。
  M6-0（静的ゲートをパイプライン生成の成否に置換）は依然やるべきだが、
  **これ単独では何も変わらない**。

> 紛らわしい点: `TsugumiKernelCheck` の `prefill-qmm path=simdgroup-matrix` は MPP とは無関係である。
> `PrefillInt4QMM.Path` は `scalar-block` と `simdgroup-matrix` の 2 つしか持たず、
> **tensor 経路の枝を最初から持っていない**。あの行を見て「tensor が落ちた」と読んではいけない。

4 が特に重要である。**M6 の利得を「tensor ops 由来」と「世代由来」に分けられる唯一の測定点**であり、
これを取らずに 231 → N tok/s を比較すると、両者が混ざって後から説明できなくなる。

---

## 8. 作業の優先順

前提: §7 の 1〜5 が終わっていること。

| Gate | 内容 | 予測 | 合否 |
| --- | --- | --- | --- |
| **M6-0** | `supportsFamily(.apple10)` ゲートを「パイプライン生成の成否」に置換 | 挙動不変（M3 Pro）、M6 で tensor 経路が選ばれる | M6 で tensor attention が実際にディスパッチされるログ |
| **M6-1** | `MPPPrefillInt4QMM` を group 32/64 両対応に（simdgroup 版と同じ索き方） | QAT チェックポイントで MPP 経路が生きる | group-32 で MPP がディスパッチされ、CPU 参照と一致 |
| **M6-2** | **routed MoE の expert GEMM を `matmul2d` 化**（最大の的） | 8.59 s → 2.0 ± 0.8 s | GPU 合計 13.44 → 7 s 以下、greedy 出力一致 |
| **M6-3** | prefill attention の tensor 経路を A/B | 2.02 s → 0.7 ± 0.3 s | `.fullTensorOps2DValidityV2` vs `.causalQBlock` の実測差 |
| **M6-4** | 射影 / 共有 MLP を `matmul2d` 化 | 1.58 + 0.74 = 2.3 s → 0.8 ± 0.3 s | 実効 2.0 TFLOP/s 以上 |
| **M6-5**（第二弾） | vision encoder の Core ML / ANE 化 | tower 1.9 → 0.5 s 前後（§3-6、導出） | §3-4。**M6 を待たず M3 Pro で着手できる**（§3-6）。GPU と重なる形かを構造で先に確認 |

> **2026-09-22 追記: A/B の取り方に罠が 2 つある（`~/dev/mflux` の同日の実測より）。**
>
> 1. **単体ベンチと本番で符号が逆転する。**mflux の M8c は `mlp.down` の K 分割が
>    単体で **−30 ms**、本番のストリーミングループで **+42 ms** だった
>    （`docs/16gb/measurements/2026-09-22-m8c-m8d-kslices-cache-limit.md`）。
>    **M6-2 / M6-3 / M6-4 の判定は必ず本番のループで取る。**
> 2. **主スレッドが I/O で離れると GPU が餓える。**M9c の正体はこれで、GPU に投げた後に
>    主スレッドが 461 MB の読みで 115〜140 ms 離れると、ops の多いカーネルほど大きく食らった
>    （100 ms 眠らせて +47 ms、250 ms で +196 ms）。
>    Tsugumi の `executeExpertCachePlan` は `concurrentPerform` でミスを読む間、
>    コマンドバッファを積む人がいない。[33 §5](../mtp/33-M8-IO-FLOOR.md) の
>    「`fetch` が減ったぶん `wait.front` が増える」はこれと地続きで、
>    **M6 は SSD が半分なので離脱時間が倍**になり、罠が M3 Pro より深い。

**やらないこと:**

- **decode 側の tensor ops 化。** §2-2 のとおり効かない。
- **expert キャッシュのスロット数チューニング。** chunk 2048 で無意味と実測済み。16 GB ではむしろ下げる。
- **M6 で llama.cpp / MLX の同一ホスト比較を取り直すこと。** §5-3-1。16 GB では成立しない。
- **`.fullTensorOps2DValidityV2` の greedy 一致を取らずに採用すること。**
  `PREFILL_THROUGHPUT.md` §7-7-4 の時点で「CLI に attention パス切替フラグがなく A/B が取れない」が
  未解決のまま残っている。M6-3 の前にフックを足す。

---

## 9. 未確認リスト

1. ~~M6 が返す `MTLGPUFamily`~~ **解決 (2026-09-22)**: apple4〜apple11 true、apple12 false、metal4 true（§7 #1）
2. 256 GB 構成の SSD 実速度、および「2 倍」の比較対象（M4 mini の何 GB 構成か）
2b. 16 GB 構成が 153 GB/s に留まる理由（LPDDR パッケージ数か、ビニングか）。実効帯域が公称どおりか
3. ~~M6 12 コア GPU の fp16 の絶対性能~~ **解決 (2026-09-22): 約 19 TFLOPS**（§7 #3）。`matmul2d` 経由で何割取れるかは未確認
4. デュアル ANE を 2 基同時に使うための条件（1 モデルで自動分割されるのか、2 モデル並走時のみか）。
   **注意 (2026-09-22)**: mflux が Metal キャプチャで確認した `affine_qmm_t_nax_*` は **GPU 内の
   neural accelerator**（Metal 4 tensor ops）であって **ANE ではない**。名前が紛らわしいが、
   あれは #4 の証拠にならない。#4 を決めるのは「Core ML モデルを 2 本同時に走らせて合計が 2 倍になるか」
5. ANE の軸長上限の正確な値（実測では 15360〜21600 の間）
6. macOS 27 で `LSMinimumSystemVersion` 由来の Xcode 26.x GUI 制約が解けること（ほぼ確実だが未確認）
7. Apple の「LM Studio で M4 比 4.8 倍」の測定条件（モデル、量子化、プロンプト長）

---

## 出典

公称値・論文:

- [Apple introduces M6 and M5 Ultra](https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/)
- [Apple unveils a more powerful Mac mini featuring the all-new M6 and M5 Pro](https://www.apple.com/newsroom/2026/08/apple-unveils-a-more-powerful-mac-mini-featuring-the-all-new-m6-and-m5-pro/)
- [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU — Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators (arXiv 2607.19438)](https://arxiv.org/html/2607.19438v1)
- [Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU (arXiv 2606.12765)](https://arxiv.org/html/2606.12765v1)
- [Optimize custom machine learning operations with Metal tensors — WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/)
- [Apple M6 — Wikipedia](https://en.wikipedia.org/wiki/Apple_M6)
- [Mac mini — 技術仕様（Apple）](https://www.apple.com/mac-mini/specs/) — 16 GB = 153 GB/s、24/32 GB = 170 GB/s の一次資料
- [M6 Mac mini: Three things Apple didn't highlight — 9to5Mac](https://9to5mac.com/2026/08/27/m6-mac-mini-three-things-apple-didnt-highlight-in-the-announcement/)
- [Updated M6 Mac mini arrives in RAM and SSD constrained environment — AppleInsider](https://appleinsider.com/articles/26/08/25/m6-mac-mini-arrives-in-ram-and-ssd-constrained-environment)
- [M4 搭載 Mac mini は 256GB モデルでも SSD が遅くない](https://softantenna.com/blog/m4-mac-mini-ssd/)

手元の実測:

- `tsugumi-m3pro-status.md` — M3 Pro の実機構成と Mac アプリ計装
- `docs/investigations/PREFILL_THROUGHPUT.md` — GPU 内訳、SSD プローブ、MPP の group-64 制限
- `docs/BENCHMARKS.md` — M5 Pro decode、MLX 同一ホスト比較、M2 の decode 内訳
- `RESULTS.md` — 48/64 スロットと `trusted-install` の効果
- `~/dev/Irodori-TTS/docs/experiments/13-ane.md`, `15-decode-ane.md` — 同じ M3 Pro での ANE 実測

---

## 10. 補遺 (2026-09-13): その後の実測で古くなった箇所

1. **§5-2「decode の expert I/O 12.5 ms/token は SSD 帯域」は Qwen 経路の実測で覆った。**
   [27-PHASE6-THROUGHPUT §9](../qwen35moe/27-PHASE6-THROUGHPUT.md) で、decode の io 15.75 ms/tok の
   85% は residency set の保守、その 94% が `MTLResidencySet.commit()` で、disk0 は decode 中 0.6〜0.85 GB/s
   しか動いていない (バイトは既にページキャッシュにある)。**費用はマッピングで、SSD ではない。**
   したがって公称「ストレージ 2 倍」が decode に効く見込みは本書の期待より小さい。
   効くのは prefill の初回読み込み (mmap の腕で `F_RDADVISE` 後 11.99 GB/s、[mtp/52](../mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md)) の側。
   §7-2 の SSD 実測は依然として取るが、判断が変わるのは 512 GB を買うかどうかではなく prefill の上限。
2. **§2-3「tensor 経路は macOS 27 なら自動で生き返る」は Gemma 経路だけの話。**
   Qwen 経路 (`QwenPrefill.swift` / `QwenForwardRunner.swift`) は `MPPPrefillInt4QMM` も
   `.fullTensorOps2DValidityV2` も参照していない。M6-2 (routed MoE の `matmul2d` 化) は Qwen 側では
   新規に書く。[43 §3](../qwen35moe/43-LILY-IDEAS.md)。
3. **§5-3-2「Qwen3.5-MoE の参照器は載らない、層ストリーミングで自前に書く」は実行済み。**
   Qwen 経路は M3 Pro 18 GB で decode 20 tok/s / prefill 265 tok/s (32 スロット、mmap、chunk 2048、
   [27 §2](../qwen35moe/27-PHASE6-THROUGHPUT.md))。16 GB で困るのは本書の見立てどおり比較ベースライン側のみ。
4. §7 の 5 つの実測と §8 の M6-0 / M6-1 は有効のまま。**M6-0 (apple10 の family ゲート → パイプライン生成の成否) は
   Qwen 側で tensor 経路を書く前に済ませる。**
5. **Qwen3.8-Flash-Next (177B) を理由に 24 / 32 GB へ変更するかは [M6_QWEN38_FLASH_NEXT.md](M6_QWEN38_FLASH_NEXT.md) (2026-09-13) を見る。**
   Q4_K_XL では 24 GB 9〜11、32 GB 10〜13 tok/s (導出) で検討ライン 15 に届かない。IQ3_XXS 前提なら 32 GB で短文脈 15〜18 に乗る
   (24 GB は 14〜17)。全常駐でも 170 GB/s の天井は約 25。買うなら 32 GB だが、IQ3_XXS の品質判定 (PC / サーバーで可能) が先。
