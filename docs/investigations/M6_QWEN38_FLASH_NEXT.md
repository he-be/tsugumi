# 調査: M6 Mac mini の 24 GB / 32 GB 構成で Qwen3.8-Flash-Next は 15 tok/s を超えるか

作成: 2026-09-13 (同日改訂: 量子化を Q4_K_XL → **IQ3_XXS** 前提に、PLE の SSD 読みは第 1 層に隠れる前提に)
きっかけ: M6 Mac mini 16 GB / 256 GB (2026-09-22 着) を予約した後に Qwen3.8-Flash-Next がオープンウェイトになった。
予約を 24 GB または 32 GB に変えるべきか、変えた場合に何 tok/s 出るか (検討ライン **15 tok/s 以上**)。
機種側の前提は [M6_MAC_MINI_TARGETING.md](M6_MAC_MINI_TARGETING.md)、Qwen 経路から見た M6 は [qwen35moe/43-LILY-IDEAS.md](../qwen35moe/43-LILY-IDEAS.md)。

表記は他の調査文書と同じ:

- **実測** = 手元で数字を取ったもの。条件を併記する。
- **導出** = 実測または公称値から計算したもの。誤差要因を併記する。
- **公称** = Apple / 論文 / 報道が言っているだけで、手元では取っていないもの。
- **未確認** = 根拠を持っていないもの。断定しない。

**Mac 上で Flash-Next を動かした実測は 1 つもない。** 本書の実測はすべて Windows PC (RTX 5080) と llm-server (V100 + 3090) のもので、
Mac 側の数字は全部導出である。

---

## 0. 結論を 3 行で

1. **IQ3_XXS なら 32 GB は短文脈で 15〜18 tok/s (導出) に乗る。24 GB は 14〜17 で線をまたぐ。** 買うなら 32 GB (§3、§4)。
   ただし PC の評価指標である 128k 埋め切り平均では両方とも 11〜13 に落ち、15 を保つには SSD 先読みが計算と重なる必要がある (未確認)。
2. **天井は容量でなく帯域。** 全部 RAM に載っても M6 (170 GB/s 公称) では 1 トークン約 4 GB の読み出しで **約 25 tok/s** が上限 (§3-1)。
   24 GB と 32 GB の差は導出で 1〜2 tok/s しかなく、大きいのは SSD 先読みの重なりと dense の bpw の方 (§3-1)。
3. **前提が 3 つ揃わないと数字は出ない**: (a) Tsugumi の `qwen4exp` 移植 (IQ2_S / IQ4_NL の Metal dequant を含む)、
   (b) IQ3_XXS の品質が実タスクで許容できること (expert gate/up 2.5 bpw、`PLAN-quant-eval-pi.md` の比較は**未実施**)、
   (c) M6 の SSD が 2 MB ランダム読みで 5〜8 GB/s 出ること。**(b) は Mac と無関係に PC / サーバーで先に決められる。**

---

## 1. 材料 (すべて PC / llm-server の実測。原本は `192.168.0.199` の `C:\LLM`)

### 1-1. モデル (GGUF メタデータ、`HANDOVER-qfn-engine.md` §2 と本日の読み出し)

| 項目 | 値 |
| --- | --- |
| アーキテクチャ | `qwen4exp`。総 176.94B = 本体 125B (active 6B + shared 1B) + n-gram 埋め込み 51B + MTP 4B |
| 層 | 48 (full attention 12 + Gated DeltaNet 36、indexer 4 head × 128 の疎注意付き) |
| hidden | 2560、per-layer input embedding 160 |
| MoE | **512 expert、top-10**、expert ff 640、shared expert ff 640 |
| PLE (n-gram 埋め込み) | **`ple.layers = [1]` (第 2 層に 1 か所)**、`ngram_size 3` × `heads_per_ngram 8` = 16 head、各 head 約 2000 万行 × 160 次元。1 トークンに読むのは 16 行 (IQ4_NL で約 90 B ずつ、計約 1.4 KB)。キーは直前 3 トークンのハッシュ (`layer_multipliers` 3 本) |
| MTP ヘッド | blk.48 の expert 3 テンソル、Q8_0 2.6 GiB / expsQ4_0 1.46 GiB |

### 1-2. 量子化ごとのサイズ (llm-server 実測、`PLAN-article-and-3rd-gpu.md` §2-1)

| | UD-IQ3_XXS (**本書の前提**) | UD-Q4_K_XL (参考) |
| --- | --- | --- |
| 全体 | 76.3 GiB (82 GB) | 104 GB (111 GB) |
| expert 全体 | **45.27 GiB (48.6 GB)**、962 MiB/層 (blk2 のみ 1138) | 70.5 GB、1.47 GB/層 |
| expert 1 個 | **1.97 MB** (gate/up IQ2_S 2.5 bpw、down IQ4_NL) | 2.87 MB (gate/up Q4_K、down Q5_1) |
| dense | 3.24 GiB (Q6_K / Q8_0 混在)、うち `token_embd` 486 MiB は行 gather のみ | 約 3.5〜4.8 GiB (Q8_0) |
| PLE 表 | 26.82 GiB (IQ4_NL) | 同じ |
| 品質 (Q4_K_XL 基準) | 推定 KLD 0.2〜0.3、top-1 82〜86% (**要実測**) | 基準 |

### 1-3. 1 トークンに読むバイト (導出、IQ3_XXS)

| 区分 | バイト | 根拠 |
| --- | ---: | --- |
| routed expert | **0.945 GB** (10 × 48 × 1.97 MB) | §1-2 |
| dense (attention / GDN / shared / 出力ヘッド) | **約 3.0 GB** (3.24 GiB − token_embd 486 MiB) | §1-2。`output.weight` は毎トークン全読み (CPU に出すと +12 ms、llm-server 実測) |
| PLE | 1.4 KB (16 回のランダム読み) | §1-1 |
| KV (q8_0、128k) | 全量 2.2 GiB だが indexer の疎注意で decode は top-k 2048 セル | `RESULTS-pc-qsa-gather.md` |

**合計約 3.95 GB/token** (Q4_K_XL なら約 5.0 GB)。dense が 4 分の 3 を占める。これが §3-1 の天井を決める。

### 1-4. 到達済みの速度 (実測、thinking)

| 機材 | 構成 | tok/s |
| --- | --- | ---: |
| llm-server: V100 32 GB + RTX 3090 24 GB | **IQ3_XXS、CPU 0 層、96k** | 素 **50.2** (19.9 ms)、MTP n_max 2 で 61.8、32k で 50.9、96k で 30.4 |
| 同 | Q4_K_XL、CPU 20 層 | 28.5 (短文脈) |
| PC: RTX 5080 16 GB + DDR5-5600 128 GB | Q4_K_XL、MTP n_max 1、gather パッチ | 128k 埋め切り平均 **23.13** (短文脈 24.5) |

llm-server の IQ3_XXS 19.9 ms は「全 GPU 常駐、dense Q6_K 混在」の床で、そこから 1 トークン約 4 GB を 2 枚合計の帯域で読んでいる。
Mac の天井 (§3-1) はこの床を 170 GB/s に引き直したものになる。

### 1-5. expert ルーティングの局所性 (`RESULTS-e2.md`、thinking 4 プロンプト × 2000 tok のトレース)

- 層あたり 2000 トークンで **450〜495 種類**の expert が使われる (ほぼ平坦)。静的 hot set は 128/層でも被覆 45%。
- 直前 k トークン以内に同じ expert が出る割合: k=1 33%、4 53%、16 74%、32 83%、64 89%。
- 動的キャッシュ (LRU + 直近 8 tok で 2 回以上要求されたら admit) の hit:

| スロット/層 | 全 expert 比 | hit |
| ---: | ---: | ---: |
| 32 | 6.25% | 0.55 |
| 48 | 9.4% | 0.64 |
| 64 | 12.5% | 0.71 |

hit はスロット**数**で決まりバイト数に依らないので、IQ3_XXS では同じ RAM で 1.46 倍のスロットが取れる。
§3 の hit は、この 3 点を `hit ≈ 1 − 0.45 × (32/s)^0.6` で外挿した (64 で 0.70、実測 0.71)。64 より上は裏付けが無く、保守的に出るはず (**未確認**)。

### 1-6. MTP は Mac では効かない (導出)

連続 2 トークンの expert 和集合は 1.665 倍 (E2 §1)。受理率 0.7 で 1.7 tok/step なのでトークンあたりのバイトは素と同じ。
llm-server で +23% (50 → 62) 出たのは GPU 側オーバーヘッドの償却分で、SSD ミスが律速の状況では利得が無い。全常駐 (M5 Pro 64 GB) なら効く。

---

## 2. Mac で動かす形 (前提)

> **補足 (同日夜)**: 本ランタイムは affine int4/int8 (g32/g64) しか持たないので、「IQ3_XXS 前提」は実装上
> **2-bit または 3-bit affine の expert カーネルを新規に書く**ことを意味する (oQ4e をそのまま使うと Q4_K_XL 列に戻る)。
> ウェイトの供給源・移植差分・M6 到着前に決着させる項目は [QWEN38_FLASH_NEXT_VERIFY_PLAN.md](QWEN38_FLASH_NEXT_VERIFY_PLAN.md)。
> 既製の Metal ランタイム `ds4-metal` (SSD ストリーミングあり) が Flash-Next を動かすので、M3 Pro での実測はそれで先に取れる。

- **既製ツールでは成立しない (llama.cpp / MLX の場合)。** llama.cpp (Metal) や MLX で 82 GB のモデルを 24〜32 GB に mmap すると、ページキャッシュが
  カーネルの LRU (admission 無し) でスラッシングする。1〜3 tok/s 級と見る (**未確認**)。
- §3 の数字は **Tsugumi (turbo-fieldfare) を `qwen4exp` に移植した後**の値である。移植の差分:
  Gated DeltaNet 36 + full 12 は Qwen3.5 (30 + 10) の兄弟だが、indexer の疎注意、512 expert top-10、PLE の n-gram ハッシュと非同期 pread、
  **IQ2_S / IQ4_NL / Q6_K の Metal dequant** (本線は affine Q4 g64 のみ) が新規。
- **PLE は費用ゼロで RAM も使わない。** キーが直前 3 トークンで決まるので、トークンが確定した時点で 16 行の pread を投げれば
  第 1 層 (attention / GDN + MoE、1 トークン約 40 ms の 48 分の 1 ≈ 0.8 ms) の計算に隠れる。NVMe のランダム読み 16 本並列は 0.2 ms 級 (**未確認**)。
  prefill でも 2048 × 16 = 32,768 本/チャンクがチャンクの計算 (十数秒) に隠れる。熱い行はページキャッシュに自然に残る。
  **mmap のページフォルトに乗せると第 2 層で 1 回ずつ待つことになる**ので、ここだけは pread で先行発行する。
- RAM の固定費 (導出): macOS + アプリ 3 GB、dense 3.0 GB、KV (128k q8_0) 2.2 + compute 0.8 GB、その他 0.5 GB = **9.5 GB**。
  残りを expert のページキャッシュに充てる。

---

## 3. 試算 (導出。誤差は大きい)

条件: thinking、短文脈、UD-IQ3_XXS、自前エンジン。1 トークン = DRAM 読み + SSD ミス + ホスト/residency、**重ならないものとして足す** (下段に重なった場合の上限)。

- DRAM: (3.0 + 0.945 × hit) GB ÷ 実効帯域。実効は公称の 75% (M6 170 → 130 GB/s、M3 Pro 150 → 112、M5 Pro 307 → 230)。
- SSD: 0.945 × (1 − hit) GB ÷ 5〜8 GB/s。M3 Pro の実測は 3.72 MB ランダム pread で 4.19 GB/s (`F_NOCACHE`、M6 文書 §5-2)。
  M6 mini は「前世代比 2 倍」(公称)。5 は保守、8 は楽観。
  > **2026-09-22 実測でこの前提は落ちた。M6 256GB の天井は 3.34 GB/s (M3 Pro 1TB の半分)。
  > §3 の表の引き直しは [M6_SSD_BANDWIDTH.md](M6_SSD_BANDWIDTH.md) §4-1。**
- ホスト/residency: 10 ms。Tsugumi の M3 Pro 実測は io 15.75 + host 5.73 ms/tok ([27 §2](../qwen35moe/27-PHASE6-THROUGHPUT.md)) で、
  expert タッチが 320 → 480 個/tok に増えるので、そのままなら 20 ms 超。10 は `MTLResidencySet.commit()` を減らした後の値。

| 構成 | expert キャッシュ | スロット/層 | hit | SSD ミス/tok | DRAM | SSD (8〜5 GB/s) | 直列合計 | **tok/s (直列)** | 重なった場合の上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M6 16 GB (予約済) | 6.5 GB | 69 | 0.72 | 0.27 GB | 28 ms | 34〜54 ms | 72〜92 ms | **11〜14** | 16〜23 |
| M6 24 GB | 14.5 GB | 153 | 0.82 | 0.17 GB | 29 ms | 21〜33 ms | 60〜72 ms | **14〜17** | 23〜26 |
| M6 32 GB | 22.5 GB | 238 | 0.87 | 0.13 GB | 29 ms | 16〜26 ms | 55〜65 ms | **15〜18** | 25 |
| M5 Pro 64 GB | 54.5 GB (全常駐) | 512 | 1.0 | 0 | 17 ms | 0 | 27 ms | **37** | 37 |

「重なった場合」= max(DRAM, SSD) + 10 ms。次層の expert を前層の router から予測して先読みする腕 (Tsugumi の `mtp_router_xlayer.py` 系) が要り、
Flash-Next での的中率は**未確認**。IQ3_XXS の 32 GB は重なれば全常駐の天井 (§3-1) に着く。

参考 (Q4_K_XL、改訂前の値): 16 GB 6〜8、24 GB 9〜11、32 GB 10〜13。IQ3_XXS で 1.5 倍前後になるのは、SSD ミスが 0.25 → 0.13 GB に減る分と、dense が 0.5 GB 減る分。

### 3-1. 全常駐でも 25 tok/s が天井 (導出)

1 トークン 3.95 GB ÷ 130 GB/s = 30 ms + 10 ms → **約 25 tok/s**。RAM を無限にしても M6 (170 GB/s) では超えない値で、24 GB と 32 GB の差 (1〜2 tok/s) より大きい取り分が 2 つある:

- **dense の bpw。** 3.95 GB の 3.0 GB が dense (Q6_K / Q8_0)。affine Q4 g64 に落とせば 1.6 GB → 合計 2.55 GB → 20 ms + 10 → **約 34 tok/s**。品質側は**未確認**。
  llm-server も `output.weight` を毎トークン全読みしているので、同じ手が向こうにも効くはず。
- **SSD 先読みの重なり。** 直列 55〜65 ms → 重なって 39 ms。上の表の 2 列の差。

M5 Pro (307 GB/s) の天井は約 47、M5 Max (546 GB/s) で約 80。Maker Faire のグループが M5 Max MacBook Pro で回していたのはこの帯域があるから (公称)。

### 3-2. 長文脈

PC では 0k → 120k で 24.2 → 17.6 tok/s (−27%、実測)、llm-server の IQ3_XXS は 0k → 96k で 62 → 30 (MTP 込み、−51%)。
Mac は帯域が小さいので attention 側の劣化率はこれより大きい。**128k 埋め切り平均で読むなら §3 の直列値から 3 割引く**:
32 GB で 11〜13、24 GB で 10〜12。15 を保つには「重なった場合」の側 (32 GB で 18 前後) が要る。

### 3-3. 参考: 手元の M3 Pro 18 GB / 150 GB/s で動かした場合 (導出)

M6 を待たずに移植の検証をするなら、この機械で回すことになる。

**decode: 約 9 tok/s (直列)、SSD が隠れれば 13。**
expert キャッシュは 18 − 9.5 = 8.5 GB → 90 スロット/層 → hit 0.76 → SSD ミス 0.23 GB/tok。
DRAM (3.0 + 0.72) GB ÷ 112 GB/s = 33 ms、SSD 0.23 GB ÷ 4.19 GB/s (実測) = 55 ms、residency は現状の 20 ms → 直列 108 ms。

**prefill: 短文脈で 100〜150 tok/s。SSD の 1 回の全走査がチャンクごとに要る。**

- チャンク 2048 トークン × top-10 = 20,480 回の割り当てが 512 expert に散るので、**層内の全 expert をほぼ必ず触る**。
  つまり 1 チャンクごとに expert 全体 48.6 GB を SSD から読み直す (RAM に残せないので毎回)。トークンあたり 24 MB。
  mmap + `F_RDADVISE` の実測 11.99 GB/s ([mtp/52](../mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md)) で 2.0 ms/tok、`F_NOCACHE` 連続読みの 4.74 GB/s なら 5.0 ms/tok。
  **チャンク幅が I/O の床を直接決める** (4096 なら半分)。
- GPU: prefill で動く重みは routed 2.36B + shared 0.24B + dense 約 3B = 約 5.6B param → 約 11 GFLOP/tok。
  Tsugumi の Qwen3.5 prefill は GPU busy 中 約 2 TFLOP/s (265 tok/s、gpu 3.05 ms/tok、[27 §2](../qwen35moe/27-PHASE6-THROUGHPUT.md)) なので 5.6〜7 ms/tok。
  IQ2_S の dequant は affine Q4 より重い (コードブック引き) ので、この値より悪い側に振れる (**未確認**)。
- 壁時計: io と gpu が重なれば 6〜8 ms/tok (125〜165 tok/s)、直列なら 8〜12 ms/tok (85〜125 tok/s)。
  PC の 320〜390 tok/s (実測) の 3 分の 1。**32k のプロンプトで 3.5〜6 分、128k で 15〜25 分。**
  長文脈では full attention 12 層の分がさらに乗る (PC でも 32k → 120k で −16%)。
- PLE: 32,768 本のランダム読み/チャンクはチャンクの計算に隠れる (§2)。

decode の 9 tok/s は M6 16 GB (11〜14) より SSD の分だけ遅く、prefill は GPU 18 コアの分だけ M6 12 コア (tensor ops 無し) より速いかもしれない (M6 文書 §7-4 の「比の分母」と同じ未確認)。
**移植の正しさの検証には使えるが、運用速度の代表値にはならない。**

---

## 4. 判定

| 選択肢 | 差額 (公称) | Flash-Next IQ3_XXS、短文脈 (直列 / 重なり) | 128k 埋め切り平均 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 16 GB / 256 GB (予約済) | — | 11〜14 / 16〜23 | 8〜10 | Tsugumi の Qwen3.5-35B 系の対象機としては有効。Flash-Next には足りない |
| 24 GB / 512 GB | +¥72,000 | 14〜17 / 23〜26 | 10〜12 | 線をまたぐ。SSD が 8 GB/s 側なら届き、5 GB/s 側なら届かない |
| **32 GB / 1 TB** | 米国 +$800 (日本価格未確認) | **15〜18 / 25** | 11〜13 | **Flash-Next を Mac で回すならこれ。** SSD が保守側でも短文脈は 15 に乗る |
| M5 Pro 64 GB / 1 TB | 日本価格未確認 | 37 | 25 前後 | 全常駐で MTP も効く。PC (23) を明確に超える唯一の Mac mini |

**24 か 32 かなら 32。** 差額の対価は tok/s の 1〜2 ではなく、(a) SSD が保守側に振れても 15 を割らない余裕、(b) 128k KV 2.2 GiB と 256k を抱えたときのキャッシュ減りの吸収、(c) 1 TB。
ただし短文脈 15〜18 は「PC の Q4_K_XL 23 より遅く、品質は 2.5 bpw」であり、**PC の代替にはならない**。Mac mini で買う理由は据え置きの静音・省電力側にしかない。

順序として、**IQ3_XXS の品質判定 (`PLAN-quant-eval-pi.md`、PC vs llm-server の実タスク比較) を先に済ませる**。ここで劣化が許容できなければ Q4_K_XL 前提 (32 GB で 10〜13) に戻り、24 / 32 GB を買う理由が消える。

---

## 5. 到着後 (9/22) に取る数字

M6 文書 §7 の 5 つに加えて 1 つ:

| # | 測るもの | 方法 | 効く判断 |
| --- | --- | --- | --- |
| 6 | **2 MB のランダム pread を 10 本並列で投げたときの実効 GB/s** (`F_NOCACHE`) | M6 文書 §7-2 のプローブをサイズ 1.97 MB / 並列 10 にする | §3 の SSD 列。5 なら 32 GB の下限側 (15)、8 なら上限側 (18)。256 GB 構成の値は 512 GB / 1 TB と違う可能性がある (M6 文書 §5-2) |

---

## 6. 未確認リスト

1. M6 mini の SSD 実速度 (256 GB / 512 GB / 1 TB それぞれ)。「2 倍」の比較対象
2. IQ3_XXS の品質 (実タスク比較は未実施。KLD の基準 logits は llm-server の `/root/qfn-bench/logs/kld_q4kxl.bin`)
3. expert キャッシュ hit の高スロット側 (100〜250/層) の実値。§1-5 の外挿は 64 までしか裏付けが無い
4. 層先読みの的中率 (Flash-Next の router で次層を予測できるか)
5. IQ2_S / IQ4_NL の Metal dequant の速度 (decode は帯域律速なので効かないはずだが、prefill の GEMM 内展開は affine Q4 より重い)
6. dense を affine Q4 に落としたときの品質 (§3-1 の 34 tok/s 案)
7. 既製ツール (llama.cpp Metal / MLX) で RAM を超えるモデルを mmap したときの実速度
8. M6 32 GB / 1 TB と M5 Pro 64 GB / 1 TB の日本価格

---

## 出典

リモート (`ssh masah@192.168.0.199`、`C:\LLM`、シェルは PowerShell):

- `HANDOVER-qfn-engine.md` — GGUF メタデータ、E1/E2/E3 の要約、PC の 1 ステップ分解
- `HANDOVER-pc-qfn-mtp.md` — IQ4_XS / Q4_K_XL、MTP、128k 埋め切り平均の計測法
- `HANDOVER-llm-server.md` — V100 + 3090 の配置、PLE の lazy read 問題、VRAM 定数、IQ3_XXS 本番構成
- `PLAN-article-and-3rd-gpu.md` §2-1 — IQ3_XXS のサイズ内訳 (expert 962 MiB/層、dense 3.24 GiB、PLE 26.82 GiB)、19.9 ms の床、50.2 / 61.8 tok/s
- `PLAN-quant-eval-pi.md` — IQ3_XXS の品質を実タスクで比べる計画 (未実施)
- `qfn-bench\RESULTS-e2.md` — ルーティングのトレース、局所性、動的キャッシュのシミュレーション
- `RESULTS-pc-qsa-gather.md` — 疎注意 gather パッチの結果 (128k 23.13、240k 20.70)
- `models\unsloth\Qwen3.8-Flash-Next-GGUF\UD-Q4_K_XL\*-00001-of-00004.gguf` — `qwen4exp.ple.*` (layers [1]、ngram_size 3、heads_per_ngram 8、head_vocab_sizes 約 2000 万) を gguf-py で読んだ
- `llama.cpp-gather\src\models\qwen4exp.cpp` — PLE は 1 層のみ (`n_ple != 1` で例外)

手元:

- `/Users/mh/LLM/qwen38-flash-next-article/draft_v2.md` — note 記事草稿 (2026-09-07)。モデル概要、PLE 表、llm-server の確定構成
- [M6_MAC_MINI_TARGETING.md](M6_MAC_MINI_TARGETING.md) — 帯域 153 / 170 GB/s、SSD 実測、構成と価格
- [qwen35moe/27-PHASE6-THROUGHPUT.md](../qwen35moe/27-PHASE6-THROUGHPUT.md) — Tsugumi の decode 内訳 (residency commit 13.6 ms/tok)、prefill の gpu / io

公称:

- [Apple Japan: Mac mini 購入ページ](https://www.apple.com/jp/shop/buy-mac/mac-mini) — M6 16/256、16/512、24/512、32/1TB、M5 Pro 24/512、64/1TB
- [AppleInsider: M6 Mac mini arrives with a new record-high starting price](https://appleinsider.com/articles/26/08/25/m6-mac-mini-arrives-in-ram-and-ssd-constrained-environment) — 32 GB +$200、1 TB +$500
