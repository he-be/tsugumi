# 15. 今後の実験方針 — RAM を増やさずに 12K の decode を詰める順序 (A〜G) と、運用点に向けた統合

書いた日: 2026-09-14。対象は [13](13-KERNEL-TO-GPU-WAIT.md) までの `Qwen38Runner` (DS4-IQ2 GGUF + PLE Q4_1、M3 Pro 18 GB)。
表記は 01 と同じ (実測 / 導出 / 未確認)。**この文書に新しい実測は 1 つも無い。**§2 G は 2026-09-14 に書き直した: 基準を Gemma の server 経路に置き、Ornith の現状 (末尾だけのチェックポイント) を要件にしない形にした (Gemma・Ornith のコードから)。数字はすべて 01〜13 と `scratch/qwen38/{pipe12,long13}/` のログからの引用か導出で、
進退の根拠にはしない (メモリ `no-decision-on-derived-projections`)。各段は「測れる最小の段」から並べ、1 段ごとに実測を 1 つ出して次に進む。

番号 14 は n_max 2 (連鎖ドラフト、[13 §9](13-KERNEL-TO-GPU-WAIT.md)) が使う (別セッションで進行中、`Q38_MTP_CHAIN`)。本書はその次以降の順序。

## 0. 結論 (方針)

1. **Qwen3.8 の decode は Gemma と同じ regime にいる。**待ちの通貨はデバイスのバイト (SSD から読む非常駐 expert) で、Ornith の「host のページ写像」ではない
   ([qwen35moe/27 §9-6](../qwen35moe/27-PHASE6-THROUGHPUT.md))。加えて dense が 1 トークン約 5 GB (Gemma の core 1.27 GB の 4 倍) なので GPU 床も高い。
   late パイプラインが Ornith の +31〜41% に対し +4〜13% で止まったのはこの構造で、**隠す先が小さい**。
2. だから残る手は「隠す」より **(a) 露出している host の仕事を減らす、(b) GPU 床を下げる、(c) 1 ステップで出すトークンを増やす** の 3 つに絞る。
   RAM 軸 (先読み既定化・上位 N・前もって常駐、[13 §4](13-KERNEL-TO-GPU-WAIT.md)) と prefill、行 y の先行 ([13 §6-1](13-KERNEL-TO-GPU-WAIT.md)) はユーザー判断で閉じたままにする。
3. 順序は **A (n_max 2、進行中) → W (ウェイトのダイエット: `down` の pad 列) → B (expert の台帳) → C (hc) → D (12K の MTP ドラフト) → E (decode の QSA 選択) → F (小カーネル) → G (統合)**。
   W を B の前に置くのは、B の台帳の X (MB) と非常駐 MB が expert 1 個あたりのバイトで変わるため (§2 W)。
   B が一番大きく一番無茶で、Gemma D-P6/P7 ([mtp/51](../mtp/51-D-P6-MMAP-PROTOTYPE.md)・[52](../mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md)) と Ornith q/39 ([qwen35moe/39](../qwen35moe/39-RESIDENCY-COMMIT.md)) で勝った形をまだ持っていない唯一のもの。
   ただし B は計器だけの段 (B-1) を先に置き、当たり率が出なければ B-2 以降に進まない。
4. G (server 経路・prompt cache・英語のツール呼び出しでの品質) は速度と独立に要る。**12K の prefill 133 s をターンごとに払う限り運用点は成立しない**ので、
   G-0〜G-2 は B〜F の途中でも先に入れてよい。**基準は Gemma の server 経路で prefill を払わずに済む場面で、Ornith の現状は要件にしない** (Ornith は 1 ターン目の再生成が全再処理、再生成・指示つき再生成で前の回答まで払う)。
   GDN の再帰状態はカーソルで戻せないので手段は [SPEC CACHE-8](../serving/SPEC.md) のチェックポイントだが、取る位置 (P−1・最後のメッセージ本文の末尾・`<tool_call>` の直前) は Gemma の場面表から決める (§2 G)。
   Gemma でも全部払う場面 (履歴の切り詰め・チャットの切り替え) は基準の外。

## 1. 現在地と床 (引用)

steady な decode 1 トークン (instruct、短文脈、腕 late = 既定、[12 §2-1](12-PIPELINE-PREVIEW.md))。12K は long13 のログ。

| | 短文脈 late | 短文脈 prev10 (既定 off) | 12K late |
| --- | ---: | ---: | ---: |
| wall (ms) | 130〜156 | 108〜120 | 144〜196 |
| GPU busy (pre + routed + shared + head、導出) | 66〜72 | 66〜68 | 70〜85 |
| kernel→GPU の待ち | 32〜36 | 25〜31 | 30〜40 |
| advise (syscall 約 1,400 回) | 15〜20 | 7〜8 | 15〜22 |
| 非常駐 MB / ステップ (11 §1、13 §3) | 156〜189 | — | (未計測) |

床 (どれも実測済み、詳細は各文書):

| 床 | 値 | 出所 |
| --- | --- | --- |
| RAM | 18 GB。decode の wired 12〜15 GB、file-backed 1〜3 GB | 09 §1、10 §5、13 §4 |
| SSD 冷読み | 6.2〜6.6 GB/s (pread)。Gemma の F_NOCACHE 6.0〜6.5 と一致 | 05 §1、[mtp/33 §0-2](../mtp/33-M8-IO-FLOOR.md) |
| ドライバのページイン | 2.2 GB/s、デバイス全体で直列 | 13 §1、§6 |
| ユニファイドメモリ | 150 GB/s 公称。Q8_0 GEMV 126〜130 GB/s | 02 §1-1 |
| ドライバの固定費 | 使用済みビュー 0.05 ms/本、48 層で 2.4 ms | 13 §1、§5 |

導出: dense 約 5 GB + routed 715 MB を 150 GB/s で読むと約 38 ms → **T=1 の絶対上限は約 26 tok/s**。非常駐 170 MB を 6.2 GB/s で読むと 27 ms で、
観測の kernel→GPU 25〜36 ms とほぼ同じ → **routing の鎖の中で SSD 床に当たっている**。手が届くのは advise の host 時間、固定費の床超過分 (7 → 2.4 ms)、GPU 床の超過分 (hc など)、
そして 1 ステップあたりのトークン数。

## 2. 段の順序

各段に「最小の段 → 計器 → 記録する数字 → 止める条件」を置く。**閾値は置かない** (Gemma の教訓、[mtp/22 §2](../mtp/22-GOAL-RESET.md): 達成/未達の欄を作らない)。
止める条件は「腕の差が反復の差に埋もれる」(11 §4 の形) と「トークン列が変わる」の 2 つだけ。

### A. n_max 2 (連鎖ドラフト) — 進行中 (14)

手順は [13 §9](13-KERNEL-TO-GPU-WAIT.md) のとおり。本書で足すのは見る場所 1 つ:

- **畳めるのは dense の GPU と固定費だけで、I/O は畳めない。**非常駐 MB は T に比例して増える (T=1 156〜189、T=2 322〜432 MB、11 §1)。
  検証の pre GPU は T=2 で 53 ms (T=1 50〜57、12 §2-1、long13 でも 12K で 53)。expert の和集合は 12K で T=1 480 / T=2 838 / T=3 1,222 本 (09 §3-1)。
- Gemma [mtp/21](../mtp/21-GOAL-CONDITION-RESULTS.md) は「ブロックが SSD ミスを畳む利益は冷たいキャッシュで実測ゼロ」、Ornith [qwen35moe/36](../qwen35moe/36-MTP-DECODE.md) は
  「勝ち負けを決めるのは受理率ではなく expert 取得の相乗り」。n_max 2 の端から端では **トークン/ステップと非常駐 MB/ステップの両方を同じ行に書く**。
- B が入ると A の損益が変わる (A の検証 T=3 の非常駐が B の pread で読まれる) ので、A の後に B を測るときは n_max 1 と 2 の両方を腕に入れる。

### W. ウェイトのダイエット — `down` (Q2_K) の pad 列を落とす

**状態 (2026-09-14)**: W-1〜W-4 を実施 ([16](16-WEIGHT-DIET.md))。
- どちらもビット一致を確かめたうえで、ユーザーの判断で既定にした。
  - `down` の pad を落としたサイドカー: 12.42 GiB
  - GEMV で読む F32 の BF16 化: 279 → 140 MiB
- 12K の spec では wired の最大が 14.84〜15.00 → 14.28〜14.57 GB になった。
- W-3 の 12K の long2 / long3 と、非常駐 MB の別走行は未走行。

**Gemma との対応。**Gemma の W1/W2 ([mtp/44](../mtp/44-W1-WEIGHT-DIET.md)・[45](../mtp/45-W2-SYM-ADOPTION.md)) は
「`bias == −8 × scale` がビットパターンで成り立ち bias は情報ゼロ → 落とすと帯域床の上でバイト比どおり速い」で、出力バイト一致のまま +8.8%。
GGUF の IQ2_XXS / Q2_K / Q8_0 には「付帯値が別の値から導出できる」冗長は形式上無い (付帯値はブロックの `d`、Q2_K は `d`/`dmin` と 4 bit の scale/min で、どれも独立)。
ただし**「読んでいるが出力に効かないバイト」**という意味での同類が 1 か所ある。

**何が情報ゼロか (形式の定義から、データ検定は不要)。**GGUF ヘッダ (2026-09-14 に読んだ、テンソルのページには触れていない) は
`ds4.qwen4.down.logical_input 640` / `physical_input 768`。`ffn_down_exps` (768 × 2560 × 512、Q2_K) は 640 列を 256 × 3 ブロックに水増ししている。
acts の列 640〜767 は常に 0 (`moe_ggml.metal` の phase 1 は書かない、08 §1、GEMM 経路は捨てる、06)。Q2_K でサブブロック j の逆量子化に効くのは
`d`・`dmin`・`scales[j]`・その 16 重みの `qs` だけなので、**3 ブロック目の `scales[8..15]` (8 B) と `qs[32..63]` (32 B) は値が何であっても出力に効かない**。
Gemma §1 の恒等式と違い、成り立つかをデータで確かめる必要が無い。

| (導出、ヘッダの形から) | いま | 切り詰め後 |
| --- | ---: | ---: |
| `down` 1 行 | 252 B | 212 B (−15.9%) |
| expert 1 個 (gate 422,400 + up 422,400 + down) | 1,489,920 B | 1,387,520 B (−6.9%) |
| 1 トークンの routed (10 × 48) | 約 715 MB | 約 666 MB |
| `ffn_down_exps` 48 層 | 14.77 GiB | 12.42 GiB (−2.34 GiB) |

**効く先は I/O の側だけ。**非常駐 MB (待ち 0.08 ms/MB、13 §3)、SSD の読み、同じ file-backed に入る expert の本数、ビューの wired。
GPU の計算はほぼ変わらない (GEMM は既に pad を捨て、レーン版の pad レーンは ±0 を足しているだけ)。Gemma の +8.8% は dense/head (帯域床) と io の両方から来ていたが、
Qwen3.8 の dense (Q8_0 / F16) には同じ冗長が無いので**移せるのは io の側だけで、バイト比も 10% ではなく 6.9%**。tok/s の見積もりは置かない。
Gemma W2 では io が −11.7% とバイト比 (−9.7%) より効いた (1 本の読みが短くなり、キャッシュの効率も上がる) ので、非常駐 MB はバイト比と別に測る。

**W-1. 詰め直し (GPU 無し、計測タスクと SSD を奪い合うので計測の無い時間に)。**
- `ffn_down_exps` だけを 1 行 212 B に詰めたサイドカーを `ple/` の隣 (`down/`) に書く。3 ブロック目は `scales[0..7]`・`qs[0..31]`・`d`・`dmin` の順 (44 B、構造体の並びを保つ)。
  元の GGUF は残す (参照器 `reference_forward.py` と `expert_kernel_fixture.py` は元を読む)。
- 14.8 GiB 読んで 12.4 GiB 書く。層ごとに書いて RSS を固定し、`memlog.sh` 越しで最初の数分に RSS・実書き込み量・Swapouts を見る (メモリ `long-jobs-watch-first-minutes`、`bench-hygiene-m3pro`)。
- 検定: 全層で「切り詰め後の各行 = 元の行から該当バイトを抜き出したもの」をバイト比較。加えて第 24 層の数個の expert を gguf-py で逆量子化し、列 0〜639 が元と一致すること。

**W-2. カーネル (GPU 半日)。**
- Q2_K の行を読むのは `moe_ggml.metal`・`Qwen38Runner.swift`・`GGUFFile.swift`・`Q2ExpertCheck.swift`・`Q2GemmBench.swift`・`expert_kernel_fixture.py` (2026-09-14 の grep)。
  **綴りではなく型で探す** (Gemma [mtp/45 §3a](../mtp/45-W2-SYM-ADOPTION.md): 検定 69/69 が通ったまま decode の gate/up だけが束縛を読み違えていた)。
  最終確認は `block_q2_K` へのポインタと `down_row_bytes` の全参照を機械的に並べて行う。
- レーン版 phase 2 は、ブロック 2 の後半 (ix = 2, iq = 1) のレーンを読まない (いまも ±0 しか足していない)。GEMM の逆量子化は最後のブロックを半分の形で読む。
- 正しさ: `--q2-expert` を詰めたフィクスチャで回し、**元の形の出力とビット一致**を見る (相対誤差ではなく一致)。レーン版と GEMM 版の両方、T = 1 / 2 / 32 以上。
  MTP の blk.48 (Q4_K / MXFP4、640 は 32 で割り切れ pad 無し) は対象外。

**W-3. ランナー (GPU 1 日)。**
- `down` の `bytesPerRow` (いま `Qwen38Runner.swift` の expert 範囲・ビュー・advise・pread) をサイドカーから取る。フラグは `Q38_DOWN_SIDECAR` (既定 off)。
- 腕: late (既定) / W。n_max 2 が既定になっていれば n_max 1 / 2 の両方で。ABBA、§3 の規則、短文脈 3 本 + 12K の long。
- 記録: steady tok/s、kernel→GPU ms、advise ms、wired 最大 / file-backed / Swapouts、トークン列の一致。非常駐 MB は `Q38_COUNT_MISS=1` の別走行で (速度の腕に入れない)。
- 止める条件: トークン列が変わる (= 読み違い、速度を読まない)。腕の差が反復の差に埋もれる。

**W-4. F32 の BF16 化 (検定だけ、GPU 無し)。**
- `ffn_gate_inp` (F32、245 MiB)・`ssm_alpha`/`ssm_beta` (34 MiB)・`ssm_conv1d`・norm 類。上流が bf16 なら F32 の値は BF16 でビット単位に表せるはずで、
  成り立てば半分に詰めても出力は変わらない (**未確認**)。Gemma §1 と同じくビットパターンで検定する (約 290 MB を読む)。
- ただし実測の費用は router の GEMV 1.04 ms / 48 回・alpha 0.47 ms / 48 回 (02 §2-1) で、取り分の上限は 1 トークン 1 ms 未満と wired 約 145 MB。
  恒等式が成り立っても W-3 に混ぜず、数字を書いて閉じるかをユーザーに渡す。

**当てはまらないもの。**
- hc の F16 (1 トークン 1.2 GB): 冗長が無い。C-2 (Q8_0 サイドカー) は情報を落とすので Gemma のダイエットとは別種。
- Q8_0 の dense・`output.weight`・IQ2_XXS の gate/up: 付帯値はブロックの `d` だけ。
- scale の圧縮 (Gemma W1 §5) とコードのエントロピー符号化 (W3): 「展開器が読み出しより遅い」で閉じた理由がそのまま当てはまる。
- PLE の Q4_1 (30 GB): 1 トークン 16 行しか読まないので decode に効かない。

### B. expert の台帳 (host が「何が常駐か」を持つ) — 一番大きく、一番無茶

**狙い。**いまの decode は 64K 本の no-copy ビューを作ったまま持ち、何がページキャッシュに残っているかを知らない (12 §4「キャッシュ状態を持たないので未着手」)。
mincore で数えると計器そのものが 30 ms 効く (11 §1)。host が近似の台帳を持てば、RAM を増やさずに 3 つが取れる:

| 取り分 | いまの費用 | 出所 |
| --- | ---: | --- |
| 常駐と分かっている expert への advise を省く | 15〜22 ms/トークン (syscall 約 1,400 回) | 12 §2-1 |
| 非常駐だけを host の pread (6.2 GB/s) で commit 前に読む。ドライバのページイン (2.2 GB/s、直列) を通さない | kernel→GPU の待ちのうち 0.08 ms × 非常駐 MB ≈ 13〜15 ms | 13 §3 |
| 常駐 expert を先に流す hit-first、ビューを捨てて固定費を床 (2.4 ms) に近づける | 定数 6.5〜7.7 ms | 13 §5、Gemma hit-first +14.4% (メモリ `port-io-hiding-concept-first`) |

前例: 02 §3 の R 腕 (2 GB の residency set、直列ランナー) は単独で A より速く、advise と重ねて引き分け、裾は悪化。11 §3 の pread 腕 (T ≥ 1) は
「キャッシュ済みも全部読む」ので負けた (route 70 ms)。**どちらも「欠けだけを知る」手段が無かった**。パイプライン化 (12) 後の形では未測定。

**B-1. 計器だけ (推論経路は無改変、GPU 半日)。**
- 台帳: expert (層, id) ごとに「最後に advise / pread / GPU で使った時刻 (ステップ番号)」を持ち、直近 X MB ぶんを常駐と見なす LRU の近似。X は wired の観測 (使ったビューの分 6〜7 GB、13 §2) と file-backed (1〜3 GB) から振る (2 / 4 / 6 GB)。
- 同じ走行で `Q38_COUNT_MISS=1` の mincore を正解にし、**台帳の当たり率 (常駐と言って本当に常駐 / 非常駐と言って本当に非常駐) をステップごとに記録する**。
  速度は見ない (mincore が計器として中立でないため、11 §1)。
- 記録: X ごとの precision / recall の中央値と p10、12K と短文脈で各 1 本 (数字だけ)。
- 止める条件: 非常駐の recall が上がらない X が無い (= LRU 近似が実際の追い出しと合わない)。その場合は LFU (Ornith q/27 §6-1 で lfu が lru に 3〜3.5 pt 勝つ) を 1 本だけ試して閉じる。

**B-2. advise を省く (製品差分は小さい、GPU 1 日)。**
- 台帳で常駐と見なした expert には advise を出さない (`Q38_LEDGER_ADVISE=1`)。外れたときの費用はいまと同じ (ドライバのページイン)。
- 腕: late (既定) / B-2。ABBA、instruct、seed = パス、200 トークン、code / tool / explain + long、memlog 越し。
- 記録: steady tok/s、advise ms、kernel→GPU ms、トークン列の一致、wired / file-backed / Swapouts。
- 止める条件: kernel→GPU が advise の減った分だけ増える (= 省いた advise が実は要った)。

**B-3. 欠けだけを host スレッドで pread (GPU 1〜2 日)。**
- 台帳で非常駐と見なした範囲だけを `preadRanges` (4 スレッド) で読み、その間に shared(L) を流す (12 の late の形)。advise は読まない側に残す。
- 12 §1 の順序: `top-10 → shared(L) commit → [pread 欠けだけ] → views → cb2(L) commit`。pread は shared の GPU 2.8 ms と重なるだけで、ほぼ露出する。
  それでも 2.2 GB/s (ドライバ) → 6.2 GB/s (pread) の差が出るかを見る。
- 記録: B-2 と同じ + pread ms / MB。**pread の MB と mincore の非常駐 MB を突き合わせる 1 本を計器つきで取る** (メモリ `ratio-vs-base-mismatch`: 代用の絶対値を 1 度は本番と合わせる)。
- 止める条件: 11 §3 と同じく route が伸びて wall が縮まない。

**B-4. ビューを持つ量を切る (RAM を減らす方向、GPU 1 日)。**
- 台帳の LRU で外れた expert のビューを解放し、持つビューを X MB に上限する。狙いは 2 つ: (1) wired を今の 12〜15 GB から下げる (Swapouts の縁から離れる)、
  (2) ドライバが勝手に外して対応付け直す約 200 MB/ステップ (13 §5) を、自分で外す形にして定数を床に寄せる。
- 13 §5 の作り直し腕 (定数 18〜21 ms) が上限の悪い側、保持 (6.5〜7.7 ms) が今。X を 2 / 4 / 6 GB で振る。
- 記録: 定数 (13 §3 の回帰)、wired 最大、tok/s。
- 落とし穴: `requestResidency` した分は回収されない wired になる (09 §7)。residency set は使わず、ビューの寿命だけで制御する。Gemma P-6 は 1 ステップ分 806 MB だけ wire した ([mtp/47 §3](../mtp/47-D-MMAP-RESIDENCY-PROPOSAL.md))。

**B-5. hit-first (B-4 の後、GPU 半日)。**常駐と見なした expert の組を先のスロットに並べ、cb2 をそこから先に流す。Gemma は +14.4%、Ornith は「払うスレッドが間違っていた」(q/39) で 36.5 → 8.3 ms/パス。
Qwen3.8 ではドライバのページインがデバイス全体で直列 (13 §6) なので、cold が 1 本でもあれば後ろの warm も待つ。**先に流した warm が cold の前に GPU で始まるかを、13 §6 のプローブに腕を足して確かめてから**ランナーに入れる。

### C. hc の F16 (1 トークン 1.3 GB)

hc_attn + hc_ffn で 16 ms、帯域なら 8.7 ms (02 §2-1「行幅 320 は stride のまま」、09 §3-2)。T=2 では 11〜15 ms ずつ (09 §3-2)。

**C-1. カーネルの形 (checkpoint 無改変、GPU 半日)。**
- `--ggml-dense-bench` に hc の 5 テンソル (`hc_attn_down` F16 320×10240 / `hc_attn_up` 10240×320 / `hc_attn_inject` 4×10240 / `hc_ffn_*`) を足し、行幅 320 用の形を 2 つ比べる:
  (a) 1 行を 10 レーン (32 要素チャンク) で分け、3 行を 1 SIMD グループに、(b) 行幅 10240 側 (up/inject) は chunk のまま、down だけ (a)。
- 記録: GPU ms / 48 回、出力差、T = 1 / 2 / 3。
- 止める条件: 帯域換算 (1.3 GB / 150 GB/s = 8.7 ms) に対して伸びなくなったところ。

**C-2. Q8_0 のサイドカー (checkpoint を変える。ユーザー判断が要る)。**
- hc の F16 を Q8_0 に打ち直したサイドカー GGUF (PLE と同じ作り、`ple/` の隣) を読む。読むバイトが 1.3 → 0.65 GB、dense の wired も 0.65 GB 減る。
- Gemma の weight diet (bias 落としで tok/s +8.8%、peak −0.4 GB、[mtp/45](../mtp/45-W2-SYM-ADOPTION.md)) は**出力バイト一致の可逆な変更**で、ここの前例にはならない
  (Qwen3.8 で同種なのは §2 W の pad 列)。Q8_0 化は情報を落とすので、
  **参照器 (`reference_forward.py`) にも同じサイドカーを読ませて logits を比べ、hc だけの差を出す**。France / Fuji の一式と、long の最初の 200 トークンの top-1 の一致数。
- 記録: logits の相対誤差、top-1 のずれ、C-1 と同じ速度。

### D. 12K の MTP ドラフト

long13 と pipe12 のログ: ドラフト 1 回は短文脈 15〜20 ms、12K 22〜35 ms。差は pre 9〜16 ms (短文脈 2〜3) = 11.7K 位置を見る密な注意が T < 32 の host レーン経路 (`attentionHost`) に落ちている分。
Ornith [qwen35moe/38](../qwen35moe/38-MTP-VERIFY-PATH.md) の split-KV がこれと同じ形で、t4 ×0.80 → ×1.11。

**D-1. 長い n の T < 32 注意 (GPU 1 日)。**
- `attentionHost` の 5 パスを、n をブロック (例 1024) に割って各ブロックの max / Σexp / 部分和をレーンで出し、最後に 1 パスで畳む split-KV に。
  幹の decode (QSA 選択後の 2048 列) にも同じカーネルが効く。
- 記録: `--q38-small-bench` 相当の単体 (n = 256 / 2048 / 12K、T = 1 / 2 / 3) と、long の影モードのドラフト ms。
- 正しさ: MTP は `--qwen38-mtp-dump` + `mtp_reference.py`、幹は Fuji 予算 16 の一式。

**D-2. ドラフタの窓 (影モードだけ、GPU 半日)。**
- 参照実装はドラフタを密にしている (10 §2-1)。窓 (直近 2K / 4K) に切ったときの受理率を `--q38-mtp shadow` で測る。速度はまだ見ない。
- 記録: 受理率 (long / long2 / long3、greedy と instruct)、窓なしとの差。
- 止める条件: 受理率が窓なしから下がる (数字だけ書いて閉じる)。下がらなければ D-3 と一緒に MTP の KV も窓に切れる。

**D-3. MTP の prefill を縮める (影モード、GPU 半日)。**
- いまはプロンプト全体を MTP に通す (12K で 11.3 s = 幹の 8.5%、10 §3)。末尾 N トークンだけ通したときの受理率を影モードで測る (N = 512 / 2K / 全部)。
- 記録: 受理率と MTP prefill s。

### E. 幹の decode の QSA 選択 (T < 32 は host のまま)

04 は T ≥ 32 だけを GPU (`q38_idx_topk`) にした。12K の decode は 3,000 ブロックのスコアを host で `sorted()` している。

**E-1. まず大きさ (GPU 数分)。**`Q38_SPLIT_PRE=1` で long の decode を 1 本流し、pre の wall − GPU を短文脈と比べる (09 §3-1 の 22 vs 10 ms はパイプライン化前)。
**E-2.** 10 ms 台なら `q38_idx_topk` を T < 32 にも使う (同点規則は `--q38-select-check` で済んでいる)。1 ms 台なら閉じる。

### F. 小カーネル (GPU 半日、まとめて 1 本)

02 §2-4 で「後回し」のまま: `attn_prep` 7.06 ms/48 回 (1 トークン約 1.8 ms)、GDN 3 本 (step / norm_gate / qk_norm、約 4 ms)。
`--q38-small-bench` は 03 で消したので、T 行版の引数で作り直してから。取り分の上限は合計 6 ms 弱 (導出) なので、B〜E の後。

### G. 運用点に向けた統合 (速度と独立) — エージェントとして使えるための prompt cache

**状態 (2026-09-14)**: G-1 の結線 ([17](17-SERVER-APP-WIRING.md))、G-0 のチェックポイントと G-2 の取る位置・捨て方 ([18](18-PROMPT-CACHE-CHECKPOINTS.md)) を入れた。
12K の場面表で prompt_n は全場面 Gemma 以下 (止めた後の再生成は Gemma 全部・Qwen3.8 1)。チェックポイントは SSD 置きが既定 (RAM 4 本は Swapouts)。
残りは G-1 の INV-1 (文法・テンプレート変種) と、ツール定義の `tojson` が上流の描き方と違う件 (17 §3-1)。

**基準は Gemma (2026-09-14、ユーザー指定)。**Qwen3.8 がエージェントとして使えると言えるのは、Gemma の server 経路 (`ServerInference` + `ServerPromptCache`) で prefill を払わずに済む場面が、
Qwen3.8 でも同じだけ済むときだけ。**Ornith の現状は要件の出所にしない**。Ornith は下の表のとおり Gemma に届いておらず、それを移すと届かないまま残る。

**Gemma の形 (2026-09-14 にコードから)。**
- 保持は 1 エントリ (直前の要求のプロンプト + 生成)。一致はトークン列の LCP (CACHE-1)。全一致なら末尾 1 トークンを捨てて再デコード (CACHE-3)。
- 分岐点が KV の末尾から 2048 トークン以内なら、カーソルを分岐点へ戻して続きだけ prefill (CACHE-2、DEV-13 = `KVCacheManager.maximumSafeRewind`、SWA リングの余裕)。それより深ければ全再処理。
- 例外で終わった要求はエントリを捨てる (`ServerInference.generate` の `defer`)。アプリの中断も `Task.checkCancellation` の `CancellationError` なのでここに入る。
- 描き直し == 生成 (INV-1) を、tool call の文法を描き直しの正準形に縛ること (GEN-8) とサーバー用のテンプレート変種 (DEV-12) で閉じている。閉じないのは自由形式の値の中のキー順だけ (DEV-15)。

**Gemma の形をそのまま使えない理由。**Qwen3.8 は GDN 36 層の再帰状態を持ち、1 トークンを引き算で取り出せない。カーソルで戻せるのは full attention 12 層の KV・インデクサ・MTP KV だけで、
`KVCacheManager.maximumSafeRewind` も再帰層があれば 0 を返す。**分岐点に戻るには、分岐点以前で取ったスナップショットが要る** (CACHE-8)。
Ornith から借りるのはこの機構 (`captureCheckpoint` / `restore` の API の形と `--qwen-resume` の検査の形) だけで、**どの位置で何本取るかは Gemma の場面表から決める**。

**Ornith の現状が Gemma に届いていない点 (コードから)。**`QwenServerSession.captureAtPromptEnd` はプロンプト末尾 (位置 P) でだけ取り、`--ctx-checkpoints 2` で直近 2 要求分を持つ。
SPEC CACHE-8 が求める「最後の user メッセージの直前」では取っていない。P のチェックポイントは同じ長さのプロンプトに使えない (`QwenPromptCache.match` の `candidate < promptIDs.count`) ので、
1 ターン目の再生成は全再処理、2 ターン目以降の再生成・指示つき再生成は前の回答まで prefill し直す。

**場面ごとの要件。**P = プロンプト長、U = 最後のメッセージ本文の末尾 (その `<|im_end|>` の手前) の位置。Gemma の列はコードの規則からの導出で、**Qwen3.8 も Gemma もこの表の実測は 0** (G-2 で両方測る)。

| 場面 | 分岐点 | Gemma (基準) | Ornith の形を移した場合 | Qwen3.8 の要件 → 取る位置 |
| --- | --- | --- | --- | --- |
| 次のターン・ツール応答の往復・ストリーミング | KV の末尾以降 | 足した分 | 足した分 | 足した分 (生きている状態から継続) |
| 同じプロンプトの再送・再生成 / 別回答 (U4)、**1 ターン目を含む** | P (全一致) | 1 トークン (回答が 2048 以内) | 1 ターン目は全部、以降は前の回答 + ユーザーターン | **1 トークン** → P−1 |
| 指示つき再生成 (U5: 最後のユーザーターンの末尾に 1 行) | U 付近 | 足した 1 行 + 生成プロンプト | 前の回答 + ユーザーターン | **足した 1 行 + 生成プロンプト** → U−m |
| tool call ターンの描き直し | 生成したターンの中 | INV-1 で分岐しない。ずれたら分岐点から (DEV-15) | そのターンの先頭 (前の要求の P) から | **INV-1 で分岐させない** (G-1)。残るずれは分岐点の直前から → decode 中の `<tool_call>` の直前 |
| 回答が 2048 を超えた後の再生成 | P | 全部 (DEV-13) | 前の回答 + ユーザーターン | 1 トークン (P−1 と同じ。基準より上) |
| 生成中に止めた後の再生成 | P | 全部 (エントリを捨てる) | 前の回答 + ユーザーターン | 基準は全部。P−1 は prefill 中に取るので、prefill 後に止めたなら 1 トークン (基準より上) |
| 生成中の例外 (文法の行き止まりなど) | — | 次は全部 | 次は全部 | 基準は全部 |
| 履歴の切り詰め・チャットの切り替え・ツール宣言や thinking の変更 | 先頭付近 | 全部 | 全部 | 基準も全部 (§G の末尾) |

- m は U5 の 1 行を足したときに、元の本文の末尾のトークン化が変わる幅 (BPE の境界)。Qwen の pre_tokenizer は改行で区切るので 0 の見込みだが**未確認**。
  トークナイザで「U5 の指示文を足した列の [0, U−m) が元と一致する」最小の m を決める。
- 1 要求で持つのは P−1・U−m・その要求の生成中の `<tool_call>` の直前。前の要求の分は、次の要求が restore と取り直しを終えた時点で捨ててよい (Gemma の 1 エントリと同じ深さ)。
  本数は 2 + tool call 数、1 本 112.6 MiB (G-0)。12K spec の wired 14.28〜14.57 GB に乗るかは memlog で測る (**未確認**)。RAM で乗らない場合の形 (SSD へ退避など) は測ってからユーザーに渡す。

**G-0. ランナー側の状態 API と `--qwen38-resume` (GPU 半日、server 無し)。**

`Qwen38Runner` には reset / position / checkpoint / restore が無い (2026-09-14 の grep。`rollback(keep:)` は投機用で、直前の 1〜2 行にしか戻れない)。ランナーが持つ状態と戻し方 (コードから):

| 状態 | 持ち方 | 戻し方 | 1 チェックポイントのバイト (導出) |
| --- | --- | --- | ---: |
| GDN 36 層 (`linState` Hv×Dl×Dl f32、`linHist` (convK−1)×C f32) | 上書き。引き算が無い | コピーを取る | 36 × (3,145,728 + 122,880) B = 112.2 MiB |
| PLE の履歴 (`pleHist` 3×ngram×hc×e f32、`plePrev` 2 トークン) | 上書き | コピー (トークン列からの再計算も可) | 0.35 MiB |
| full attention 12 層の KV (`kCache` / `vCache`、f32) | 位置ごと (append-only) | カーソルを戻すだけ | 0 (12K で 576 MiB は生きている分) |
| インデクサ鍵 (`idxRawKeys` 位置ごと、`idxBlockKeys` は 4 行ブロックが揃った時点で書く) | 位置ごと | カーソル。**ブロック途中の位置に戻したとき、次の prefill がそのブロック鍵を書き直すか** (未確認) | 0 |
| MTP ヘッド (blk.48) の KV | 位置ごと | カーソル。ただし投機ループでは MTP KV が幹より `kept` 行遅れる (`mtpTokens` / `mtpRows`、[14](14-MTP-NMAX2.md)) | 0 (12K で 48 MiB) |

→ 1 チェックポイント ≈ 112.6 MiB (文脈長に依らない)。

- 足すもの: `reset()`、`position`、`captureCheckpoint()` (GDN・conv 履歴・PLE 履歴のコピー + 位置)、`restore()` (コピーを戻し、KV・インデクサ・MTP KV のカーソルを戻す)。
- **取れる位置の要件。**要件表の P−1・U−m は prefill の途中なので、prefill を任意の位置で区切ってその場で取れること (P−1 → P は T=1 の 1 行)。`<tool_call>` の直前は decode の途中なので、
  投機ループの受理・巻き戻しが済んだステップの境目で取れること (位置はランナーが言う位置、下の落とし穴 1)。
- 検査 (`--qwen38-resume`、greedy、短文脈): 1 要求で通したときと接頭 N を持ち越したときで同じトークン列 (N を 3 つ) / 負例 (続きが違えば答えも違う) /
  P−1 に戻して 1 トークンから同じトークン列 / U−m に戻して 1 行足した列を流し、全部計算し直したときと同じトークン列 / n_max 1 と 2 の投機で終わったときの位置。
- 落とし穴 1 ([qwen35moe/41 §3-2](../qwen35moe/41-PROMPT-CACHE.md)): 記録する位置は**ランナーが言う位置**で、`prompt + 生成 − 1` の式ではない。いまの `--qwen38-generate` のループ (`Qwen38MTPCheck.swift`) は検証で kept+1 行を食って kept+1 個を出し、
  最後に出したトークンは食っていないので式が成り立つが、server のループがそのままとは限らない。
- 落とし穴 2 (Qwen3.8 固有): 投機の最後のステップで受理した `kept` 行は幹の状態に入っているが MTP KV には入っていない。継続の prefill は MTP ヘッドにその行から食わせる
  (いまは prefill 全体を MTP に通している、[10 §8-1](10-MTP-SHADOW-SPEC.md)。D-3 と絡む)。restore したら MTP KV のカーソルも戻す。
- 続きから走った答えは、全部計算し直した答えとバイト一致するとは限らない (q/41 §6: 前ターンで生成した行は decode カーネル、計算し直すと prefill カーネルが書く)。検査は短い決定的な答えで「同じトークン列」を見る。

**G-1. server 経路と INV-1 (GPU 1 日)。**

- `TsugumiServer` / `TsugumiApp` に Qwen38 の参照は 0 (2026-09-14 の grep)。生成ループ・MTP・instruct サンプラは検査 CLI (`--qwen38-generate`、`Qwen38MTPCheck.swift`) にだけあり、トークン化は Python (トークンファイル) で済ませている。
- トークナイザ: `~/LLM/Qwen3.8-Flash-Next-tokenizer/tokenizer.json` は Ornith のものと vocab 248,044・merges 247,587・pre_tokenizer / normalizer / decoder / post_processor が同一で、
  違いは added_tokens 7 個 (248070〜248076、audio / tts) だけ (2026-09-14 に json で比較)。`QwenTokenizer` に Qwen3.8 の `chat_template.jinja` と added_tokens を読ませれば足りる
  (未確認: swift-jinja がこのテンプレートの `namespace` / `messages[::-1]` / `|items` / `preserve_thinking` を通すか)。
- 重みは `.moepack` ではなく GGUF + PLE + サイドカー 2 本 (16)。manifest の `arch.family` で振り分ける既存の形 (`RealInferenceClient` は `gemma` / `ornith` の 2 択) に qwen4exp を足す。
  `capacity` (12K) と `maxBatch` (prefill チャンク、12K は 2048) は init で固定。
- **INV-1 (描き直し == 生成) を Gemma と同じ手段で閉じる。**テンプレート (2026-09-14 に読んだ) で描き直しがずれうるのは、文字列でない引数の `tojson` (メモリ `ornith-tojson-roundtrip`: `/` → `\/`)、
  `arguments|items` の順、`content|trim`。thinking off の空の思考ブロック (`<think>\n\n</think>\n\n`) は読んだ限り生成プロンプトと合う (未確認、実機の cached で見る)。生成の文法をテンプレートが描く正準形に縛り (GEN-8 の Qwen3.8 版)、文法で縛れない分はサーバー用のテンプレート変種で描き直しを生成に合わせる (DEV-12 の Qwen3.8 版)。
  検査は `QwenTurnRedrawTests` (Ornith の INV-1) と同じ形を Qwen3.8 のテンプレートで。残るずれは DEV-15 (自由形式の値の中のキー順) だけにする。
- 品質の見方は 01 §1-1 のとおり「英語の短答 + ツール呼び出し」。IQ2_XXS の品質の根拠はいま参照一致・PLE 負例 1 本・受理率 0.72〜0.95 だけなので、ここで初めて外から測れる。
- 「当たったか」はクライアントの記録と突き合わせる (メモリ `check-client-session-first`: pi は `~/.pi/agent/sessions`、Mac アプリは Inspector の cached / prefill)。

**G-2. prompt cache のセッション (CACHE-1 / 2 / 3 / 8、GPU 1 日、G-0 と G-1 の後)。**

- 判定: LCP で分岐点を出し、分岐点以前で戻れる最も新しい位置 (生きている状態か、要件表の位置で取ったチェックポイント) から続きだけ prefill。戻れる点が 1 つも無いときだけ全再処理。
- 取る位置と捨て方は要件表のとおり。**Ornith の `captureAtPromptEnd` (P で 1 本、直近 2 要求) は使わない。**
- **これは prefill を速くする話ではなく、prefill をしない話。**
- 測り方: 要件表の場面を Mac アプリ (U4 / U5) と pi (ツールループ) で 1 回ずつ踏み、`cached` / `prompt_n` / prefill s を 1 行ずつ。**同じ操作を Gemma でも流して、場面ごとに `prompt_n` を並べる**
  (Gemma の列はコードからの導出なので、基準の側も実測にする)。加えてチェックポイントを持った状態の wired 最大 / Swapouts (12K、memlog 越し)。
- 止める条件: どれかの場面で Qwen3.8 の `prompt_n` が Gemma より大きい (= 取る位置か描き直しが要件を満たしていない。速度を読まずに直す)。

基準の外 (Gemma も全部払う。進めるかはユーザー判断):

- **CACHE-9 (会話 2 本以上)**: 1 会話の保存は 12K で ≈ 826 MiB (GDN 112 + KV 576 + インデクサ 90 + MTP KV 48、導出)。RAM には無い (wired 14.5 GB)。SSD なら書き 0.8 GB・読み 6.2 GB/s で 0.13 s (導出、未確認)。Mac アプリの複数チャットの切り替え。
- **履歴の切り詰め**: サーバ側では塞げない。クライアント (Mac アプリ / pi) が 12K に収める形を、切り詰めではなく「要約して新しい会話」にするか。

**G-3. 文法つきツール強制** (Ornith [qwen35moe/40](../qwen35moe/40-MTP-GRAMMAR.md))。G-1 の後。Qwen3.8 のツール呼び出しは Ornith と同じ `<function=…><parameter=…>` の XML 形 (テンプレート) なので `QwenToolCallGrammar` を流用できる見込み (未確認)。
G-1 の INV-1 の文法 (正準形への縛り) と同じ文法の上に載せる。

## 3. 測り方の共通規則

- 腕は交互 (ABBA)、プロンプトごとに順と逆順。instruct、seed = パス番号、200 トークン。短文脈は code / tool / explain、12K は long / long2 / long3。
- **全腕でトークン列が一致することを毎走行で確かめる** (Gemma の出力不変ゲート、[mtp/30 §2](../mtp/30-M8-B-PREFETCH.md))。変わったら速度を読まない。
- 反復 3 未満のセルには数字だけ (メモリ `no-interpretation-on-n1-cells`)。
- 12K は必ず `memlog.sh` 越し (wired / file-backed / Swapouts)、短文脈は `guarded.sh`。走行後に `pgrep -x TsugumiKernelCheck` が空であること。
- `Q38_COUNT_MISS=1` (mincore) は速度の腕に入れない (11 §1)。
- 代用の経路で比を測ったら、本番の経路の絶対値と 1 度突き合わせる (メモリ `ratio-vs-base-mismatch`)。
- GPU プロセスは 1 個まで。クールダウン 20 秒。
- 1 段ごとに `docs/qwen38/` に実測を書く。番号は書く前に `ls docs/qwen38/` (14 は n_max 2)。

## 4. やらないこと

- RAM 軸: 先読み (`Q38_PREVIEW_N`) の既定化、上位 N の掃引、予測した expert の前もっての常駐 (13 §4、ユーザー判断)。B は台帳を持つだけで常駐量を増やさない (B-4 は減らす)。
- 行 y の先行 (13 §6-1、ユーザー判断で不要)。
- prefill の速度 (メモリ `dont-divert-to-prefill`)。チャンク 4096 は RAM で塞がっている (08 §4-1、09)。G-2 は prefill をしない話なので別。
- 32K (運用点は 12K、メモリ `qwen38-operating-point`)。
- 損益表で段を飛ばす・止める。

## 5. 順序のまとめ

| 順 | 段 | 推論経路の変更 | GPU の目安 | 出る実測 |
| --- | --- | --- | --- | --- |
| 1 | A (14) | 投機ループ | 進行中 | n_max 1 / 2 の tok/s・受理・非常駐 MB |
| 2 | W-1 → W-2 → W-3 | `down` のサイドカー・Q2_K の最後のブロック | 2 日 (W-1 は GPU 無し) | ビット一致・kernel→GPU ms・非常駐 MB・wired・tok/s |
| 3 | B-1 | 無し (計器) | 半日 | 台帳の当たり率 (X = 2 / 4 / 6 GB) |
| 4 | B-2 → B-3 | advise の省略 → 欠けだけ pread | 1〜3 日 | advise ms・kernel→GPU ms・tok/s |
| 5 | C-1 | hc カーネル | 半日 | hc の GPU ms (T = 1 / 2 / 3) |
| 6 | G-0 → G-1 → G-2 | reset / 任意位置の checkpoint / restore・server 結線・INV-1 (文法とテンプレート変種)・Gemma の場面表どおりの取る位置 | 2〜3 日 | `--qwen38-resume`・場面ごとの `prompt_n` (Gemma と並べる)・チェックポイントを持った wired |
| 7 | D-1 → D-2 → D-3 | 注意カーネル・影モード | 2 日 | ドラフト ms・受理率 |
| 8 | B-4 → B-5 | ビューの寿命・hit-first | 1〜2 日 | 定数 ms・wired 最大・tok/s |
| 9 | E-1 → E-2 | 選択の GPU 化 | 半日 | pre の host ms |
| 10 | C-2 | サイドカー (ユーザー判断) | 1 日 | logits の差・tok/s |
| 11 | F | 小カーネル | 半日 | 単体の GPU ms |
| 12 | G-3 | 文法つきツール強制 | 半日 | ツール呼び出しの品質 |
| — | W-4 | 無し (検定) | GPU 無し | F32 の BF16 恒等式の成否 (成り立っても進退はユーザー判断) |

W-2 / W-3 は `Qwen38Runner.swift` と `moe_ggml.metal` を触るので、A の未コミットの変更が入ってから始める。W-1 は SSD を使うので計測の無い時間に回す。

G を 6 番に置いたのは、B〜D の 12K の腕を回すたびに prefill 133 s を払っているためでもある (継続が入れば long の腕は 1 度の prefill で続けられる)。

## 6. 文書の食い違い (直すのは別作業)

- `docs/SYSTEM_DESIGN.md` が指す `OPTIMIZATION_JOURNEY.md` は存在しない。実験史は `docs/mtp/` と `docs/qwen35moe/` にある。
- `docs/OPENAI_SERVER.md` の「Ornith installs」の表 (prompt reuse: **none**、`cached_tokens` は常に 0) と `docs/MAC_APP.md` の「Ornith: 厳密な延長のみ」は、
  2026-08-23 の CACHE-8 (`QwenServerSession` のチェックポイント、`--ctx-checkpoints 2`) より前の記述。q/41 自体も §0 #2 の「全ミス」が履歴になっている。
- 同文書の「`RDADVISE` は既定 off」は [mtp/40 §2a](../mtp/40-HANDOFF.md) (2026-08-20 に mmap + residency set + `F_RDADVISE` を既定 on) と矛盾。
- Ornith の prompt cache は Gemma に届いていない: `QwenServerSession.captureAtPromptEnd` はプロンプト末尾でだけ取り、SPEC CACHE-8 の「最後の user メッセージの直前」では取らない。
  末尾のチェックポイントは同じ長さのプロンプトに使えないので、1 ターン目の再生成は全再処理。`RealInferenceClient.swift:58` のコメントは Ornith を「exact-extension prompt cache」と書いたまま。
