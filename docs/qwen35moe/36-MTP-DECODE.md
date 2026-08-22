# 36. MTP を実機で回して決着させた — 実タスクのプレフィルと生成 (実測(手元)、2026-08-22)

[30](30-MTP-HEAD-GRAFT.md) から [35](35-PREFILL-CHUNK-WIDTH.md) までの 6 本は、
**1 行も推論経路を書かずに** MTP の取り分を見積もってきた。本書はそれを止めて、
**ヘッドを GPU に載せ、幅 2 の検証パスを書き、実タスクを通しで生成して**測る。
エージェント / コーディング用途を前提に評価する。

**結論を 1 行で: 採る。ただし条件付きで** — エージェント形のターン
(短いプロンプト、構造化 / コードの出力) で **+15〜20%**、散文で −6%、
2,698 トークンの長文脈で **−16%**。既定 on にはしない。

そして [33 §3-7](33-MTP-ACCEPTANCE.md) の「+15〜29%」は**比が正しく、土俵が
間違っていた**。本書 §5 が正。

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | 動くのか | **動く。**`--qwen-mtp` でヘッドが GPU に載り、幅 2 の検証パスが回り、答えが出る。`.gturbo` は 1 バイトも作り直していない — ヘッドは 503 MB の sidecar 1 枚 (§1) |
| 2 | 速いのか | **タスク次第で符号が変わる** (§5、192 トークン × 2 反復、腕は交互)。エージェントのコード修正 **×1.146**、エージェントのツール JSON **×1.199**、コード生成 ×0.959、英語散文 ×0.941、2,698 トークンの要約 **×0.843** |
| 3 | 受理率は当たったか | **当たった。**実機の on-policy P1 は 64.7〜88.6%。[33 §2-1](33-MTP-ACCEPTANCE.md) の off-policy 予測とタスクの並びまで一致する。**プロンプト履歴を捨てた影響も予測どおり** (§2-2 の ablation が t3 で 64.92% と出し、実機が 64.7%) |
| 4 | 何が一番効いているのか | **受理率ではなく、エキスパート取得の相乗り。**1 パスが 1.8 トークンぶんの取得を 1 回で済ませるので、**素の decode が遅いタスクほど MTP が勝つ**。MTP 側の tok/s は 16.3〜19.3 とほとんど動かないのに、素の decode は 13.5〜20.5 と 1.5 倍動く (§5-2) |
| 5 | [33 §3-7](33-MTP-ACCEPTANCE.md) の +15〜29% は | **比は生きていて、土俵が違った。**あれは幅 2 / 幅 1 の比を**プレフィル経路で**測り、それを **decode の予算**に掛けた。プレフィル経路は 1 行あたり decode より重い — 実測で **1.52 倍** (§4-1)。比 (実測 1.27〜1.40) は移せても、掛ける先が違っていた。**34/35 がプレフィルの話ばかりなのは、検証パスをプレフィルで代用していたからである** |
| 6 | 費用はどこに居たか | **カーネルの選び方 1 つ。**T 行経路の密射影は `qwen_int8_qmm_f16_block` (1 出力行 = 1 スレッドが K=2048 を単独で歩く、隣接スレッドは 2048 B 離れる) に落ちていた。タイル版は **8 行未満を断る**ので、検証パスの幅はちょうどその外側。decode の SIMD/行 GEMV を T 行に広げて **GPU 60.9 → 27.4 ms/tok** (§4-3) |
| 7 | 潰した仮説 | **コマンドバッファの本数ではなかった** (§4-2)。1 パス 202 本 → 76 本にしても時間は動かない (117.9 → 116.8 ms)。数だけ見て納得しなくてよかった |
| 8 | 巻き戻しは正しいか | **正しい。**強制棄却の対照 (`TF_QWEN_MTP_FORCE_REJECT=1`) と本番の腕が**トークン列で完全一致**する (§3-2)。最初の実装は「パス前のスナップショットを復元」で、これは**確定行 0 の再帰も一緒に消していた** — 対照が degenerate ループで捕まえた (§3-1) |
| 9 | 再帰状態の代償 | **ゼロ。**[33 §3-6](33-MTP-ACCEPTANCE.md) の二重バッファをそのまま組んだ: 確定行は in-place、投機行は影へ、受理はポインタ 2 本の入れ替え。**コピー 0 回** (`snapshot=0.00ms`)。61.4 MiB の第 2 バッファだけが費用 |
| 10 | 素の decode と同じ答えが出るか | **出ない。**投機のせいではなく (#8)、**T 行カーネルと decode カーネルの数値が違う**から。a1 は 192/192 一致、t2 は 3 本目で割れて再収束する。**「MTP を入れると答えが変わる」は運用上の事実として書いておく** (§3-3) |
| 11 | プレフィルは動くか | **1 バイトも動かない。**MTP は decode ループだけを差し替える。t4 のプロンプト 2,698 トークンは両腕とも 16.8 s、TTFT も同じ。Phase 4 の検査 (`--qwen-prefill`) は負の対照 5 本込みで通る (§6-1) |
| 12 | 深さ 3 の MTP ヘッドは | **プロンプト履歴を持たせていない。**33 は教師強制なので MTP の K/V が全位置埋まっていた。実機で同じにするには fc と k/v 射影の T 行プレフィルが要る。**先に代償を測った** — 平均 P1 78.40% → 76.05% (§2-2)。2 ポイントに 2 本目の prefill 経路は釣り合わない |
| 13 | 文法 / ツール呼び出しと併用できるか | ~~**できない。**~~ **[40](40-MTP-GRAMMAR.md) で併用できるようになった** (2026-08-22)。原因は「2 行版が `normalizedHidden` を埋めない」ではなく、**再畳み込みがどの行を読むか指定できなかった**こと。`encodeMaskedRescore` に行を渡すだけで済み、新カーネルは 0 本 |
| 14 | 次に何をすれば勝ち幅が広がるか | **残る 1.21 倍**。幅 1 を T 行経路に流すと素の decode の 1.21 倍かかる (§4-4)。内訳は GPU +2.7 ms / ホスト +11 ms で、ホスト側は層ごとの route grouping、GPU 側は prefill attention と router。ここを decode 級にすれば全タスクで勝つ側に回る (§7) |
| 15 | 要ったもの | Metal カーネル 2 本 (`dequant_int8_gemv_rows_simd`、`qwen_lm_head_greedy_int8_rows_chunk_raw_multi`) + BF16 GEMV 1 本。Swift 3 ファイル。Python 2 本。**`.gturbo` の repack は 0 回** |

---

## 1. ヘッドを実機に載せる — sidecar 1 枚

`.gturbo` に MTP は入っていない。`RepackPlanner.classify` が
`language_model.mtp.` を `.excludedDraft` に落とすからで、
[30 §6-6](30-MTP-HEAD-GRAFT.md) の「pack を作り直す理由が無い」はそのまま生きて
いる。**503 MB の sidecar を 1 枚足す**のが答えだった。

```bash
scratch/mtp-venv/bin/python Scripts/qwen35/build_mtp_sidecar.py \
    ~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa-baked \
    --out ~/LLM/ornith-mtp-head --gturbo scratch/ornith-oq4e-g64.gturbo
```

| 出るもの | 中身 |
| --- | --- |
| `mtp_core.bin` | **50,167,808 B** / 17 エントリ = 平文 9 + 8bit 8 本 (weights/scales/biases で 33 テンソル)。`MTLBuffer` 1 本で丸ごと `bytesNoCopy` するのでページ倍数に詰めてある |
| `mtp_experts.bin` | **452,984,832 B** = 256 × **1,769,472**。**本体の `packed_experts` とバイト並びが同じ** (gate w/s/b, up w/s/b, down w/s/b、ページまで padding) |
| `mtp_head.json` | 形・量子化・offset・sha256 |

- **`expertStride` が本体と一致することを書き出し時に検算する** (`--gturbo`)。
  MoE のカーネルは 8 本の blob に**同じ** `MoEExpertOffsets` を当てるので、
  並びが違う sidecar は「もっともらしいバイトを違う意味で読む」— それは
  流暢な出力になって現れるので、実行時にも `checkExpertLayout` で照合する。
- エキスパートは `MAP_SHARED` + エキスパートごとの `bytesNoCopy` バッファ 256 本、
  **residency set は 1 個を load 時に 1 回 commit するだけ**。本体は 256 中 32 本
  しかスロットに載らないので層ごと・トークンごとに付け替えるが
  ([27 §9](27-PHASE6-THROUGHPUT.md))、ヘッドは 256 本とも張りっぱなしでよい。
- 書き出したバイトはチェックポイントと**照合済み** (`fc` / `q_proj` の weights と
  scales、`shared_expert_gate` の biases、expert 0 / 5 / 255 の gate と down が
  バイト一致)。

## 2. GPU 上の MTP ブロック

`Scripts/qwen35/mtp_acceptance.py` の `MTPHead.step` と同じ順序で、本体の
カーネルをそのまま使う (`Attention` / `MoE` / `RMSNorm` / 逆量子化 GEMV /
埋め込み / 508 MB の `lm_head`)。**新しく要ったのは BF16 GEMV 1 本だけ** —
`mtp.fc` (2048×4096) が差し替え元の shisa で BF16 のままだからで、router の
BF16 GEMV は FP32 の logit を書くので次の RMSNorm に渡せない。

### 2-1. K/V は生成ぶんしか持たない

[33](33-MTP-ACCEPTANCE.md) は教師強制なので、MTP の K/V は**プロンプトを含む
全位置**が埋まった因果キャッシュだった。実機で同じにするには、プロンプト全体に
`fc` と k/v 射影を T 行で通す **2 本目の prefill 経路**が要る。

### 2-2. その代償を先に測った (**実測(手元)**)

`Scripts/qwen35/mtp_history_ablation.py`。**モデルは 1 度も走らせていない** —
[33](33-MTP-ACCEPTANCE.md) が残した hidden の npz に、MTP の注意の可視範囲を
変えながらヘッドだけ当て直す。深さ 1、post_norm。

| タスク | `full` (33 と同じ) | `gen` (生成開始から = 実機) | `last` (履歴なし) |
| --- | ---: | ---: | ---: |
| t1 日本語の説明 | 76.96% | 76.96% | 71.73% |
| t2 コード | 87.43% | 85.34% | 81.68% |
| t3 英語の散文 | 69.11% | 64.92% | 68.06% |
| t4 2,698 tok の要約 | 80.10% | 76.96% | 75.39% |
| **平均** | **78.40%** | **76.05%** | **74.22%** |

**捨てて 2.35 ポイント。**2 本目の prefill 経路を書く理由にはならない。
実機は `gen` で作り、**その予測が §5 で当たった**。

### 2-3. 追いつきの行は k/v しか要らない

1 パスで 2 トークン受理すると、間の位置でヘッドは走っていない。その位置が
キャッシュに負っているのは **`(k, v)` だけ**である — 深さ 1 の鎖は本体 hidden を
食べるので、飛ばした位置の**出力**は誰も読まない
([33 §2-4](33-MTP-ACCEPTANCE.md) の鎖の話は深さ 2 以降で、幅 2 はそこを使わない)。
なので追いつき行は `fc` → `input_layernorm` → q/k/v → epilogue で終わり、
attention も MoE も 508 MB のヘッドも走らない。

---

## 3. 正しさ

### 3-1. 対照が設計の誤りを捕まえた

最初の巻き戻しは「パス前に再帰状態を blit で退避し、棄却時に復元」だった
([33 §3-6](33-MTP-ACCEPTANCE.md) の (c))。`TF_QWEN_MTP_FORCE_REJECT=1`
(draft を必ず外す腕) を回すと **"Swift で、Swift で、Swift で、" と degenerate
ループ**になった。

**理由: 復元しすぎていた。**幅 2 のパスは行 0 が確定トークン、行 1 が投機である。
残すべきは**行 0 のあとの状態**で、これはパス前でもパス後でもない。
パス前のスナップショットに戻すと行 0 の再帰も消える。

[33 §3-6](33-MTP-ACCEPTANCE.md) の (a) に組み直した:

```
行 0   stateIn = A, stateOut = A     (in-place、確定ぶんを進める)
行 1   stateIn = A, stateOut = B     (投機ぶんは影へ)
受理   A と B を入れ替える (ポインタ 2 本)
棄却   何もしない — A は既に正しい
```

`qwen_delta_rule` も `qwen_delta_qkv_prepare` も Phase 2 から `stateIn` と
`stateOut` を別引数で取っているので、**カーネルは 1 行も触っていない**。
コピーは 0 回で、計器も `snapshot=0.00ms` と出る。conv の窓も同じ形。

### 3-2. 投機は中立である (**実測(手元)**)

同じプロンプト・同じ 16 トークンで:

| 腕 | トークン列 |
| --- | --- |
| `--qwen-mtp` (P1 60%、a=1.600) | `[2, 22929, 220, 171323, 97728, …]` |
| `--qwen-mtp` + `TF_QWEN_MTP_FORCE_REJECT=1` | **同一** |

**受理しても棄却しても同じ列が出る。**巻き戻しに漏れは無い。

### 3-3. ただし素の decode とは同じにならない

同じ 16 トークンで素の decode は 4 本目が `170733`、MTP 側は `171323`。
その 1 本だけ違って**すぐ再収束する** (16 中 15 一致)。

**原因は投機ではなく経路である。**検証パスは T 行カーネル (prefill attention、
block router、per-pair MoE) で走り、decode の 1 行カーネルとは加算順が違う。
接戦の argmax がひっくり返る。192 トークンで見ると:

| タスク | 素の decode と一致 | 最初に割れた位置 |
| --- | ---: | ---: |
| a1 エージェントのコード修正 | **192 / 192** | — |
| t3 英語の散文 | 45 / 192 | 44 |
| t4 要約 | 34 / 192 | 33 |
| t2 コード | 34 / 192 | 3 |

**どちらが正しいという話ではない** ([33 §4](33-MTP-ACCEPTANCE.md) が CPU 参照器と
ランタイムで 184〜190/192 だったのと同じ性質) が、**`--qwen-mtp` を足すと答えが
変わる**のは運用上の事実である。出力を固定したいなら §7 #1 が要る。

---

## 4. 費用 — どこに居たか (**実測(手元)**)

t2 コード、16 トークン、幅 2、per-pair。素の decode の基準は同条件で
**16.86 tok/s / GPU 27.19 ms/tok**。

### 4-1. 幅 1 の対照が土俵の差を出した

`TF_QWEN_MTP_NO_DRAFT=1` は draft を作らず、**1 行を T 行経路に流す**腕である。
これが素の decode と同じ費用なら、その上は全部「2 行目の値段」になる。

| 腕 | 1 パス | GPU/tok | tok/s |
| --- | ---: | ---: | ---: |
| 素の decode | (59.4 ms/tok) | 27.19 | **16.86** |
| T 行経路・幅 1 | **90.20 ms** | 50.35 | 11.82 |
| T 行経路・幅 2 | 117.94 ms | 37.89 | 12.74 |

**幅 1 が既に 1.52 倍。**幅 2 / 幅 1 = **1.307** で、
[33 §3-7](33-MTP-ACCEPTANCE.md) の 1.27〜1.30 と一致する。
**比は正しかった。掛ける先が違っていた。**

### 4-2. 潰した仮説 — コマンドバッファの本数

1 パス 202 本 (decode は 114 本/トークン)。層ごとに tile / reduce / residual を
別のバッファに分けているためで、これが差だと踏んだ。層の tile・reduce・residual
を 1 本に畳んで **202 → 76 本**にした。

| | 1 パス | GPU/tok |
| --- | ---: | ---: |
| 幅 2、202 本 | 117.94 ms | 37.89 |
| 幅 2、**76 本** | 116.81 ms | 37.44 |

**動かない。**余分な 88 本はただ同然だった。畳んだ形は残してあるが
(`compactChunkCommandBuffers`、検証パスだけ・プロンプトは従来どおり)、
**これは費用の説明ではない。**

### 4-3. 犯人 — 密射影のカーネルが 8 行未満で落ちる

`QwenPrefillInt8QMM.usesTiledPath` は `t >= 8` を要求する。検証パスは 1〜2 行
なので**必ずタイル版を外れ**、`qwen_int8_qmm_f16_block` に落ちる —
**1 出力行 = 1 スレッドが K=2048 バイトを単独で歩く**カーネルで、隣り合う
スレッドは 2048 B 離れたアドレスを読む。coalesce しない。

decode 側の `dequant_int8_gemv_simd` は 1 行を SIMD group 32 レーンで分担する。
**それを T 行に広げた** (`dequant_int8_gemv_rows_simd`、4bit 版は Gemma 用に
[mtp/16](../mtp/16-M4.5-PLAN.md) で既にあった)。重みは 1 回読んで T 個の活性に
使う。T=1 では decode の GEMV と**ビット一致**する。

| | 1 パス | GPU/tok | tok/s |
| --- | ---: | ---: | ---: |
| 幅 1、QMM scalar | 91.11 ms | 50.67 | 11.70 |
| 幅 1、**行 GEMV** | **71.97 ms** | 33.04 | **14.80** |
| 幅 2、QMM scalar | 116.81 ms | 37.44 | 12.97 |
| 幅 2、**行 GEMV** | **100.96 ms** | 27.36 | **14.88** |

**GPU が 60.9 → 27.4 ms/tok。**素の decode の 27.19 とほぼ同じところまで来た。

### 4-4. 残る 1.21 倍

幅 1 の T 行経路は 71.97 ms/パス、素の decode は 59.4 ms/トークン。
**まだ 1.21 倍。**内訳 (計器の io は T 行経路では prefill 側に計上されるので
ホストに含まれる):

| | GPU | ホスト + 取得 |
| --- | ---: | ---: |
| 素の decode | 27.19 | 32.14 |
| T 行経路・幅 1 | 33.04 | 34.52 |

GPU +2.7 ms は prefill attention と block router、ホスト +11 ms は層ごとの
route grouping (`readPrefillRoutes` が層ごとにペアを組んで並べ替える) が主犯と
読める。**どちらも未確認**で、§7 の測定対象である。

---

## 5. 実タスクの通し (**実測(手元)**)

`bench/qwen35/mtp_ab.sh`。192 トークン、temp 0、**腕は交互、回ごとに順も入れ替え**
([32 §5-1](32-NVMAI-ADOPT.md) の作法)、クールダウン 10 秒、2 反復の中央値。
運用点 (32 スロット、mmap、pipeline on)。

### 5-1. 表

| タスク | 素の decode | `--qwen-mtp` | 倍率 | P1 | a |
| --- | ---: | ---: | ---: | ---: | ---: |
| **a1 エージェントのコード修正** (Swift のバグ指摘 + 修正 + テスト) | 15.101 | **17.301** | **×1.146** | 87.3% | 1.882 |
| **a2 エージェントのツール JSON** (次の行動を JSON だけで) | 13.881 | **16.645** | **×1.199** | 88.6% | 1.886 |
| t2 コード生成 | 19.593 | 18.789 | ×0.959 | 81.1% | 1.811 |
| t3 英語の散文 | 20.457 | 19.256 | ×0.941 | 64.7% | 1.655 |
| t4 2,698 tok の要約 | 15.122 | 12.743 | **×0.843** | 74.5% | 1.745 |

単位は tok/s (decode のみ)。反復間の振れは小さい (a1 で ×1.145 / ×1.197、
t4 で ×0.829 / ×0.843)。

**受理率は [33 §2-1](33-MTP-ACCEPTANCE.md) の off-policy 予測とよく合う。**
§2-2 の `gen` 腕の予測と実機:

| | t2 | t3 | t4 |
| --- | ---: | ---: | ---: |
| ablation の予測 (off-policy) | 85.34% | 64.92% | 76.96% |
| **実機 (on-policy、4bit)** | **81.1%** | **64.7%** | **74.5%** |

### 5-2. 勝ち負けを決めているのは受理率ではない

表を tok/s の**振れ**で見ると、機構が出る:

| | 最小 | 最大 | 幅 |
| --- | ---: | ---: | ---: |
| 素の decode | 13.88 | 20.46 | **×1.47** |
| `--qwen-mtp` | 12.74 | 19.26 | ×1.51 |
| `--qwen-mtp` (t4 を除く) | 16.28 | 19.26 | **×1.18** |

**t4 を除くと MTP 側はほとんど動かない。**素の decode は 1.47 倍動く。
理由は取得の相乗りである — 1 パスがエキスパートを 1 回取って 1.8 トークンぶんに
使うので、**エキスパート局所性の悪いタスクほど MTP が勝つ**。a1 / a2 は素の
decode が遅い側 (13.9〜15.1 tok/s) で、そこで 15〜20% 取れている。t3 は素の
decode が一番速く (20.5)、相乗りする余地が小さい。

**t4 だけは別の理由で負ける。**プロンプト 2,698 トークンなので検証パスの attention
が 2,900 位置 × 2 行になり、1 パス 130〜134 ms まで伸びる (t2 は 90 ms)。
長文脈では T 行 attention の非効率がそのまま乗る。

### 5-3. プレフィルは動いていない

| | 素の decode | `--qwen-mtp` |
| --- | ---: | ---: |
| t4 プロンプト 2,698 tok の prefill | 16.768 / 16.859 s | 16.848 / 16.809 s |
| t4 TTFT | 16.769 s | 16.849 s |

**MTP は decode ループだけを差し替える。**エージェント用途で TTFT を決めるのは
プロンプトのプレフィルで、そこは 1 ミリ秒も動かない。

### 5-4. 出力の中身

a1 は両腕で**トークンまで完全一致** (バグ原因 = `size == 0` の無限ループ、
`guard size > 0` を足した修正、テスト 3 本)。a2 は割れるが、どちらも妥当な JSON:

| 腕 | `tool` | `path` |
| --- | --- | --- |
| 素の decode | `read` | `.` |
| `--qwen-mtp` | `read_file` | `config/dev.yaml` |

**品質の差は見ていない** (n=1、採点していない)。ここで言えるのは「壊れていない」
だけである。

---

## 6. 既存経路に触れていないことの検算

### 6-1. Phase 4 の検査

```
PASS  41 tokens, every one equal to the float32 reference
PASS  chunk 8 (3 chunks) — the same 41 tokens
PASS  routed experts on the per-pair path, chunk 512 (1 chunks) — the same 41 tokens
PASS  routed experts on the per-pair path, chunk 8 (3 chunks) — the same 41 tokens
  negative controls — each must disagree within 16 tokens:  5/5 PASS
```

`runChunkLayers` の切り出しと `tail` の畳み込みはプロンプト経路を変えていない
(`compactChunkCommandBuffers` も行 GEMV も検証パスだけで on になる)。

### 6-2. `swift test`

1,350 テスト / 206 スイート。**失敗 24 件はすべて
`remote HTTP 404: https://hf.test/…`** — vision / draft の install fixture が
ネットワークを要求するもので、本書の変更とは無関係 (Qwen の decode も prefill も
通っている)。

---

## 7. 残る未確認と、次の一手

1. **出力が素の decode と変わる** (§3-3)。検証パスの行 0 を decode と
   ビット一致させるには、prefill attention / block router / per-pair MoE の
   加算順を decode 側に揃える必要がある。行 GEMV は既に一致している。
2. ~~**文法・ツール呼び出しと併用できない**~~ → **[40](40-MTP-GRAMMAR.md) で
   閉じた** (2026-08-22)。ここに書いた診断は外れていた: 2 行の
   `normalizedHidden` は検証パスのスクラッチに**ちゃんとある**。無かったのは
   `encodeMaskedRescore` に**どの行かを渡す口**だけで、足したのは引数 2 つ、
   新しい Metal カーネルは 0 本である。
3. **残る 1.21 倍** (§4-4)。ホスト側 +11 ms (層ごとの route grouping) と
   GPU +2.7 ms (prefill attention / block router)。ここが decode 級になれば
   t2 / t3 も勝ち側に回る計算になる。
4. **長文脈の T 行 attention** (§5-2)。t4 が ×0.843 なのはここ。2,900 位置 ×
   2 行の attention を decode の 1 行カーネル 2 回と比べていない。
5. **深さ 3 / 幅 3 以上は測っていない。**[33 §3-1](33-MTP-ACCEPTANCE.md) の
   和集合の議論から k=2 に決め打ちしてある。
6. **サーバー経路には出していない。**`--qwen-mtp` は CLI だけ。

**運用の推奨 (ユーザー判断):** 既定は off のまま、**エージェント / コーディングの
ターンで明示的に on** にする。プロンプトが 1,000 トークンを超えるターンでは
off が速い。#2 が片付くまでツール呼び出しとは併用できない。

---

## 8. コードと文書の根拠 (2026-08-22 に確認した現物)

| 事実 | 場所 |
| --- | --- |
| sidecar の生成と本体との照合 | `Scripts/qwen35/build_mtp_sidecar.py` → `~/LLM/ornith-mtp-head/` (**実測(手元)**) |
| 履歴 ablation (full / gen / last) | `Scripts/qwen35/mtp_history_ablation.py` → `scratch/qwen35/mtp-a/*.history.json` (**実測(手元)**、モデル再実行なし) |
| ヘッドの GPU 実装 | `Sources/TurboFieldfare/MTP/QwenMTPSidecar.swift` / `QwenMTPDrafter.swift` |
| 幅 2 のループ・巻き戻し・対照 2 本 | `Sources/TurboFieldfare/Runtime/Inference/QwenSpeculativeDecode.swift` |
| 再帰状態の影とポインタ交換 | `Sources/TurboFieldfare/Runtime/KVCache/RecurrentStateManager.swift` (`ensureShadow` / `adoptShadow`) + `QwenPrefill.encodeSplitRecurrent` |
| T 行の密射影 | `Sources/TurboFieldfare/Metal/Quant/dequant_int8.metal` `dequant_int8_gemv_rows_simd` + `DequantInt8GEMV.encodeRows` |
| 2 行を 1 回の 508 MB 読みで採点する頭 | `Sources/TurboFieldfare/Metal/Qwen/qwen.metal` `qwen_lm_head_greedy_int8_rows_chunk_raw_multi` + `QwenLMHeadChainInt8.encodeGreedyDecodeRows` ([33 §3-8](33-MTP-ACCEPTANCE.md) の測定 3) |
| `mtp.fc` 用の BF16 GEMV | 同 `qwen_bf16_gemv_f16` + `QwenKernels.encodeBF16GEMV` |
| A/B の生データ | `bench/qwen35/mtp_ab.sh` → `scratch/qwen35/mtp-ab/*.footer` / `*.json`、集計は `bench/qwen35/mtp_ab_report.py` |
| エージェント形の 2 タスク | `bench/qwen35/a1-agent-edit.json` / `a2-agent-tool.json` |
| Phase 4 の検査 | `TurboFieldfareKernelCheck --qwen-prefill scratch/ornith-oq4e-g64.gturbo` |
| 予測の出どころ | [33 §2-1 / §3-6 / §3-7 / §3-8](33-MTP-ACCEPTANCE.md)、[30 §6](30-MTP-HEAD-GRAFT.md) |
