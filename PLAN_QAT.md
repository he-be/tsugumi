# PLAN_QAT — lattice-aligned QAT チェックポイント (`qat-q4_0-mlx-aligned`) の導入

作成: 2026-08-17 / 更新: 2026-08-16 (Phase A+B 完了、実測を反映)
M3 Pro 18GB / macOS 15.7.5 / `macos15-support` ブランチ
入力: `docs/investigations/QAT_MTP_CHECKPOINT_ANALYSIS.md` (Candidate 2 の分析)、
`PLAN.md` / `RESULTS.md` (Phase 0-2 の実測)、リモートリポジトリの直接確認、
**DL 済みスナップショットと本リポジトリのソース読み** (**実測**、§1-3 / §3-1-a)
方針: **64 スロットで実験する。メモリガードに弾かれたら 48 に落とす。**

表記は PLAN.md と同じ: **実測** / **導出** / **未確認**

### 更新履歴 — 設計レビューで覆った点

| # | 当初の記述 | 訂正 |
| --- | --- | --- |
| 1 | §3-1「カーネル本体のロジックは 1 行も変わらない / 定数置換のみ」 | **誤り。**decode の 4 箇所がブロック分割を直書きしており、定数置換だけでは OOB + スケール不整合を起こす。§3-1-a |
| 6 | §3-1-a「対象は 4 箇所」 | **過少。実際は 7 箇所。**INT8 ルーター (§3-1-a-2) と INT8 の GEMV / shared expert (§3-1-a-3) が漏れていた。後者 2 つは**現行ピンが毎トークン通る**経路 |
| 2 | §3-1 function constant の却下理由「PSO 二重管理が必要」 | **現状と不一致。**PSO キャッシュは既に定数をキーにしている。§3-1-c |
| 3 | §2 resident ≈ 1.2 GB (「shared 4bit 化とスケール増が相殺」) | **実測 1.51 GB。**相殺していない。結論 (64 スロット可) は不変 |
| 4 | §5 の検証は旧モデル回帰 + 実機の目視 | **group 32 のバグを検出できない。**§5-A の数値検証を Phase A の出口条件に追加 |
| 5 | §4 のファイル一覧 | `tensorops.metal` / `MPPPrefillInt4QMM` / `validateQuant` の `scheme` 検査が漏れ |

---

## 進捗 (2026-08-16 時点)

| Phase | 状態 | 結果 |
| --- | --- | --- |
| A0 数値検証ハーネス | **完了** | `TsugumiKernelCheck`。commit `4cd8c69` |
| A カーネルの group パラメータ化 | **完了** | §3-1-a の 4 箇所を修正。group 64 の数値は修正前と同値。commit `4cd8c69` |
| A 既存モデル回帰 (§5-0) | **完了・合格** | 3 本とも ±1% 以内 (下記 §5-0 の実測) |
| B bf16 ルーター | **完了** | ハーネスは 20/20 PASS。§3-1-a-2 / §3-1-a-3 で group 幾何のバグを計 3 箇所追加で発見・修正。既存モデルの実機出力は文字単位で不変 |
| C repacker のローカル入力 | **完了** | `--source-snapshot`。commit `7a6b5ff`。repack 40.6s、§2 の試算と厳密一致 |
| D インストールと受入 | **完了・採用** | 既定スロットを 64→48 に変更。`RESULTS_QAT.md` |

### 結論 (2026-08-16): **採用。既定スロットは 48**

詳細は `RESULTS_QAT.md`。要点だけ:

| ゲート | 判定 | 実測 |
| --- | --- | --- |
| 品質 | **合格** | 三次方程式は QAT が明確に上 (閉形式到達 2/3 対 0/3)、俳句は引き分け |
| 性能 | **不合格** | −17〜−25%。ただし**ゲート指標そのものが誤り** (下記 3) |
| メモリ | 合格 | 48 スロットで peak **6.0 GB** (64 では 7.8 GB) |
| 目視 / 検証モード | 合格 | |

採用はゲートを満たしたからではなく、**受入指標が誤っていたとユーザーが判断した**
ため。既定ピンの切り替えは未実施で、独立に判断する (§0-1、§8)。

**Phase A/B/C の実装はすべて残す。**判定が成立したのは group 32 カーネルと
BF16 ルーターが正しく動いたからで、§5-A は 20/20 PASS のまま。

### 本 PLAN が間違えていた点 (すべて実測で判明)

1. **§5-2-2 の品質ゲートは、1 回目の測定で結論が反転していた。**
   `GFTokenizer.applyChatTemplate` がテンプレートの `enable_thinking = false` を
   ハードコードしており、モーラ数を数える課題で
   「no-think 指示を無視して推論するベースライン」対「指示に従って即答する QAT」
   を比べていた。**品質ではなくコンプライアンスを測っていた。**
   `--thinking on|off` を追加して測り直し、結論が逆転した。
   → **モデルを比較する前に、ハーネスが何を暗黙に固定しているかを列挙すべきだった。**
2. **§5-2-3 のゲート指標 (tok/s) が誤っていた。**測るべきは正解到達までの時間で、
   答えに出なければ再生成 = −100% であり、トークン単価の −20% は霞む。
   実際、同一予算で閉形式に到達したのは QAT 2/3 対ベースライン 0/3 で、
   **tok/s が速いほうが答えに遅い**。
   加えて生成長がモデルで変わるため、384 tok 前提の `haiku tg` は評価すらできなかった。
3. **性能低下の原因の内訳を取り違えていた。**当初「decode ヒット率の低下」を
   主因と読んだが、96 スロットまで積むと QAT のヒット率はベースラインを上回り
   (97.7% 対 95.1%)、io も同等以下になるのに **tok/s は動かない**。
   損失は §2 が**導出**していた group 32 の expert GEMV そのもので、
   **メモリを積んでも買い戻せない。**
4. **peak は安全側に外れた。**§2 の**導出** 9.3 GB に対し 64 スロットで 7.8 GB。
5. **アラビア語混入の観察項目は検証不能だった。**temp 1.0 では両モデルとも 0 件で、
   混入は temp 0 のベースラインでのみ起きる。
6. **スロットの knee はモデルで違う。**ベースラインは 64 だが QAT は 48。
   group 32 で expert GEMV が重い分だけ隠せる I/O が増え、
   **20% を奪っているのと同じ性質が小さいキャッシュへの耐性を買っている。**

### 品質判定の射程 (**未確認**を明示)

言えるのは「今回測った範囲では、品質面で非 QAT を積極的に選ぶ理由は
見つからなかった」まで。標本は **2 課題 × 3 seed** で、優位が出たのは三次方程式
1 問、俳句は引き分け。**QAT の shared expert 4bit (現行ピンは 8bit) の影響は
狙って検証していない。**長文脈・コード・翻訳・事実想起も未測定。

### 再開手順

```bash
swift build -c release
./.build/release/TsugumiKernelCheck          # 20/20 PASS を確認
```

### 副産物: `bench.sh` の既存バグを修正した (**実測**)

`cmd_ja` は `MAXNEW="${MAXNEW:-384}"` と書いてあるが、56 行目で先に
`MAXNEW="${MAXNEW:-128}"` が埋めるので**効かない**。結果 `./bench.sh ja` は
黙って 128 tok で回り、RESULTS §3-4 (384 tok) と比較できない行を
`bench/results.tsv` に追記していた。`MAXNEW_EXPLICIT` を見る形に直した
(`ja` と `loopcheck` の 2 箇所)。**私の変更とは無関係な、以前からのバグ。**

参考: 128 tok で回った回帰値 (3 回中央値、比較対象なし)。
haiku 27.82 / math 24.85 / story 28.09 tok/s、peak 6.93-7.10 GB、
decode hit 96.1-98.5%。ヒット率と peak は §3-4 の 384 tok 版と整合する。

### もう 1 つの落とし穴: ベースラインは greedy で測られている (**実測**)

RESULTS §3-4 の 30.84 / 31.31 / 28.42 tok/s は `--temperature 0` の表
(§3-4 冒頭の注記)。一方 `bench.sh` の既定は §3-6 以降 **temp 1.0**。
±4% の判定幅は「温度差は run 間の振れの中」という §3-4 の但し書きと同じ大きさ
なので、**温度をベースラインに合わせないと判定が交絡する**。
§5-0 は `TEMP=0 MAXNEW=384 ./bench.sh ja` で測った。

---

## 0. 決定事項 (ユーザー指示 + 本 PLAN 内での確定)

| 項目 | 決定 |
| --- | --- |
| 採用チェックポイント | `mlx-community/gemma-4-26B-A4B-it-qat-q4_0-mlx-aligned` (調査資料 Candidate 2) |
| ソース取得方法 | **配布のままローカル DL 済み** (`scratch/qat-aligned-snapshot/`、§1-2)。repack はローカルスナップショットから行う |
| エキスパートキャッシュスロット | **64 を基本**。`ExpertCacheBudget` に弾かれた場合のみ 48 |
| 既存ピン (`gemma-4-26b-a4b-it-4bit`) | **残す**。repacker はローカルスナップショット入力モードを追加するだけ |
| 出力先 | `scratch/gemma4-qat.moepack` (既存 `gemma4.moepack` と共存。受け入れ後に入れ替えを判断) |
| 品質と性能の両方が受入条件 | 性能 ≥ ベースラインの −10%、メモリ < 12 GB、品質は寿司俳句等で目視評価 |
| **group 64/32 の両対応** | **受入後も残す** (ユーザー判断、2026-08-16)。下記 §0-1 |

### 0-1. 両対応を残す判断 (2026-08-16)

「QAT が成功したら既定モデルを使う理由がなくなるのに、両対応は必要か」を
検討した結果、**残す**で確定。根拠は汎用性ではなく、次の 2 点:

1. **受入判定が比較判定だから。** §5-2 の品質・性能ゲートはどちらも
   ベースラインとの比較で、PLAN §6 はサーマルドリフト約 4% のため
   **インターリーブを要求する**。1 つのバイナリが `--model` で両モデルを
   受け付けないと、20 秒クールダウンを挟んだ交互測定そのものが成立しない。
   ビルドし直しながら交互に測るのはプロトコル違反。
2. **測定器の校正になるから。** Phase A で Metal 4 箇所を書き換えたとき、
   **group 64 の相対誤差が修正前と完全同値**であることが「意味を変えていない」
   唯一の直接証拠だった。group 32 だけで書いていたら、CPU 参照と GPU が
   同じ思い込みで一致している可能性を排除できない。

コストの実測: Phase A の +315 行のうち**両対応に固有なのは約 110 行 (15%)**。
内訳は `MetalContext` の遅延コンパイル + define 注入 (約 70)、6 つの `.metal` の
`#ifndef` フォールバック (18)、Mac アプリのコンテキスト張り替え (約 12)、
`Model.affineGroupSize` + runner での設定 (約 8)。
**Metal 4 箇所の書き換え・MPP のゲート・precondition のモデル値化は
group 32 単独でも同じ編集量**なので、両対応の追加コストではない。

含意: 両対応が残るので、**受入後に repacker の既定ピンを QAT に切り替えても
ランタイム側は何も壊れない** (group 64 のモデルもそのまま動く)。
ピン切り替えは独立に判断してよい。

---

## 1. ソース側の事実確認 (**実測**、2026-08-17 時点の pinned revision)

revision `745a97a754ed4b7713163c7d0e9c11da41809e0c` (2026-07-06、調査資料と同一)。
`config.json` / `model.safetensors.index.json` を直接取得して確認した。

| 項目 | 値 | 備考 |
| --- | --- | --- |
| 量子化 | bits **4** / group_size **32** / mode affine (ベース、オーバーライドなし) | `quantization` と `quantization_config` 同値 |
| `router.proj.weight` | **BF16 非量子化** (`.scales`/`.biases` 同伴なし) | 30 層すべて |
| shared expert (`mlp.*`) | **4bit** / group 32 | 現行ピンは 8bit。`SharedExpertInt4` は既存 |
| embedding / attention / routed | 4bit / group 32 | |
| テンソル名集合 | 現行ピンの部分集合 (vision 系なし、text のみ 1279 テンソル) | 調查資料どおり |
| `v_proj` | 25 層分のみ (full attention 層はなし) | 現行と同じ K=V 構造 |
| `tokenizer.json` | 取得済み (インストール時に同一性を再確認) | |
| index SHA-256 | `7dbbeef0345505798abcf0ac54434116a48c2f1e7aad828071c17a7a871adfe7` | 指紋としてピンする |

`config.json` の `quantization` はオーバーライドテーブルを持たない
(`bits`/`group_size`/`mode` のみ)。`IndexLoader` のオーバーライド検証
(`group_size != base` で拒否) には引っかからない。

### 1-2. ローカルスナップショット取得 (**実測**、完了)

uv + venv + hf CLI (`huggingface_hub==1.27.0`、`scratch/hf-venv/`) で
revision をピンして配布のまま取得済み:

```
scratch/qat-aligned-snapshot/   15 GB
  model-0000{1,2,3}-of-00003.safetensors  (計 15.28 GB)
  model.safetensors.index.json  SHA-256 = 7dbbeef0… (ピンと一致)
  config.json / generation_config.json / tokenizer.json /
  tokenizer_config.json / chat_template.jinja / README.md / conversion/
```

- index の `weight_map` が参照する shard は 3 本とも完全取得 (欠けなし、エラー 0)
- これにより §3-3 の遠隔ストリーミング dual-source 化は不要になり、
  **ローカルスナップショットから repack するモード** に簡略化した
- ダウンロード分のディスク 15.8 GB は消費済み。repack 出力ぶん約 15.5 GB は別途必要
  (空き 535 GB **実測** → 問題なし)

### 1-3. ローカル実物での再確認 (**実測**、2026-08-16)

§1 の表は当初リモートの `config.json` / index から取ったものだった。DL 完了後、
**手元の safetensors ヘッダを直接読んで**全項目を再確認した。

```
index SHA-256                     = 7dbbeef0…  ← §1 のピンと一致 (shasum で確認)
config.quantization               = {group_size: 32, bits: 4, mode: affine}  (キーは 3 個のみ)
config.quantization_config        = 同値
テンソル数                        = 1279 (vision 系なし、v_proj は 25 層のみ)
```

| テンソル | dtype | shape | 判定 |
| --- | --- | --- | --- |
| `layers.N.router.proj.weight` | **BF16** | [128, 2816] | 非量子化。`.scales`/`.biases` なし |
| `layers.N.router.scale` | BF16 | [2816] | effective_scale 相当 |
| `layers.N.router.per_expert_scale` | BF16 | [128] | |
| `layers.N.mlp.gate_proj.weight` | U32 | [2112, 352] | 352×8 = 2816 → **4bit** |
| `layers.N.mlp.gate_proj.scales` | BF16 | [2112, 88] | 2816/88 = **32** |
| `experts.switch_glu.gate_proj.weight` | U32 | [128, 704, 352] | 4bit |
| `experts.switch_glu.gate_proj.scales` | BF16 | [128, 704, 88] | group 32 |
| `experts.switch_glu.down_proj.weight` | U32 | [128, 2816, 88] | 88×8 = 704 → 4bit |
| `experts.switch_glu.down_proj.scales` | BF16 | [128, 2816, 22] | 704/22 = 32 |
| `embed_tokens.weight` / `.scales` | U32 / BF16 | [262144, 352] / [262144, 88] | 4bit / group 32 |

→ **§1 の表は全項目が実物と一致。**shared expert (`mlp.*`) が 4bit であることも
確認済みで、`Model.swift:37` (`sharedExpertWeightBits`) → `SharedExpertInt4.swift:106`
の既存経路がそのまま使える (**実測**、追加作業ゼロ)。

## 2. メモリとディスクの試算 (**導出**)

group 32 の affine 格納は重量あたり約 5.0 bit (group 64 は約 4.5 bit)。

expertStride は**実物のヘッダから逆算して厳密に一致した** (**実測**):

```
gate  = 704 rows × 1408 B + scales 704×88×2B + biases 同 = 1,239,040 B
up    = 同                                                 = 1,239,040 B
down  = 2816 rows × 352 B + scales 2816×22×2B + biases 同  = 1,239,040 B
expertStride = 3,717,120 → 16KiB 切上げ 3,719,168 B
               (= 227 ページちょうど。現行 3,358,720 B = 205 ページ 比 +10.7%)
layer file = 128 × 3,719,168 = 476,053,504 B (454 MiB) × 30 層 = 14.28 GB
KV @4K     = 0.28 GB (アーキ変更なし → PLAN 1-7 と同じ)
```

**resident は当初 1.2 GB と見積もったが、実測は 1.51 GB だった** (**実測**、
スナップショットの非 routed テンソルのバイト総和 = 1,512,804,412 B)。
shared 8bit→4bit の減りより group 32 のスケール倍増のほうが大きく、
「ほぼ相殺」という当初の読みは外れている。現行ピンの 1.354 GB に対し **+0.16 GB**。

`ExpertCacheBudget` の式 (resident + 30×slots×stride + KV vs 12.88 GB):

| slots | 計算 | 合計 | 判定 |
| ---: | --- | ---: | --- |
| 48 | 1.51 + 5.36 + 0.28 | **7.15 GB** | 余裕 |
| **64** | 1.51 + 7.14 + 0.28 | **8.93 GB** | **OK** (推奨 12.88 GB 内) |
| 80 | 1.51 + 8.93 + 0.28 | 10.72 GB | OK だが余白小 |

→ **64 スロットは載る。**resident が 0.3 GB 増えても結論は変わらない。
ガードは実装済みなので、実測がさらに大幅に上振れた場合だけ 48 へ落とす
(ユーザー許可済み)。

ただし**予算の余白は確実に減る**: peak は現行 7.11 GB → **約 9.3 GB** の見込み
(**導出**)。12 GB 予算に対する余白が 4.9 GB → 約 2.7 GB になる。
Phase 5 (文脈長を伸ばす) と併用する場合は KV を足した上で再判定すること。

ディスク: ダウンロードは完了済み (§1-2)。repack 出力約 15.5 GB + 予備 1 GB。
空き 549 GB (**実測**) で問題なし。

性能の予想 (**導出**): 64 スロットで decode は GPU 律速 (RESULTS §4)。expert
読み出しバイトが +10.7% なので、MoE カーネルと io が少し重くなり
**haiku 30.84 tok/s → 28-31 tok/s 台**と予想。I/O は 99.2% ヒットでほぼ隠れている
ため、悪くても −10% 程度に収まるはず (未確認)。

TTFT: 初回充填 6.4 GB → 7.1 GB なので +0.7 s 程 (**導出**)。

---

## 3. 設計

### 3-1. グループサイズ: 「プロセス単一のビルド時定数」方式

group サイズはモデルごとに一様 (config の base のみ。オーバーライドなし) であり、
1 プロセス = 1 モデル (CLI / サーバ / アプリ / decode service すべて)。
よって **function constant やカーネル複製ではなく、MetalContext がソース結合時に
`#define MOEPACK_AFFINE_GROUP_SIZE <n>` を注入する** 方式を取る。

- `MetalContext` に `affineGroupSize: Int` (既定 64) を持たせ、シェーダライブラリは
  **最初の `pipeline()` 呼び出しまで遅延コンパイル** に変える。runner は
  `Model.load` 後、カーネル生成前に `context.affineGroupSize = manifest 値` を設定する。
- 各 `.metal` の `kGroupSize`/`kMoEGroupSize`/`kPrefillGroupSize`/`kFusedGroupSize`/
  `kInt8GroupSize`/`kLMHeadGroupSize` を `MOEPACK_AFFINE_GROUP_SIZE` 参照に置換
  (既定 64 なので**既存モデルの挙動は不変**)。
  D=2816, F=704/2112, 4096, 8192 はいずれも 32 の倍数なので N % GS == 0 は成立
  (**実測**)。
- `Quantization.groupSize` (Swift 側の静的定数 64) は「既定値/レガシー値」として残し、
  **manifest から伝わるモデルごとの値** (`Manifest.quant.!.embedding.groupSize`) を
  真の値として使う。precondition は各ラッパーがモデル値で検証する。
- tensorops.metal (MPP) は tileK=64 固定 (`MPPPrefillInt4QMM.swift:12` =
  `Quantization.groupSize`) のため **group 32 モデルでは MPP 経路を無効化**。
  この機体は macOS 15 で MSL 4.0 が無く元から未使用 (**実測**) だが、
  macOS 26 で黙って誤動作するのを防ぐためにゲートは必ず入れる。

#### 3-1-a. 訂正: 「定数置換のみ」では通らない (**実測**、最重要)

> **当初この節には「カーネル本体のロジックは 1 行も変わらない / 定数置換のみで
> 追従する」と書いていた。ソースを読んで確認した結果、これは誤り。**

decode の INT4 GEMV は**ベクタ化ブロックの分割を数値リテラルで直書き**している。
`kGroupSize` では書かれていない:

```
Sources/Tsugumi/Metal/Quant/dequant_int4.metal:126,135
Sources/Tsugumi/Metal/Sampling/logit.metal:648,651
Sources/Tsugumi/Metal/MoE/moe.metal:207,209      (routed down)
Sources/Tsugumi/Metal/MoE/moe.metal:273,278      (routed gate/up, u16load 版)
```

```metal
const uint full_blocks = n_groups / 4;           // 「4 グループ = 128B」= GS 64 前提
const uint byte_base = blk * 128u + lane * 4u;   // ← 128 が直書き
const uint g = blk * 4u + (lane >> 3);           // ← 「8 lane / group」が直書き
```

`dequant_int4.metal:115-123` のコメント自身が
"32 lanes split 8-per-group, each handling 8 contiguous elements of one
**64-element** group" と前提を明記している。

define だけ 32 に差し替えた場合に起きること:

1. **行の 2 倍を読む OOB。** N=2816 → `n_groups`=88 → `full_blocks`=22 →
   22×128 = 2816 B 読むが `row_bytes` は 1408 B。
2. **スケールの対応が全部ずれる。** 正しくは `g = blk*8 + (lane>>2)`。
3. **末尾ループも壊れる。** `W_row[g*(GS/2) + lane]` は GS/2 = 16 に対し
   lane が 31 まで走る。routed down は N=704 → 22 groups で 2 ブロック +
   **末尾 6 グループ**が必ずこの経路に入るので、確実に踏む。

いずれも**コンパイルエラーにならず、数値だけが狂う**。これが本 PLAN 最大のリスク
(§6 参照)。

**修正方針** (4 箇所、各 3-4 行):

```
groupsPerBlock = 256 / GS      // 64→4, 32→8
lanesPerGroup  = GS / 8        // 64→8, 32→4
full_blocks    = n_groups / groupsPerBlock
byte_base      = blk * 128u + lane * 4u            // 変更なし (常に 32 lane × 4B)
g              = blk * groupsPerBlock + lane / lanesPerGroup
末尾ループ     = lane >= GS/2 のレーンをマスク (simd_sum に 0 を寄与させる)
```

GS=64 を代入すると現行コードと完全に同一に戻るので、**既存モデルの挙動不変は
この式のまま保証される**。

一方 **prefill 側は本当に定数置換だけで通る**: `prefill.metal:414-476` /
`503-525` は純粋なスカラーループで `kPrefillGroupSize` のみに依存している
(**実測**)。「定数置換のみ」は prefill には当てはまり、decode には当てはまらない。

#### 3-1-a-2. 5 箇所目が Phase B で見つかった (**実測**、2026-08-16)

上の 4 箇所を数えたとき、**decode の INT8 ルーター
(`moe.metal` `router_gemv_gemma4_body`) を見落としていた**。同じ病気で、

```metal
const uint idx = g * kMoEGroupSize + lane * 2u;   // 32 lane × 2 = 「1 群 = 64 要素」前提
```

group 32 では 1 反復で 64 要素読むのに群は 32 要素しかなく、隣の群を
**前の群のスケールで**読んだうえ最終群で行末を 1 行ぶん飛び越す。

見落とした理由も、見つかった理由も同じ: **QAT モデルはルーターが bf16 なので
このカーネルを通らない**。§5-A の 6 ケースは int4 経路しか見ておらず、
Phase B でルーターのケースを足して初めて露出した。修正は §3-1-a と同じ形:

```
groups_per_step = 64 / GS     // 64→1, 32→2
lanes_per_group = 32 / groups_per_step
g = st * groups_per_step + lane / lanes_per_group
```

GS=64 を入れると `g == st` で元のコードに戻る。実測でも group 64 の相対誤差は
修正前後で `3.320e-04` の同値、group 32 は修正前 FAIL / 修正後 PASS。

**教訓:** 「新モデルが通らない経路」は棚卸しから落ちる。group パラメータ化を
入れた以上、**group 32 で動きうるカーネルは全部**ハーネスに載せる。
なお prefill 側の INT8 ルーター (`prefill_router_gemma4_block`) はスカラー
ループなので健全 (**実測**)。

#### 3-1-a-3. 棚卸しの結果: さらに 2 箇所 (**実測**、2026-08-16)

上の教訓を受けて `MOEPACK_AFFINE_GROUP_SIZE` を参照する 6 ファイルを全走査した。
**同じ「1 群 = 64 要素」を仮定していたのはあと 2 つ**、どちらも
`dequant_int8.metal`:

| カーネル | 用途 | 判定 |
| --- | --- | --- |
| `dequant_int8_gemv_simd` | shared expert down、旧 lm_head | **同じ病気** → 修正 |
| `shared_int8_gate_up_act_simd` | shared expert gate/up 融合 | **同じ病気** → 修正 |

**この 2 つは現行ピンで毎トークン通る**(現行の shared expert は 8bit)。
ルーターと違って「使われていない経路」ではないので、group 64 での恒等性が
そのまま既存モデルの正しさになる。§5-A の group 64 側は修正前後で
`3.454e-04` / `3.761e-04` の同値、実機の greedy 出力も文字単位で一致、
ヒット数も `3341/7854` `7238/7440` で不変 (**実測**)。

安全な残り (**実測**、いずれも group 依存の幾何を持たない):

- `dequant_int4.metal` / `moe.metal` / `logit.metal` のベクタ化ブロック
  → §3-1-a で修正済み、§5-A のケース 1-5 が担保
- `embed_lookup_int4` / `prefill_embed_lookup_int4_block` — 要素ごとに
  `gid / GS` でスケールを引くだけ
- `prefill.metal` の GEMV / QMM / ルーター — 純粋なスカラーループ
- `tensorops.metal` (MPP) — group != 64 で経路ごと無効化済み (§3-1)

これで **group 32 で走りうる経路はすべてハーネス上にある**。

#### 3-1-b. 初期化順序 (**実測**)

`MetalContext()` は 3 つの入口すべてで `Model.load` より**前**に構築される:

```
Sources/TsugumiCLI/Run.swift:63          → :64 で Model.load
Sources/TsugumiServer/Core/ServerInference.swift:403
Sources/TsugumiApp/Core/Inference/RealInferenceClient.swift:194
```

`MetalContext.init` は `compileShaderLibrary` を即時に呼び `library` を
`let` で持つ (`MetalContext.swift:65-71, 121-138`)。よって define 注入方式では
**ライブラリの遅延コンパイル化が必須**。`moduleLibrary` (`:141`) も同じ扱いにする。

#### 3-1-c. 却下理由の訂正: function constant

> 当初「複数モデル混在や PSO 二重管理が必要になるため却下」と書いていたが、
> これも現状と合わない。`MetalFunctionConstant` と**定数をキーにした PSO
> キャッシュは既に実装済み** (`MetalContext.swift:56-62, 164-218`) で、
> 二重管理は発生しない。

それでも **define 注入を採る**。理由は却下ではなく積極的な選択で、
`for k < GS/2` のようなループが**コンパイル時定数として畳み込まれる**ため。
function constant でも Metal は特殊化するが、遅延コンパイル化 1 箇所で済む
define のほうが変更範囲が小さい。**どちらでも成立する**ことは記録しておく。

### 3-2. bf16 ルーター経路

- **decode**: `moe.metal` に新カーネル `router_gemv_gemma4_bf16_r4` を追加。
  既存 int8 版 (`router_gemv_gemma4_r4`) と同じ SIMD 構成 (4 専門/SIMD、32 lane) で、
  積は `w(bf16) × (x(half) × es(bf16))` のみ。scales/biases バッファは不要。
  `MoE.swift` は `encodeRouterGemma4BF16(...)` を追加し、PSO はルータービット数で選択。
- **prefill**: `prefill.metal` に `prefill_router_gemma4_bf16_block` を追加
  (top-k 選択ロジックは既存 block と共通の inline 関数に切り出して再利用)。
  `PrefillRouter.encodeGemma4Block` に bf16 変種、または `encodeGemma4BF16Block`。
- **RealForwardRunner**: `model.routerWeightBits` (新規 computed) で分岐。
  prefill/decode 両経路。
- **manifest**: router スロットを `weightBits: 16, scheme: "bf16", scaleType: "none",
  biasType: "none"` で書く。`ManifestReader.validateQuant`
  (`ManifestReader.swift:141-158`) は 3 点を直す:
  1. router の許容ビットを `[8] → [8, 16]`
  2. **`scheme.lowercased() == "affine"` の一律要求を router だけ緩和**
     (bits==16 なら "bf16" を許す)。当初これを書き落としていた
  3. 16 のとき `scaleType`/`biasType` が "none" であることを要求
  同関数の `slot.groupSize == Quantization.groupSize` (`:154`) は
  §3-1 のとおり manifest 駆動へ。
  `Model.validateRuntimeSchema` の `requireAffine(router.proj.weight)` は
  bits==16 のとき `requireBF16` (companion なし) に切り替える
  (`Model.swift:585` の `weightBits == 4 || == 8` ガードも同時に拡張)。

### 3-3. repacker のローカルスナップショット対応

スナップショットは配布のまま手元にある (§1-2) ので、遠隔レンジ取得の
dual-source 化ではなく**ローカル入力モード**を追加する:

- `TsugumiRepack` CLI に `--source-snapshot <dir>` を追加
  (既定の従来挙動 — pinned リポジトリのストリーミング取得 — は不変)。
- ローカルモードは `IndexLoader.load(snapshotDir:)` + shard ヘッダ読み出しから
  `RepackPlanner.plan` に入り、既存の `RangeCopyPlanner` / 書き出し / ハッシュ /
  manifest / 受領証の経路をそのまま流用する (byte provider をローカル pread に差し替え)。
- 指紋検証: ローカルモードでも index SHA-256 が `qat-aligned` のピン
  (`7dbbeef0…`) と一致することを要求する (`SourceFingerprint.knownFingerprints` に
  エントリ追加)。リビジョン表記はスナップショット由来の pinned revision `745a97a7…`。
- `writeManifest` の `QuantBitWidths.router`: 既定 8 のまま、router エントリの
  dtype が BF16 (quantSpec nil) のとき 16 を書く。groupSize は
  `plan.baseGroupSize` を既存どおり流用 (32 が入る)。
- `AppModelInstallDescriptor.default` の複製定数は既定ソース側のみ参照 (既定は不変)。

### 3-4. スロット数・その他

- 既定 64 は維持 (`RuntimeConfiguration` 変更なし)。`ExpertCacheBudget` が
  resident 実測込みで自動判定する。
- expert I/O / RDADVISE / telemetry / prefill チャンク: ロジック変更なし
  (stride が manifest 駆動で大きくなるだけ)。
- KV・attention・norm: アーキ変更なしなので触らない。

---

## 4. 変更ファイル一覧

### Phase A0 — group 32 の数値検証ハーネス (Phase A より先に作る) — **完了**

> `TsugumiKernelCheck` として実装。`swift build -c release` に含まれる。
>
> ```
> ./.build/release/TsugumiKernelCheck                 # group 64 と 32 の両方
> ./.build/release/TsugumiKernelCheck --group-size 32
> ```
>
> **ハーネス自身の穴を 1 つ潰した (**実測**)。**当初 `RelError.compute` を
> 使っていたが、これは要素ごとの差を `max(_:_:)` で畳む。Swift の
> `max(0, .nan)` は **0 を返す**ので、**出力が NaN だらけのカーネルが
> 相対誤差 0 = PASS になる**。実際に routed-moe の group 32 が壊れたまま
> PASS していた。NaN-safe な `relativeError` に差し替え、併せて
> (a) コマンドバッファのエラー検査、(b) 参照側に信号があることの検査
> を足した。**このハーネスが存在する理由そのものを無効化する穴だった。**

### Phase A0 — 変更ファイル

§5-A の出口条件を機械判定にするための最小の実行可能ファイル。**Phase A の
コードを書く前にこれを用意する** (書いた後だと合わせ込みの誘惑が働く)。

| ファイル | 変更 |
| --- | --- |
| `Package.swift` | `.executableTarget(name: "TsugumiKernelCheck")` を追加。依存は `Tsugumi` + `TsugumiValidationSupport` |
| `Sources/TsugumiKernelCheck/main.swift` | 合成入力を group 32 / 64 の両方で量子化し、GPU カーネルと CPU 参照を突き合わせて PASS/FAIL を出す |

CPU 参照は既にある: `TsugumiValidation/Support/Reference/Quant/`
(`DequantInt4Gemv.swift`, `EmbedLookup.swift`, `DequantInt8Gemv.swift`)。
これらは `Quantization.groupSize` を直接参照しているので、**参照側も
group を引数で受け取れるようにする**(参照実装は素直なループなので容易)。

対象カーネルと N は §5-A の表を参照。

### Phase A — カーネルの group パラメータ化 (既存モデルの挙動不変を維持) — **完了**

> **§3-1-a の診断は実機で裏が取れた。**修正前に §5-A を回した結果:
>
> ```
> group 64: 6/6 PASS   (int4-gemv 4 形状 / routed-moe / lm-head)
> group 32: 6/6 FAIL   rel = 5.4e+01, 1.0e+01, 4.8e+01, 7.9e-01,
>                            inf (routed-moe が NaN), lm-head は argmax 総崩れ
> ```
>
> 修正後:
>
> ```
> group 64: 6/6 PASS   rel は修正前と 1 桁も変わらず同値
>                      (3.683e-04 / 3.224e-04 / 3.729e-04 / 2.665e-04 / 4.330e-04)
> group 32: 6/6 PASS   rel = 3.043e-04 / 1.963e-04 / 2.227e-04 / 2.451e-04 / 3.752e-04
> ```
>
> **group 64 の相対誤差が修正前後で完全に一致した**ことが、既存モデルの
> 挙動不変の直接の証拠になっている (式に GS=64 を代入すると元のコードに戻る)。
>
> 実機スモーク (`gemma4.moepack` / haiku / 32 tok) も、ヒット率のカウントが
> 修正前と 1 件も違わない (`prefill 3341/7854`, `decode 7238/7440`)。
>
> **Phase B 以降が使う API (実装済み):**
>
> | シンボル | 場所 | 用途 |
> | --- | --- | --- |
> | `MetalContext.setAffineGroupSize(_:)` | `MetalContext.swift` | 最初の `pipeline()` / `library` 前に呼ぶ。以後はエラー |
> | `MetalContext.affineGroupSize` | 同上 | 焼かれた (or 焼かれる) 値 |
> | `MetalContext.canUseAffineGroupSize(_:)` | 同上 | 使い回しコンテキストが使えるかの非 throw 判定 |
> | `MetalContext.library` | 同上 | **`get throws` になった。**呼び出し側は `try context.library` |
> | `Model.affineGroupSize` | `Model.swift` | manifest 由来。`quant.embedding.groupSize ?? 64` |
> | `Quantization.supportedGroupSizes` | `Quantization.swift` | `[32, 64]` |
> | `Quantization.quantizeInt4Affine(_:groupSize:)` 他 | 同上 | 4 関数に `groupSize:` を追加 (既定 64) |
> | `MOEPACK_AFFINE_GROUP_SIZE` | 各 `.metal` | `preprocessorMacros` で注入。各ファイルに `#ifndef` の 64 フォールバックあり |
>
> `RealForwardRunner.init` が `try context.setAffineGroupSize(model.affineGroupSize)`
> を**カーネル生成より前に**呼ぶ。3 つの入口 (CLI / Server / App) はいずれも
> `MetalContext()` → `Model.load` → `RealForwardRunner` の順なので安全
> (**実測**、確認済み)。Mac アプリだけはコンテキストを使い回すので、
> `canUseAffineGroupSize` が false のとき張り替える。

### Phase A — 変更ファイル

| ファイル | 変更 |
| --- | --- |
| `Metal/MoE/moe.metal` | `kMoEGroupSize` を define 参照に + bf16 ルーターカーネル追加 (Phase B で一緒に) |
| `Metal/Quant/dequant_int4.metal` | `kGroupSize` を define 参照に |
| `Metal/Quant/dequant_int8.metal` | `kInt8GroupSize` を define 参照に (旧モデル用に残す) |
| `Metal/Sampling/logit.metal` | `kLMHeadGroupSize` を define 参照に |
| `Metal/Prefill/prefill.metal` | `kPrefillGroupSize` を define 参照に + bf16 ルーター block (Phase B) |
| `Metal/Fusions/fused.metal` | `kFusedGroupSize` は宣言のみで**未使用** (`fused.metal:13` の 1 箇所だけ — **実測**)。除去する |
| `Metal/TensorCore/tensorops.metal` | `kW4A8GroupSize` (`:8`, `:54`)。MPP 経路ごと gate するので define 化のみ |
| `Infrastructure/Metal/MetalContext.swift` | `affineGroupSize` + ライブラリ遅延コンパイル (`moduleLibrary` も) + define 注入 |
| `Kernels/TensorCore/MPPPrefillInt4QMM.swift` | `tileK`(`:12`) が group 64 固定。group != 64 なら経路を無効化するゲート |
| `Kernels/**` (ラッパー群) | precondition を `Quantization.groupSize` → モデル値へ。対象 10 箇所 (**実測**): `DequantInt4GEMV:57` `DequantInt8GEMV:60` `EmbedLookupInt4:30` `LMHeadChainInt4:71` `FusedQKVGEMV:55` `PrefillFinalRowHead:42` `PrefillPrimitives:20,89` `PrefillRouter:62` `MoE:121` |

**Phase A の本体は上表ではなく §3-1-a の 4 箇所 (計 約 16 行)。**
残りは機械的な置換で、危険度が桁違いに違う。

### Phase B — bf16 ルーター — **完了**

| ファイル | 変更 |
| --- | --- |
| `Metal/MoE/moe.metal` | `router_gemv_gemma4_bf16_r4` + **INT8 ルーターの group ジオメトリ修正 (§3-1-a-2)** |
| `Metal/Prefill/prefill.metal` | `prefill_router_gemma4_bf16_block` (top-k は `prefill_router_emit_topk` に共通化) |
| `Kernels/MoE/MoE.swift` | `encodeRouterGemma4BF16` + `RouterError` + select 部の共通化 |
| `Kernels/Prefill/MoE/PrefillRouter.swift` | `encodeGemma4BF16Block` |
| `Runtime/Inference/RealForwardRunner.swift` | ルーター分岐 (decode + prefill)、MoE/PrefillRouter に bits を渡す |
| `Runtime/Inference/Model.swift` | `routerWeightBits` 公開 + `validateRuntimeSchema` に `requireBF16Matrix` |
| `Infrastructure/ModelIO/ManifestReader.swift` | router bits [8,16]、bits 16 は scheme/scale/bias を `bf16`/`none`/`none` 要求、**groupSize を manifest 駆動に** |
| `Metal/Quant/dequant_int8.metal` | §3-1-a-3 の 2 箇所を同じ変換で修正 |
| `Kernels/Quant/DequantInt8GEMV.swift` / `Kernels/MoE/SharedExpertInt8.swift` | N % 64 のガード。前者は harness 用に `package` 可視化 |
| `TsugumiValidation/.../DequantInt8Gemv.swift` / `MoE/Moe.swift` | 参照に `groupSize:` を追加、`MoeRef.runFFNInt8` を新設 |
| `TsugumiKernelCheck/main.swift` | §5-A ケース 6-9 |

**設計上の決定 (実装で確定した点):**

- **ルーターは片方しかコンパイルしない。** `MoE` / `PrefillRouter` は
  `routerWeightBits` を受け取り、対応する 1 本だけ PSO を作る
  (`SharedExpertRuntime(weightBits:)` と同じ形)。誤った側を encode したら
  precondition で落ちる。
- **bf16 カーネルは group サイズに一切依存しない** (群構造がないため)。
  `MOEPACK_AFFINE_GROUP_SIZE` を参照しないので、group 32/64 のどちらでも同一。
- `ManifestReader.validateQuant` の groupSize 検証は
  「`Quantization.supportedGroupSizes` に入っていて、かつ全スロットで一致」に
  変えた。`Model.affineGroupSize` が embedding スロット 1 つを全体の代表として
  読む前提 (Phase A のコメント) を、これで初めて実際に保証している。

### Phase C — repacker のローカルスナップショット対応 (簡略化済み)

| ファイル | 変更 |
| --- | --- |
| `Repack/Command/main.swift` | `--source-snapshot <dir>` フラグ |
| `Repack/Core/Remote/SourceByteProvider.swift` | `LocalSnapshotByteProvider` を追加。**`SourceByteProvider` は既にプロトコル** (`:4` — **実測**) なので `HTTPRangeSourceByteProvider` の隣に 1 実装足すだけ |
| `Repack/Core/Verification/SourceFingerprint.swift` | `7dbbeef0…` を `knownFingerprints` に追加 (`:6` は現在 1 エントリのみ) |
| `Repack/Core/Remote/RemoteStreamingRepacker.swift` | writeManifest の router ビット判定 |
| `App/…/AppModelInstallDescriptor` | 既定参照の整合確認 (既定は不変) |
| `README.md` / `docs/RUNTIME_CONTROLS.md` | ローカルモードの記述 |

### Phase D — インストールと検証 (§5)

---

## 5. 検証プロトコル

各 Phase のビルド確認: `swift build -c release`。テストランナーは本環境で使用不可
(RESULTS §9: swift-testing モジュール不在 — 私の変更以前からの状態)。
**このため §5-A は `swift test` ではなく実行可能ターゲット (Phase A0) で行う。**
**既存モデルでデグレなし確認** を Phase A/B の各段階で実施 (§5-0)。

**開発環境 (**実測**、2026-08-16)。追加セットアップは不要:**

```
Apple Swift 6.2.4 / target arm64-apple-macosx15.0
xcode-select -p → /Library/Developer/CommandLineTools   (Xcode は未インストール)
swift build -c release (no-op) → 5.6 s
ディスク空き 535 GB
```

Phase 0-2 はすべてこの構成で実施済み。**Xcode は不要**で、入れても得るものは
GPU フレームキャプチャだけ。今回の主リスク (§3-1-a) には §5-A の数値検証の
ほうが速く確実なので、**入れない**。

シェーダは**ランタイムコンパイル** (`MetalContext.swift:121-138` が `.metal`
ソースをバンドルから読んで `makeLibrary(source:)`)。したがって
**`.metal` の編集に Metal のビルドステップは存在せず**、`swift build -c release`
はリソースをコピーするだけ。Phase A の反復は数秒で回る。

### 5-0. 既存モデルの回帰確認 (Phase A+B 完了時、新モデル DL 前)

```bash
.build/release/TsugumiCLI --model scratch/gemma4.moepack \
  --messages-file bench/haiku.json --max-new 64 --seed 1   # 動作 + footer
TEMP=0 MAXNEW=384 ./bench.sh ja   # tok/s が RESULTS §3-4 から ±4% 以内
```

> **§5-0 だけでは不十分。** 旧モデルは group 64 なので、この回帰確認は
> **§3-1-a の group 32 バグを構造的に検出できない。**必ず §5-A と併用する。

#### 結果 (**実測**、2026-08-16、Phase A+B 適用後)

`TEMP=0 MAXNEW=384 ./bench.sh ja`、3 回インターリーブ、64 スロット、
`trusted-install`。ベースラインは RESULTS §3-4 (同条件・greedy)。

| prompt | 3 回の tok/s | 中央値 | ベースライン | 差 | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| haiku | 31.132 / 30.838 / 30.852 | 30.852 | 30.84 | **+0.04%** | 合格 |
| math | 28.442 / 28.306 / 28.480 | 28.442 | 28.42 | **+0.08%** | 合格 |
| story | 31.001 / 31.005 / 30.645 | 31.001 | 31.31 | **−0.99%** | 合格 |

**3 本とも ±1% 以内**で、±4% のゲートに対して大きな余裕がある。
副次的な一致も確認した (性能より強い同一性の証拠):

- decode ヒット率 99.2 / 97.4 / 98.5% は §3-4 と**小数第 1 位まで同値**
- peak 7.02-7.10 GB は §3-4 の 7.05-7.13 GB と同水準
- 3 本とも `stop=maxTokens` で 384 tok 生成 (§3-4 と同じ打ち切り方)

Phase B 適用後の実機スモーク (`--max-new 32`、haiku、greedy) も
**Phase A 記録のヒット数と 1 件も違わない**: `prefill 3341/7854`、
`decode 7238/7440` (**実測**)。

### 5-A. group 32 カーネルの数値検証 (**Phase A の出口条件**)

本 PLAN の目的は品質向上そのものなので、「カーネルが微妙に間違っている」と
「チェックポイントの品質がその程度」を**区別できる手段が要る**。
§5-2 の目視ゲートにはこの分離能力がない (どちらも「日本語がやや変」に見える)。
§3-1-a の壊れ方は NaN ではなく「もっともらしいが劣化した出力」になりうる。

Phase A0 のハーネスで、**合成入力に対し GPU カーネル vs CPU 参照**を突き合わせる。

| # | カーネル | N | 狙い |
| --- | --- | ---: | --- |
| 1 | `dequant_int4_gemv_simd` | 2816 / 4096 / 8192 | ベクタ化ブロックのみ (末尾なし) |
| 2 | `dequant_int4_gemv_simd` | **2112** | **末尾ループを踏む** (group32 で 8 blk + 2 群) |
| 3 | `lm_head_greedy_int4_rows_chunk_raw` | 2816 | LM head の同型ループ |
| 4 | MoE gate/up (`…u16load`) | 2816 | routed、2 行同時読み |
| 5 | MoE down | **704** | **末尾ループを踏む** (2 blk + 6 群) |
| 6 | bf16 ルーター (Phase B) | 2816 | 新規カーネル |
| 7 | int8 ルーター (Phase B で追加) | 2816 | §3-1-a-2 の 5 箇所目 |
| 8 | `dequant_int8_gemv_simd` | 2112 | §3-1-a-3 |
| 9 | int8 shared expert (gate/up 融合 + down) | 2816 / 2112 | §3-1-a-3。現行ピンの実経路 |

- 各ケースを **group 64 と group 32 の両方**で回す。64 側は「既存の挙動が
  1 bit も変わっていない」ことの確認、32 側が本番。
- 判定は相対誤差。BF16 スケールと FP32 累算の順序差ぶんだけ緩め、
  **符号・桁が合っていれば通る程度の緩さにはしない**
  (§3-1-a のバグはスケール対応がずれるので相対誤差で明確に落ちる)。
- 出力は `PASS` / `FAIL <case> rel=…` の 1 行ずつ。Swift を読まずに判定できること。

**このハーネスが緑にならないうちは Phase C 以降に進まない。**

> **実施済み (**実測**)。**実装した case は 6 本 (上表の 1/2/3 を int4-gemv の
> 4 形状に統合し、MoE と LM head を各 1 本)。
> `n=2112` と `n=704` が末尾ループを踏む形状で、実際に group 32 で
> `blocks=8 tail=2` / `blocks=2 tail=6` と報告される。
> 結果は §4 Phase A の注記を参照。
>
> LM head は argmax しか観測できず 1 本あたりの検出力が低いので、
> **独立な 4 draw の全一致**を要求している (壊れたカーネルが 4 回とも
> 参照と同じ argmax を出す確率は無視できる)。実際、修正前は 4 draw 中
> 4 draw とも不一致だった。
>
> **Phase B で 4 ケース追加し、group あたり 10 ケース = 計 20 になった**
> (**実測**、2026-08-16)。ルーターは top-8 のインデックス完全一致 + 重みの
> 相対誤差で判定し、これも 4 draw 回す。CPU 参照はハーネス内の素の二重ループで、
> カーネルと構造を共有しない。
>
> ```
> group 64: 10/10 PASS  router-int8 3.320e-04 / router-bf16 2.308e-04
>                       int8-gemv 3.454e-04 / shared-expert-int8 3.761e-04
> group 32: 10/10 PASS  router-int8 4.261e-04 / router-bf16 3.756e-04
>                       int8-gemv 2.172e-04 / shared-expert-int8 4.372e-04
> ```
>
> **検出力も 4 本すべてで確認した** (旧ジオメトリに戻して再実行):
>
> | ケース | group 64 | group 32 |
> | --- | --- | --- |
> | router-int8 | PASS `3.320e-04` (同値) | FAIL 4 draw ともエキスパート選択が不一致 |
> | int8-gemv | PASS `3.454e-04` (同値) | FAIL `rel=1.118e+00` |
> | shared-expert-int8 | PASS `3.761e-04` (同値) | FAIL `rel=6.834e+00` |
>
> group 64 の相対誤差が**修正前後で 1 桁も変わらない**ことが、
> 既存モデルの挙動不変の直接証拠 (式に GS=64 を入れると元のコードに戻る)。

### 5-1. インストール (ローカルスナップショットから)

```bash
swift run -c release TsugumiRepack --output scratch/gemma4-qat.moepack \
  --source-snapshot scratch/qat-aligned-snapshot
```

(遠隔取得ではなくローカル pread のため `--resume` は本来不要だが、中断時に
出力ディレクトリの破棄手順は従来どおり `--discard-partial`。)

### 5-2. 受入チェック

1. 生成の目視確認 (temp 1.0 / top-k 64 / top-p 0.95、`--seed 1`):
   日本語が破綻なく、ループしないこと。
2. **品質ゲート** (PLAN 付録プロンプト、`--max-new 1024`):
   - 寿司俳句: temp 1.0 で完答すること (ベースラインも 828 tok で完答 — RESULTS §3-5)
   - 三次方程式: 手法が妥当であること (ベースラインで見られたアラビア語トークン混入が消えるかは観察項目)
3. **性能** (PLAN §6 プロトコル、インターリーブ 3 回中央値):
   `./bench.sh ja` 相当を `--model scratch/gemma4-qat.moepack` で。
   合格: haiku tg **≥ 27.8 tok/s** (30.84 の −10%)。TTFT は +1 s 以内。

   > **ゲート値に余白がほとんどないことを先に認めておく** (**導出**)。
   > decode は 64 スロットで GPU 律速 (RESULTS §4)、未計装 26 ms/token の
   > 主成分は routed expert GEMV のメモリ帯域。expert バイトが +10.7%、
   > スケール帯域は行あたり 176 B → 352 B と倍増する。MoE が decode の
   > 6 割なら素直に **−6〜−7%**、悪くて −10% 近辺に着地する計算で、
   > **予想レンジがゲート値をまたいでいる。**
   >
   > **事前の取り決め: −10% ちょうど〜わずかに超過 (27.0〜27.8 tok/s) の場合、
   > 品質ゲート (2) がベースラインより明確に良ければ採用可とする。**
   > 品質が同等以下なら §7 の中止条件どおり見送る。
   > 「速度が落ちたので品質ゲートを甘く見る」という順序は取らない。
4. **メモリ**: footer の peak < 12 GB、かつ `ExpertCacheBudget` が 64 スロットを
   通すこと。弾かれたら 48 で再測 (ユーザー指示)。
   予想 peak は約 9.3 GB (§2)。**10 GB を超えたら resident の実測値を
   §2 に戻して再評価する** (試算が 0.3 GB 甘かった前例があるため)。
5. `--verification trusted-install` / `full-sha256` 両方で起動すること。

### 5-3. 記録

`RESULTS_QAT.md` に、PLAN §6 準拠 (commit / ハード / コマンド / exit code /
footer 全文 / プロトコルからの逸脱すべて) を残す。スイープ内インターリーブ比較のみ
(ページキャッシュ依存の 注意 も PLAN どおり)。

---

## 6. リスク

| リスク | 内容 | 対策 |
| --- | --- | --- |
| ~~**ベクタ化ブロックの黙った破壊**~~ | ~~§3-1-a~~ | **解消 (2026-08-16)。**Phase A で 4 箇所、Phase B でさらに 3 箇所 (§3-1-a-2 / §3-1-a-3) を書き換え、§5-A は group 64/32 とも **10/10 PASS**。7 箇所すべて、旧コードに戻すと group 32 が FAIL / group 64 は同値のまま PASS することを確認済み (検出力の裏取り) |
| ハーネス自身が壊れを見逃す | 比較関数が NaN を「一致」に潰す、コマンドバッファのエラーを見ない、参照側が無信号 | **1 件顕在化して解消 (2026-08-16)。**`RelError.compute` は `max(0, .nan) == 0` のため NaN 出力が rel=0 = PASS になり、実際に routed-moe の group 32 が壊れたまま PASS していた。NaN-safe な比較 + コマンドバッファ検査 + 参照信号検査に差し替え済み |
| group 32 化による decode 性能低下 | expert 読み出し +10.7%、スケール帯域は倍 | GPU 律速なので影響は小さい見込み (§2)。**ゲート値に余白がないことは §5-2-3 で明示し、判定基準を事前に決めた** |
| define 注入の遅延コンパイル | 初回 `pipeline()` 前に group size を設定し忘れると誤ったライブラリが焼かれる。`MetalContext()` は 3 入口とも `Model.load` より前に構築される (§3-1-b、**実測**) | runner init 順序を 1 箇所に集約、設定前に pipeline を引かない前提を assert |
| 上流 README の品質主張が再現しない | KL 0.090 等は自己申告 (**未確認**) | 受入は §5-2 の実機ゲートで判定し、数値を信用しない |
| ~~インストール中断~~ | ~~15.8 GB の DL~~ | **解消。スナップショット取得済み (§1-2)。repack はローカル** |
| shared expert 4bit の品質 | 現行 8bit から低下する可能性 | QAT チェックポイント由来のため許容。俳句/数式ゲートで観察 |
| ~~既存モデルのデグレ~~ | ~~Phase A でカーネル定数を触る~~ | **Phase A+B について解消 (2026-08-16)。**§5-0 は 3 本とも ±1% 以内でヒット率まで同値、§5-A の group 64 は 8/8 PASS で相対誤差も修正前と同値 |
| ~~**棚卸しの漏れ**~~ | ~~group パラメータ化の対象一覧 (§3-1-a) が 4 箇所~~ | **解消 (2026-08-16)。**実際には 7 箇所だった (§3-1-a-2 で 5 番目、§3-1-a-3 で 6・7 番目)。全 6 ファイルを走査し、残りは group 依存の幾何を持たないことを確認。**group 32 で走りうる経路はすべて §5-A に載っている** |

## 7. 中止条件

- **§5-A が緑にならない → Phase C 以降に進まない** (実機の目視ゲートで
  代替しようとしない)
- Phase A/B の時点で既存モデルの回帰が ±4% を超える → 設計を見直すまで先へ進まない
- 受入で tg が −10% 超、かつ品質ゲートもベースライン並みに見えない → 導入見送りを報告
- インストールや検証で 12 GB 予算を守れない構成しかない → 48 スロットでも同じなら報告して停止

## 8. 明示的にやらないこと

- MTP ドラフター (Candidate 3) の対応 — 調査資料のとおり改造範囲外
- group 32 スケールの group 64 への併合 (非可逆のため不可、調査資料記載)
- 既定ソース (ピン) の切り替え。`--source` 指定時のみ新モデル
  (受入後の切り替えは §0-1 のとおりランタイムを壊さないので、独立に判断する)
- 最適化作業 (MoE カーネル改善など) — 本 PLAN は「載せる」まで。性能は測定のみ
- **group 64 対応の削除。**受入後も残す (§0-1 の判断)

## 9. 他エントリポイントの動作確認 (**実測**、2026-08-16)

§5 の受入プロトコルは CLI (`bench.sh` 経由) でしか測っていなかった。
`TsugumiServer` (OpenAI 互換) と `TsugumiApp` (Mac GUI) は
Phase A-D を通じて一度も動かしていなかったため、優先度の高い Server から
動作確認した。

```bash
.build/release/TsugumiServer --model scratch/gemma4-qat.moepack --port 8091 &
curl http://localhost:8091/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-26b-a4b-it","messages":[{"role":"user","content":"日本語で一言挨拶して"}],"max_tokens":30,"temperature":0}'
# → {"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"こんにちは！"}, ...
```

**結果: 正常応答。**§3-1-b のとおり Server も `MetalContext()` → `Model.load` →
`RealForwardRunner` の順で初期化しており (`ServerInference.swift:403-411`)、
CLI と同じ経路で group size / bf16 ルーターが解決されるため、コード変更なしで動いた。

**ただし以下は未検証のまま:**

- Server 固有の `--prompt-cache-mode single-prefix` (プロンプト KV 再利用) が
  group 32 の decode パスと組み合わさったときの数値的な正しさ・性能
  (§5-A のハーネスは単発のカーネル呼び出ししか見ていない)
- 複数リクエストのキューイング下での挙動
- `TsugumiApp` (Mac GUI) — ビルド・起動とも今回未実施。
  `RealInferenceClient.swift:194` も同じ3入口の1つ (§3-1-b) なので理屈上は
  動くはずだが、実機確認はしていない

§5-2 の品質・性能ゲート自体は CLI 実測のままで変更なし。本節は
「他エントリポイントが壊れていないか」の追加確認であり、受入判定を
やり直すものではない。
