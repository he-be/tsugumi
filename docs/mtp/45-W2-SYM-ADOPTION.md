# 45. W2 — `sym` を製品に入れる。出力は 1 バイトも変わらず +8.8% / −0.40 GB

実施: 2026-08-20、M3 Pro 18GB / macOS 15.7.5、[44](44-W1-WEIGHT-DIET.md) の続き。
一次資料は `bench/mtp45/`。**これは M9 以来はじめて推論経路を変える変更である。**

44 は「`bias` は情報を持たず、落とすと帯域床の上でバイト比どおりに速くなる」を
カーネル単体で測った。本書はそれを**形式・repacker・ランタイムに通し、受け入れる**。

## 0. 結論 — 4 つ

1. **出力は 1 バイトも変わらない** (**実測**、§3)。temp 0 / 256 トークン × 3 プロンプトで
   **バイト一致**。ルーティングも一致 (decode hit 51394/61200 が両者同一)。
   これは偶然ではなく**設計でそうした** — §2 の丸め順序。
2. **運用点 (32 スロット) で t/s +8.8%、peak −0.40 GB** (**実測**、§4)。
   23.715 → **25.796 tok/s**、4.88 → **4.48 GB**。
   **同じトークン列を出しての比較**なので、42 §4 が警告した生成長の交絡が無い。
3. **インストールは 14.71 → 13.28 GiB**、`expertStride` は 3,719,168 → **3,358,720**
   (= group-64 ベースラインと同一)。
4. **採用の条件付き**: 自動で `sym` になるのは**ローカルに置いたスナップショットから
   repack したときだけ**である (§6)。ストリーミングインストールは `affine` のまま。

## 1. 何を足したか — スキームは 2 つ目の「モデル 1 個ぶんの定数」である

group size が既に**コンパイル時のモデル全体定数**で、`MetalContext` が
モデルごとにライブラリを持っている (`init(sharingDeviceWith:)`、group 32 の target と
group 64 の drafter が同居できるのはこのため)。**スキームは同じ棚に乗る。**

- `Quantization.AffineScheme` = `.affine` / `.sym`
- `MetalContext.setAffineScheme` → `TURBO_AFFINE_SYMMETRIC` を注入
- manifest の 4bit スロットは `scheme: "sym" / biasType: "none"`。
  **8bit スロットは対象外** — INT8 群のゼロ点は本物のデータである
- **カーネルの引数は 1 つも変えていない。**`sym` では bias の**束縛が scales を指す**
  (resident は `biasSize == 0` で、エキスパート blob は `*_biases` が存在しない)。
  シェーダは `TURBO_AFFINE_SYMMETRIC` の側でそれを**読まない**

## 2. 丸め順序を合わせると、出力がビット一致する (**実測**)

最初の実装は `acc = fma(s, fma(-8, sum, dot), acc)` と畳んでいた (44 §3 で測った形)。
これは代数的には同じだが**丸めが違う**ので、256 トークン生成すると temp 0 でも
argmax が反転して分岐した (42 §3 が記録した現象と同じ)。

**畳むのをやめた。**`bias = -8 * scale` を FP32 で作り、affine と**同じ 2 つの FMA を
同じ順で**回す。`-8` は 2 の冪なので BF16 は `-8 * scale` を丸め無しに持ち、
戻した float は affine モデルがロードしていた値と**同一のビット列**である。
⇒ **パイプライン全体が affine とビット一致する。**

代償は測ってある (`bench/mtp45/bpw_probe_shipped_form.log`、44 §4 と同じ計器):

| `gate/up` r=2 | µs | GB/s | 対 affine |
| --- | ---: | ---: | ---: |
| affine | 290.7 | 136.4 | . |
| **sym (出荷形)** | **264.7** | **134.8** | **−8.9%** |
| sym (畳んだ形、44 §4) | 264.7 | 134.8 | −9.7% |

**0.8 ポイントでビット一致を買った。**`gate/up` は帯域床にいるので FMA は元々タダで、
`down` も −7.7% と畳んだ形 (−6.9%) より良い。**払っていない。**

## 3. 受け入れ 1 — 数値 (**実測**)

| 検定 | 結果 |
| --- | --- |
| カーネル数値検定 (`TurboFieldfareKernelCheck`) | **69/69 PASS**。**全 INT4 ケースを両スキームで 2 回**回す (group 64/32 × affine/sym)。フィクスチャは `bias == -8*scale` を満たす対称格子を作り、**bias 配列は書いたまま**にして参照値に使う — 導出を間違えたカーネルは「見えているのに読まない値」に対して落ちる |
| 生成のバイト一致 | temp 0 / 256 tok / 32 スロット、3 プロンプトすべて**バイト一致** |
| 同一データでの導出検定 | affine モデルを `TF_FORCE_AFFINE_SCHEME=sym` で走らせると affine ライブラリと**バイト一致**。データを固定して導出だけ入れ替える検定である |
| 既存テスト | `swift test` の赤は**変更前 34 件・変更後 28 件で同じ集合** (C2 の意図的な赤 + モック HTTP のフレーク)。追加の破れなし |

### 3a. ハーネスに穴があった (**実測**、これが本書で一番高くついた発見)

**カーネル検定 69/69 が通っている状態で、実モデルは壊れていた。**
原因は `moe.metal` の `moe_int4_gate_up_rows_simd_dev_vec_u16load` — **decode の
routed MoE gate/up 本体**が `float(gB_row[g])` という綴りで bias を読んでおり、
`float(b_row[` を探した最初の grep から漏れていた。

**そして `checkRoutedMoE` はこのカーネルを通らない。**通っていれば即座に落ちていた。
切り分けは `TF_FORCE_AFFINE_SCHEME` で付いた — **affine モデル + sym ライブラリが
正しく、sym モデル + sym ライブラリが壊れる**なら、誤りはカーネルではなくデータの
束縛側にある、と読める。実際はその逆で「1 本だけ束縛を読み続けるカーネルが居る」
だったが、二分は 1 回で当たった。

> **教訓:** 綴りで探した。型で探すべきだった。最終確認は
> 「`device const bfloat*` の全ローカルの全参照」を機械的に列挙して行った。
> **`checkRoutedMoE` が decode の gate/up を通らないことは今も直っていない** (§7)。

## 4. 受け入れ 2 — 運用点の A/B (**実測**)

32 スロット (42 §1 の運用点)、temp 0、max-new 256、同一プロンプト、
**交互に 3 往復・間に 20 秒のクールダウン**、中央値。
**両者は同じ 256 トークンを出す**ので、比べているのは純粋に速度と占有である。

| | affine | **sym** | 差 |
| --- | ---: | ---: | ---: |
| tok/s | 23.715 | **25.796** | **+8.8%** |
| peak | 4.88 GB | **4.48 GB** | **−0.40 GB** |
| rss | 3.81 GB | 3.60 GB | −0.21 GB |
| decode io / tok | 16.10 ms | **14.21 ms** | **−11.7%** |
| `head` / tok | 3.60 ms | **3.27 ms** | **−9.2%** |
| decode hit | 84.0% (51394/61200) | 84.0% (**51394/61200**) | **同一** |
| インストール | 14.71 GiB | **13.28 GiB** | −1.43 GiB |

反復のばらつきは 3 往復で t/s が ±0.15%、peak が ±0.01 GB。

読めること 3 つ:

- **`head` の −9.2% は 44 §6 が予測した通り。**20-M4.8 §3 の床測定
  「461 MB を 3.41 ms」は lm_head そのもので、そこはバイトで払っている。
- **io の −11.7% はバイト比 (−9.69%) より良い。**stride が小さいぶんミス 1 本の
  読みが短くなるだけでなく、33 §2 の「発行深さ = 層のミス数」が同じまま
  1 本あたりのバイトが減るので、帯域の効率も少し上がっている (**導出**)。
- **hit% が完全に同一**なのは、ルーティングが 1 ビットも動いていないことの
  直接の証拠である。38/41 が積み上げた `2^H` の議論はそのまま生きる。

## 5. 判定 — **採用する**

44 §6 が「採るのは `bias` 落としだけ」と書いた通りに入った。品質ゲートは
「差が無いこと」ではなく「**差が存在しないこと**」で通っている。

**既定スロットの再検討は本書では行わない。**32 スロットで peak が 0.40 GB 空いたが、
スロット数は 42 §1 が `Scripts/demo/serve.py` から取った運用点であり、
**要件を勝手に生やさない**。

## 6. 限界 (**未確認** / **未実装**)

- **ストリーミングインストールは `affine` のまま。**恒等式の検定には bias の
  バイトが要り、ストリーミングは「bias を書かないファイルの設計」を
  bias を取る前に決められない。**ローカルスナップショットからの repack
  (`--source-snapshot`) だけが `sym` になる。**README の既定導線は前者なので、
  ここは開いたままである。
- **drafter は `affine` のまま。**別リポジトリ由来の group-64 量子化で、
  恒等式は 29.35% の群で破れる (44 §1)。自分のコンテキストで自分のライブラリを
  持つので同居に問題は無い。
- **vision は無関係** (タワーは BF16)。
- **`checkRoutedMoE` は decode の gate/up カーネルを通らない** (§3a)。
  今回はモデルで捕まえたが、次は捕まらないかもしれない。
- **測ったのは 32 スロット × 1 プロンプト × temp 0 である。**42 の採点基準 v2
  (16 枚 / スコア 1.415) は回していない。**出力がバイト一致する以上スコアは
  定義上動かない**が、`t/s` 比の側は再走しないと言えない。

## 7. 再現手順

```bash
swift build -c release
# 形式の検定 (両スキーム × 両 group size で 69 ケース)
./.build/release/TurboFieldfareKernelCheck

# ローカルスナップショットから sym で入れる (プローブが自動で判定する)
./.build/release/TurboFieldfareRepack --output scratch/gemma4-qat-sym.gturbo \
    --source-snapshot scratch/qat-aligned-snapshot --overwrite
# → [repack] affine scheme — symmetric: 788175872 groups verified, ...

# 出力のバイト一致 (temp 0)
./.build/release/TurboFieldfareCLI --model <affine> --prompt "..." --temperature 0 \
    --max-new 256 --expert-cache-slots 32 > a.txt
./.build/release/TurboFieldfareCLI --model <sym> --prompt "..." --temperature 0 \
    --max-new 256 --expert-cache-slots 32 > b.txt && diff a.txt b.txt

# データを固定して導出だけ入れ替える (affine モデルを sym ライブラリで走らせる)
TF_FORCE_AFFINE_SCHEME=sym ./.build/release/TurboFieldfareCLI --model <affine> ...
```

`TF_FORCE_AFFINE_SCHEME` は**計器であって設定ではない**。恒等式を満たさない
チェックポイントに `sym` を強制すれば出力は壊れる。

## 8. 触ったファイル

| ファイル | 変更 |
| --- | --- |
| `TurboFieldfareFormat/GTurboAffineV1.swift` | 新規。`bias == -8*scale` の 1 か所きりの実装 (ランタイムと packer が共有する) |
| `Infrastructure/ModelIO/Quantization.swift` | `AffineScheme`、対称フィクスチャ `quantizeInt4Symmetric` |
| `Infrastructure/Metal/MetalContext.swift` | `setAffineScheme` / `canUseAffineScheme` / `TURBO_AFFINE_SYMMETRIC` |
| `Metal/{prefill,dequant_int4,moe,logit,tensorops}.metal` | ゼロ点をヘルパ経由に (19 か所)。**INT8 ルーターの 2 か所は据え置き** |
| `Runtime/Inference/{Model,ModelExpertIO,RealForwardRunner}.swift` | スキームの解決、bias 束縛の別名化、スキーマ検査 |
| `Infrastructure/ModelIO/ManifestReader.swift` | `sym` の受理と「4bit スロットは全部同じスキーム」の検査 |
| `Kernels/TensorCore/MPPPrefillInt4QMM.swift` | 別コンパイルの `tensorops` にもスキームを渡す |
| `TurboFieldfareRepack/Core/Planning/SymmetricProbe.swift` | 新規。恒等式の全数検定 |
| `TurboFieldfareRepack/Core/{Planning,Format,Remote}/` | `sym` レイアウトの計画・manifest 出力・ローカル入力の配線 |
| `TurboFieldfareKernelCheck/main.swift` | 全 INT4 ケースを両スキームで回す (`--affine-only` で従来どおり) |
| `bench/mtp45/` | 新規。一次ログ 4 本 |
