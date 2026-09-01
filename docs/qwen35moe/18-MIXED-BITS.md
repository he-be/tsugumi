# 18. 混在ビット幅の受け入れ — 本線が開いた (実測(手元)、2026-08-21)

[04-PHASES.md](04-PHASES.md) 次の一手 **#11**。[13 §4-2](13-PHASE1-REPACK.md) が
Phase 3 の宿題として残した「`quant.attention` のスロット 1 個では `oQ4e-g64` を
検証できない」を片づけた。**GPU カーネルは 1 本も動かしていない** (モデルを開く
だけなので Metal デバイスは要るが、計算はしない)。

```
swift run -c release TurboFieldfareKernelCheck --qwen-open scratch/ornith-oq4e-g64.gturbo
```

| | |
| --- | --- |
| 結果 | **本線の `oQ4e-g64` が `Model.load` を通る。**1,294 ms (うち sha256 998 ms) |
| 決めたこと | **索引から導く。**スロットは幅の**上限**を述べるだけにした (§1) |
| 検査 | 新規 8 本 (`QwenRuntimeSchemaTests`、うち 5 本は負例)。**直列実行で 1,273 件が緑** |
| 副産物 | `ArchConfig.ornith1_5_35B_A3B`、Qwen 用の常駐スキーマ、`--qwen-open` |

---

## 1. 「索引から導く」を選んだ

[13 §4-2](13-PHASE1-REPACK.md) は逃げ道を 2 つ挙げていた。**索引から導く**方を採った。

| | 索引から導く (採用) | manifest のスロットを層ごとに割る |
| --- | --- | --- |
| 形式変更 | **不要** | `quant` に層数ぶんの配列が要る (v1 の minor をもう 1 つ) |
| 真実の在処 | 常駐索引 (`sizeBytes` と `shape`) | manifest と索引の 2 か所。食い違ったら? という問いが増える |
| 結線のとき | カーネルを選ぶ側が**テンソル名で**幅を引ける | 層番号からスロットを引く経路を別に作る |
| 弱くなる検査 | スロットとの**厳密一致**が使えなくなる (§2) | なし |

決め手は「導出に曖昧さが無い」こと。**4-bit と 8-bit はちょうど 2 倍**離れているので、
形が分かっているテンソルの packed バイト数は幅を一意に決める。どちらでもない数は
第 3 の幅ではなく壊れた索引である。`Model.residentWeightBits(_:)` がこれ 1 行で、
Phase 3 の結線と INT8 LM head ([17 §5](17-PHASE2-KERNELS.md)) はここに乗る。

## 2. 弱くならないように、スロットを**上限**として使う

厳密一致をやめると、検査の力がその分落ちる。落ちきらないように 3 つ足した:

1. **スロットは上限。**`quant.attention.weightBits` は「このスロットで一番広い
   テンソルの幅」という意味に変えた。8-bit のテンソルが 4-bit 宣言のスロットの
   下にあれば `Model.load` は落ちる (負例 `rejectsATensorWiderThanItsSlotDeclares`)
2. **緩めるのは族を見てから。**`validateQuant` の許容表は
   `embedding: [4]` / `attention: [4]` のままで、**8 を足すのは `arch.family` が
   ある族だけ**。Gemma 4 の embed は INT4 専用の lookup カーネルが読むので、
   そこに 8 を通すと落ちずに壊れる。Gemma の manifest は 1 バイトも、
   検査の厳しさも動いていない
3. **導出の対象を限る。**`widthVariesByTensor` は Qwen の attention スロット
   (`self_attn.*` と `linear_attn.*`) にだけ立てる。embed / lm_head / shared
   expert / router は今までどおりスロットと厳密一致

repack 側も直した。スロットの幅は**最後に見たテンソル**ではなく**最大**を書く
(`RemoteStreamingRepacker.writeManifest`)。今までは層の走査順に依存していて、
たまたま 8 が最後だっただけだった。**一様な checkpoint では最大 = その幅**なので、
Gemma の manifest も既存の `ornith-oq4e-g64.gturbo` も出力は変わらない
(再 repack は要らなかった)。

## 3. 実物の幅の分布 (実測(手元))

`--qwen-open` が索引から数えたもの。**15 ロール中 5 つが 1 つのスロットの中で
幅を混ぜている。**

| ロール | 4-bit | 8-bit |
| --- | ---: | ---: |
| `self_attn.q_proj` | 5 | 5 |
| `self_attn.k_proj` | 5 | 5 |
| `self_attn.v_proj` | 5 | 5 |
| `self_attn.o_proj` | 6 | 4 |
| `linear_attn.in_proj_qkv` | 25 | 5 |
| `linear_attn.{in_proj_z, in_proj_a, in_proj_b, out_proj}` | 0 | 30 ずつ |
| `mlp.shared_expert.{gate,up,down}_proj` / `shared_expert_gate` | 0 | 40 ずつ |
| `embed_tokens` / `lm_head` | 0 | 1 ずつ |

[13 §4-2](13-PHASE1-REPACK.md) の表と一致する (あちらは 8-bit 側に MTP の 1 本を
数えていた。repack は MTP を外すので、ここでは `o_proj` が 6/4)。

## 4. Qwen の常駐スキーマ

`validateRuntimeSchema` は Gemma の綴り (`pre_feedforward_layernorm_2` /
`router.proj` / `layer_scalar`) を直に並べていたので、**混在ビット幅より前に
テンソル名で落ちていた**。族で分岐させ、共通部分は `ResidentSchemaChecker` に
出した (`Sources/TurboFieldfare/Runtime/Inference/RuntimeSchema.swift`)。
routed expert の区画は綴りが同じなので 1 つのまま両族が使う。

Qwen 側で Gemma と違うのは 3 つ:

- **30 層に K/V が無い。**`linear_attn.*` 9 本 (うち 5 本が affine) を、
  `manifest.arch.linearAttention` の幾何に対して検証する。`ArchConfig` の
  head 系の値は残り 10 層のものなので使えない
- **`q_proj` が倍幅。**後半は `attn_output_gate` ([01 §3-2](01-MODEL.md))。
  等倍のテンソルを通すと、ゲートを隣のテンソルから読んで走ってしまう
  (負例 `rejectsAQueryProjectionWithoutItsOutputGate`)
- **embed が tie されていない**ので `lm_head` が別テンソル。本線ではこれが 8-bit
  ([17 §5](17-PHASE2-KERNELS.md))

`arch.linearAttention.layerCount` と `fullAttentionLayerMask` の食い違いも
ここで弾く (状態のサイズを間違えても落ちない値なので)。

## 5. 負例 5 本

正例が通るだけでは物差しが働いている証明にならない ([17 §2](17-PHASE2-KERNELS.md)
と同じ作法)。**このモデルで静かに壊れる道**を索引側に作って、全部落ちることを見た:

| 負例 | 何を間違える |
| --- | --- |
| `conv1d` の軸を入れ替える | `[C, K, 1]` を `[K, C, 1]` に。**バイト数は同じ**なので形しか手掛かりが無い。カーネル側の同じ負例は相対誤差 0.83 ([17 §2](17-PHASE2-KERNELS.md)) |
| `q_proj` を等倍にする | 出力ゲートの半分が消える |
| 幅が 4 でも 8 でもない | packed バイト数を 3/4 に。第 3 の幅ではなく壊れた索引 |
| スロットより広いテンソル | 4-bit 宣言の下の 8-bit |
| 再帰層のテンソルを 1 本抜く | `conv1d` / `A_log` / `in_proj_a` / `lm_head` の 4 通り |

族の分岐そのものにも負例がある: Gemma の manifest で embed を 8-bit に書き換えると
落ち、同じ値が Qwen の manifest では通る (`rejectsAGemmaManifestThatWidensTheEmbedding`)。

## 6. この Phase が動かした結論

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) 次の一手 #11 | **完了。**本線が `Model.load` を通る |
| #15 (INT8 LM head) | **着手できる。**幅はテンソルごとに引けるようになった |
| #14 (Phase 3 の結線) | 障害は取れた。残るカーネルは LM head だけ |
| [13 §4-2](13-PHASE1-REPACK.md) | 「Phase 3 の設計で決める」→ **索引から導くに決めた** (§1) |
| 形式 v1 | **変更なし。**`.gturbo` のバイトは 1 つも動いていない |

## 7. 途中で分かったこと

**パッケージテストは並列だと落ちる。**`swift test` を素で回すと、偽 HTTP サーバーを
使う remote install 系 (Draft / Vision / RemotePayloadCopy) の 20 本前後が
`remote HTTP 404` で落ちる。**変更前の tree でも同じ集合が落ちる**ので今回の作業とは
無関係だが、[13](13-PHASE1-REPACK.md) の「1265 件が緑」は**直列実行での話**である。
`swift test --no-parallel` なら **1,273 件すべて緑**。原因は `FakeHFURLProtocol` の
登録が並列のテストどうしで共有されることだと見えるが、直していない。
