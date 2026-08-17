# 03. 設計判断 D1〜D8

前提: `02-RUNTIME-FIT.md` の N1〜N4。M0/M3 の結果が出るまでは**提案**であり、
実装の指示ではない。

---

## D1. 重みの置き場所 — vision と同じ型

- ドラフター重みは**別ファイル** (`draft_weights.bin`) に置く。テキスト側の
  `model_weights.bin` / `packed_experts/` には 1 バイトも触らない。
- `manifest.json` に optional な `draft` セクション + `flags.mtpDraft` + `versionMinor` ゲート。
  vision と同じ「フラグとセクションは 1 つの事実を 2 回書く。片方だけなら invalid」検証を書く
  (`GTurboManifestV1.swift:342-401` の写像)。
- **int4 版 (236 MB) をそのまま採用する。**bf16 (839 MB) は持たない。再量子化もしない。
- ロードは遅延。`Model.hasDraft` / `Model.draftWeights()` を `hasVisionTower` /
  `visionWeights()` (`Model.swift:118-140`) と同じ形で足す。**ドラフターを使わない実行の
  `load=` は増えない**ことを受入で測る (vision は 0.70→0.71 s で守れた)。

**理由:** 旧ランタイムが新しいモデルを黙って誤読することを防ぐ検査 (`flags`) と、
テキストのみ実行のコストゼロを、vision で 1 度実証している (`RESULTS_VISION.md` §8/§9)。

## D2. 追記インストール — `--add-draft`

`--include-draft` (新規インストール時) と `--add-draft --input-gturbo <model>`
(既存インストールへの追記) の 2 つ。後者は `VisionAppendInstaller.swift` の写像で、
**236 MB だけ取得し、15 GB のコピーを起こさない**。

受入で見るのは vision と同じ 3 点: 取得バイト数、テキスト側ファイルの (inode, mtime) 不変、
再検証 exit 0。

## D3. verify API — 既存プロトコルを広げず、新しいプロトコルを足す

```swift
protocol SpeculativeVerifier: LogitProducer {
    /// `tokens` を KV に追記しながら、各位置の logits を `logitRows` に書く。
    /// 行ストライドは vocab。greedy 構成では `logitRows` を書かず、
    /// k 個の argmax を `greedyTokens` に書く。
    func verifyBlock(tokens: ArraySlice<Int32>,
                     startPosition: Int,
                     into logitRows: MTLBuffer,
                     greedyTokens: MTLBuffer?) async throws
}
```

- 実装は**既存の prefill チャンク経路を k=2〜4 の 1 チャンクとして 1 本流す**。
  `executePrefillChunk` の `writeFinalHead` を「全行 head」に一般化するのが実体
  (`RealForwardRunner.swift:1517-1546`)。
- バッファ: FP16 logits は 512 KiB/行。bs=4 でも 2 MiB。`RawCompletionScratch`
  (`RawCompletion.swift:42-63`) にブロック用の行バッファを 1 本足す。
- **prefill 経路をそのまま呼ぶこと自体が回帰リスク**なので、「verify の全位置 logits が
  1 トークンずつ `produce` した logits と一致する」検査を M4 の出口条件に置く。

> `ChunkedPrefillRunner` に生やさない理由: あれはプロンプト前処理の契約
> (`prefillChunkState.requireClean`、`startPosition == kv.position`) を持っており、
> 生成中に何度も呼ばれる verify とは寿命が違う。

## D4. KV 巻き戻し

```swift
extension KVCacheManager {
    /// 受理長まで論理カーソルを戻す。物理スロットは position % capacity の
    /// ステートレスな写像なので、戻すのは position だけでよい。
    public func rewind(to position: Int)
}
```

- 事前条件を 2 本置く: `0 <= position <= self.position`、および
  **`capacity(layer:) >= slidingWindow + maxDraftBlock`** を init で検査
  (`02-RUNTIME-FIT.md` N2 の導出条件)。既定 (chunk 128 → capacity 1152) では自動的に
  成立するが、`--prefill-chunk-tokens 32` のような設定で margin を超えないことを明示する。
- 「巻き戻した後の生成が、巻き戻さなかった世界と一致する」検査と、
  **わざと壊して FAIL する**検査 (rewind をノーオペにする / 1 だけずらす) を M4 に置く。

## D5. 受理規則 — same-seed sample-and-compare

各検証位置 i (受理接頭辞の直後) について:

1. ターゲットの logits から、**非投機実行と同じ `position` (= 生成インデックス)、
   同じ history、同じ config** で `Sampler.sample(...)` を 1 回引く。
2. 引いたトークンがドラフトの提案と一致 → 受理して次の位置へ。
3. 不一致 → **ターゲットが引いたトークンを採用して**そのラウンドを終える。
   以降のドラフトは捨て、KV を巻き戻す。

**この規則の下では、出力は同一 seed の非投機実行とトークン単位で一致する** (**導出**):
`Sampler` は `position` から決定的にシードを作り (`Sampler.swift` の
`seedFor(config:position:)`)、`RawCompletion.swift:273-282` が生成インデックスを
`position` として渡している。全ドラフトが受理された位置の logits は非投機実行と同一
(KV の中身が同一だから) なので、同じ seed で引けば同じトークンが出る。不一致位置で採用する
トークンも、非投機実行が引いたはずのトークンそのもの。

→ **temp 0 でも temp 1.0 でも出力が変わらない。**受入ゲートが「同じ seed で 2 回走らせて
diff が空」という機械判定になる。代償は、最適な rejection sampling より受理率が低いこと。

- 採らない案: residual 分布からの再抽選 (最適な rejection sampling)。受理率は上がるが、
  `Sampler` に残差分布サンプリングを新設する必要があり、**出力一致という強い受入ゲートを失う**。
- repetition penalty は既定 1.0 (`Sampler.swift:14`) だが、有効時は history を位置ごとに
  正しく延ばす必要がある。実装が面倒なら **`repetitionPenalty != 1.0` では MTP を無効化**して
  素直に落ちる (D7 と整合)。

## D6. ループの配置

`runRawCompletion` の while (`RawCompletion.swift:204-258`) に分岐を足すのではなく、
**`runSpeculativeCompletion` を別関数**にし、prefill までの前段 (`:95-197`) を共有関数に括り出す。

1 ラウンド:

```
bonus トークン t0 (前ラウンドの受理末尾、または prefill seed)
ドラフト: bs−1 個を自己回帰で生成 (ターゲット層 28/29 の KV に attend)
verify:   ドラフト列を verifyBlock で 1 回流す
受理:     D5 の規則で受理接頭辞 a (0 ≤ a ≤ bs−1) を決める
巻き戻し: kv.rewind(to: 受理後の位置)
放出:     受理トークンを 1 個ずつ detokenizer → stop matcher → onProgress(.token)
停止:     受理列の途中で停止したら、その位置で打ち切り KV も合わせる
```

- ストリーミングの粒度はラウンド単位になる (最大 bs 個がまとめて出る)。
  UI 上は「少しまとまって出る」挙動になるので、受入で目視する。
- `.token` の `index` は連番を維持する (クライアントが数えている)。

## D7. CLI / Server の露出

| 面 | 追加 | 既定 |
| --- | --- | --- |
| CLI | `--draft-block-size <0\|2\|3\|4>` (0 = 無効) | **当面 0**。M5 の実測が出るまで既定を変えない |
| Server | 同名フラグ。リクエストごとの切り替えは**しない** | 0 |
| Mac アプリ | M6 まで露出しない | ― |

無効化して素直に落ちる条件 (設計時点で決めておく):

- モデルに `flags.mtpDraft` がない → フラグを渡してもエラーで停止 (黙って無効化しない)。
- `repetitionPenalty != 1.0` → 無効 (D5)。
- 画像つきリクエスト → 初回は**無効**。共有 KV は層 28/29 なので原理的には両立するが、
  vision と MTP を同時に検証しない (スコープを 1 つに保つ)。
- プロンプトキャッシュ: 投機で進んだ位置を publish しない。公開するのは**受理長まで
  巻き戻した後の `kvBackedTokenIDs`** に限る。既存の「画像ターンは publish しない」と
  同じ場所に置く検査で守る (`RawCompletion.swift:115-122`)。

## D8. テレメトリ

footer に 1 行足す (`bench.sh` が拾える形):

```
[mtp block=3 rounds=128 accepted=241/256 accept=0.94 drafter=1.9ms/step verify=8.4ms/round]
```

- `accept` = 受理トークン数 / ドラフト提案数。これが**唯一の意思決定用数値** (05-RISKS R2)。
- `drafter` / `verify` の内訳がないと「受理率は高いのに速くならない」ときに原因が切れない。
- 既存の `[decode/tok io=… cb1=… cb2=… head=…]` はそのまま残す。
