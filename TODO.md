# TODO

## macOS 15 フォールバックの性能影響を実機で確認する

### 状況

`103bfbc Lower the deployment target to macOS 15` で、macOS 26 必須だったビルドが
Sequoia でも起動するようになった。代償として macOS 15 では MSL 4.0 テンソルカーネルが
シェーダライブラリから落ちる。この Mac (M3 Pro / 18GB / macOS 15.7.5) で
実際にどれだけ遅くなるかは **未測定**。まず手動で動かしてから判断する。

### 実機の状態

| 項目 | 値 |
| --- | --- |
| チップ | Apple M3 Pro (CPU 12 コア / GPU 18 コア) |
| メモリ | 18 GB |
| OS | macOS 15.7.5 (24G624) |
| GPU family | apple9 = true, **apple10 = false** |
| シェーダ | MSL 3.2 でコンパイル (44 関数、テンソルカーネルなし) |
| モデル | **未インストール** (`scratch/` に install.lock のみ) |

### 影響の切り分け

3 つを混同しないこと。

| 対象 | 効果 | 入手方法 |
| --- | --- | --- |
| MPP 行列カーネル (`mpp_prefill_affine_threadgroup_f16`) | prefill 約 12% | **macOS 26 に上げれば戻る (無料)** |
| TensorOps attention (`attention_prefill_full_tensorops_2d_validity_v2`) | 長文 prefill で最大 2.4x | **M5 以降のハードが必要。M3 Pro は apple10 非対応なので macOS 26 にしても有効にならない** |
| decode | 影響なし | — |

- MPP パスは GPU family ゲートがなく、パイプライン生成の成否だけで決まる
  (`Kernels/TensorCore/MPPPrefillInt4QMM.swift:16-31`)。macOS 15 では関数が
  ライブラリに存在しないので `pipeline = nil` になり `.unavailable` を返す。
- TensorOps attention は `device.supportsFamily(.apple10)` で明示的にゲートされている
  (`Kernels/Attention/PrefillAttention.swift:51`)。**この Mac では macOS 26 で
  動かしていても最初から使われていなかった**。今回のコミットによる損失ではない。
- 影響を受けるのは `tokenCount >= 32` かつ `.q / .kv / .o` の投影のみ
  (`Runtime/Inference/RealForwardRunner.swift:646`)。shared/routed expert、MoE、
  router、attention 本体、LM head、decode 全体は影響なし。
- フォールバック先は Q が `.repeatedGEMV`、KV/O が `.qmm`。

### 12% の根拠

`docs/experiments/summaries/06-prefill.md` の PF-12（MPP を production に入れた実験）:

- M128 投影ワーク加重: 約 73.8% 改善
- 512 トークン prefill: 11.421% 改善 / TTFT: 10.947% 改善

逆算して prefill・TTFT が約 12〜13% 遅い計算。**ただしこれは PF-12 を測った機体の値で、
M3 Pro での再測定ではない。**

### README ベンチの読み方の注意

「8GB M2 = 5.1-6.3 tok/s」「24GB M5 Pro = 31-35 tok/s」はチップ世代と RAM 量が
同時に変わっているので、M3 Pro の予測には使えない。M2 側の内訳は 1 トークン 162.8ms の
うち **83.1ms が SSD からの expert 読み込み**で、半分以上が GPU ではなくストレージ待ち。
18GB あればウォームアップ後は大半がファイルキャッシュに乗るため、M2 の数字は当てはまらない。

## 手順

- [ ] モデルをインストールする（約 15GB ダウンロード）

  ```bash
  swift run -c release TurboFieldfareRepack \
    --discard-partial \
    --output scratch/gemma4.gturbo
  ```

  終わったら検証:

  ```bash
  swift run -c release TurboFieldfareRepack \
    --verify-install \
    --input-gturbo scratch/gemma4.gturbo
  ```

- [ ] macOS 15 のまま CLI でベースラインを測る。タイミングは stderr に出る。
      短い / 中くらい / 長いプロンプトの 3 点を、それぞれ新しいプロセスで。
      ウォームアップを 1 回先に走らせてから 3 回測って中央値を取る
      （`docs/BENCHMARKS.md` の M5 と同じ手順）。

  ```bash
  swift run -c release TurboFieldfareCLI \
    --model scratch/gemma4.gturbo \
    --messages-file messages.json
  ```

  記録するもの: prompt トークン数 / prefill / TTFT / decode tok/s / peak RSS

- [ ] 体感を確認する。prefill の 12% が実際に気になるかどうかが判断材料。

- [ ] 気になるなら **macOS 26 にアップグレード**して同じ 3 点を測り直す。
      MPP パスが戻るので prefill が改善するはず。TensorOps attention は
      apple10 ゲートなので M3 Pro では戻らない。

- [ ] ここまでの実測を見てから M5 Pro 買い替えを検討する。M5 が本当に効くのは
      数千〜数万トークンの長いプロンプトを日常的に投げる場合。短い会話用途なら
      差はメモリ帯域幅ぶんに留まる。

## 測ったら埋める

| プロンプト / 生成 | prefill | TTFT | decode | peak RSS |
| --- | ---: | ---: | ---: | ---: |
| | | | | |
| | | | | |
| | | | | |

## 検討事項（測定後）

- `PrefillProjectionDispatchPolicy.selectedDispatch` で `.q` が `.repeatedGEMV` に
  なっているのは MPP 導入以前の測定に基づく選択
  (`Runtime/Inference/RealForwardRunner.swift:118-131`)。macOS 15 を常用するなら
  Q 投影を `.qmm` にしたほうが速い可能性がある。要測定。
- 測った M3 Pro の値は `docs/COMMUNITY_BENCHMARKS.md` に投稿できる。
  現状 M3 Pro / macOS 15 のデータポイントは存在しない。
