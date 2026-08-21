# 11. oQ4e の取得と g64 打ち直し (実測(手元)、2026-08-21 夜)

`scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp` を取得・検証し、本ランタイムに乗らない
248 本を上流 bf16 原本から 8-bit g64 で打ち直して
`~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64` を作った ([02 §3](02-CHECKPOINTS.md) の
案「公開版を打ち直す」の実行)。**GPU は使っていない** (量子化は CPU デバイス)。
**このチェックポイントはまだ 1 度も推論を通していない。**

---

## 1. oQ4e-mtp の取得と完全性

テンソル 2,052 本、`index.json` のキー集合と 5 シャードの safetensors ヘッダが完全一致。
テンソルデータ **21,613,054,816 B** は、事前に上流ヘッダから測った値
([02 §5](02-CHECKPOINTS.md)) と **1 バイトも違わない** (差の 270,763 B はヘッダぶん)。
`oq_imatrix_report.json` も同梱されている。

**取得速度:** `hf download` (hf_transfer 有効) で 21.6 GB が約 6 分。
単一ストリーム range GET の 11.3 MB/s ([10 §1](10-MLX4BIT-AUDIT.md)) は当たらなかった。

## 2. 打ち直しの実行

### 対象の確定

`config.json` の per-tensor override から
**`group_size ≠ 64` または `bits ∉ {4,8}`** を機械的に拾うと **ちょうど 248 本**。
bf16 換算 **1,460,928,512 B = 730,464,256 param** で、事前の上流ヘッダ測定
([02 §5](02-CHECKPOINTS.md)) と完全一致した。

### 入力は上流 bf16 の原本 (逆量子化の往復ゼロ)

**「5/6/8-bit を bf16 に戻したもの」ではない。**[10 §4](10-MLX4BIT-AUDIT.md) で作った
bf16 部分抽出に 181 本が既に入っていたので、HTTP Range で新たに取ったのは残り
**67 本 (1.195 GB)** だけ:
`linear_attn.out_proj` 30 / `in_proj_z` 29 / `in_proj_qkv` 4 / `self_attn.{q,k,v}_proj` 4。
**これで打ち直し対象 248/248 の bf16 原本が揃った。**
(`bf16-partial` の `oq4e_replacement_map.json` が oQ4e 名 → 上流名 / ファイル / bytes の
対応を持つ。)

打ち直しは `mx.quantize(w.astype(bfloat16), group_size=64, bits=8)`、CPU デバイス。

### 結果 — 予測どおり、赤リスト空

| | 予測 | 実際 |
| --- | ---: | ---: |
| 248 本のバイト | 533 MB → 776 MB (+243 MB) | **533,434,368 → 776,118,272 (+242,683,904)** |
| install 合計 | 約 21.86 GB | **21.86 GB** |
| メニュー外テンソル | 0 本 | **0 本** |

`config.json` の per-tensor override は 314 本すべて `8b/g64/affine`、base は `4b/g64/affine`。
`Quantization.supportedGroupSizes = [32,64]` と「routed expert は int4 のみ」の制約に対して
**赤リストが空になった。**

### 打ち直しの効き目 (上流 bf16 に対する相対 L2 誤差)

| テンソル | oQ4e | 打ち直し後 |
| --- | ---: | ---: |
| `layers.0.linear_attn.out_proj` (5b/g128) | 0.06628 | **0.00765** |
| `layers.3.self_attn.q_proj` (5b/g128) | 0.05647 | **0.00743** |
| `layers.20.linear_attn.in_proj_z` (5b/g128) | 0.05939 | **0.00800** |
| `layers.0.mlp.shared_expert.down_proj` (8b/g128) | 0.00856 | **0.00781** |

5-bit だったものは 8 倍近く改善して 8-bit の床 (0.0075 前後) に張り付いた。
**「oQ が 5〜8 bit に昇格させた 248 本を 4-bit に落とすのは感度測定の結論を
捨てることに等しい、上げる側に倒す」という判断が、数字の側から支持された。**
(4-bit g64 に落とせば −123 MB だが採らない。)

### ディスクの実装

シャード 1〜5 は `oQ4e-mtp` への hardlink。新規実体は打ち直した 744 本
(248 × weight/scales/biases) が入った `model-00006-of-00006.safetensors` (740.3 MiB)
だけ。増えたディスクは 740 MiB で、元リポジトリは無傷。シャード 1〜5 に残る旧 248 本の
508.7 MiB は index から参照されない。`metadata.total_size` は上流の流儀
(シャードのファイルサイズ和) だと死にバイトを含んで嘘になるので、
**index が参照するテンソルデータのバイト和**に変えてある (`total_size_convention` に明記)。
経緯は同ディレクトリの `README.md`。

## 3. これで候補が 2 本になった

「打ち直し 0 本」を理由に公式 MLX-4bit を第一候補に繰り上げた判断
([10 §6](10-MLX4BIT-AUDIT.md)) は、**打ち直しが実際には 1 回の CPU 作業で終わったので
理由が弱くなった。**現在の比較と未決定の理由は [02 §1](02-CHECKPOINTS.md)。
次の照合 (norm 規約 / conv1d 軸順 / router bit) は [04](04-PHASES.md) の「次の一手」。
