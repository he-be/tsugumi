# 45 の受け入れを通しで再走した記録 (2026-08-20 15:00〜15:40)

`6abb72a` (W2、`sym` の採用) を**インストールから生成・サーバー・テストまで通して**確認した
一次ログ。45 本文の測定とは**別の走り**で、同じ機体・同じコミットである。

- 機体: M3 Pro 18GB / macOS 15.7.5 (24G624) / Swift 6.3.3
- コミット: `6abb72a`、ブランチ `macos15-support`
- `swift build -c release` はキャッシュ済み (4.0s) の状態から開始

## 結論

**45 §0 の 4 つはすべて再現した。機能上の破れは 1 つも出ていない。**
数値の一致 (バイト一致・hit 一致・sha256 一致) は決定的に一致し、
速度は +9.48% (45 の報告は +8.8%) で方向と桁が一致した。

## ファイル

| ファイル | 中身 |
| --- | --- |
| `01_kernelcheck.log` | `TsugumiKernelCheck` **69/69 PASS**。group 64/32 × affine/sym + vision。`../kernelcheck_both_schemes.log` とケース構成が一致 |
| `02_repack_from_snapshot.log` | `--source-snapshot` からの repack。35.2 秒、`symmetric: 788175872 groups verified` |
| `03_verify_install.log` | `--verify-install` 37 files / 14,299,490,432 B |
| `04_install_hashes.log` | 出来たモデルが出荷済み `gemma4-qat-sym.moepack` と **37 ファイル全部の sha256 一致** (repack は再現可能) |
| `05_greedy_equivalence.log` | temp 0 / 256 tok / 32 スロット。**3 プロンプト 5 ペアすべてバイト一致**、decode hit も同一。`TF_FORCE_AFFINE_SCHEME=sym` の導出検定も一致 |
| `06_ab_tps_32slots_rerun.log` | 運用点 (32 スロット) の A/B、3 往復・クールダウン 20 秒。**静穏時に取り直した本命の系列** |
| `07_ab_rerun_analysis.log` | 上の中央値と差分 |
| `08a_add_draft.log` / `08_mtp_ab.log` | sym target (group 32) × affine drafter (group 64) の同居。`--draft-block-size 4` で **バイト一致**、rounds/accept/accepted 分布まで同一 |
| `09_long_prefill_ab.log` | prefill 702 tok の A/B。**tensor-core INT4 経路 (`MPPPrefillInt4QMM`) は `tokenCount >= 32` でしか発火しない**ので、短いプロンプトだけでは踏めない経路をここで踏んでいる |
| `10_server_smoke.log` | `TsugumiServer` に sym を載せた smoke (`/health`, `/v1/models`, `/v1/chat/completions`) |
| `11_swift_test_summary.log` | `Scripts/test.sh` 945 tests / 8 issues。**全部 "C2 prompt token invariants"** (既知の意図的な赤)。新規の破れなし |
| `generations/` | 上の全生成の生出力 (footer 込み)。`gen_*` = 等価性、`rerun_*` = 静穏時 A/B、`mtp_*`、`long_*` |
| `driver_equivalence.sh` / `driver_ab_rerun.sh` | 使ったドライバ (プロンプトはこの中にある) |

## 数字

| | affine | sym | 差 | 45 §4 |
| --- | ---: | ---: | ---: | ---: |
| tok/s | 23.094 | **25.284** | **+9.48%** | +8.8% |
| peak | 4.88 GB | **4.51 GB** | **−0.37 GB** | −0.40 GB |
| rss | 2.73 GB | 2.60 GB | −0.13 GB | −0.21 GB |
| io / tok | 16.31 ms | 14.29 ms | −12.4% | −11.7% |
| `head` / tok | 3.56 ms | 3.25 ms | −8.7% | −9.2% |
| decode hit | 51798/61200 | 51798/61200 | **同一** | 同一 |

インストール容量は `verified-install.json` の text-only 合計で
**14.748 → 13.317 GiB (−1.430 GiB / −9.70%)**、`expertStride` は 3,358,720。
45 §0-3 の「14.71 → 13.28 GiB」とは基準ファイルの取り方が違うが、**差の −1.43 GiB は一致する**。

## この走りの限界 (**未確認**)

- **t/s のばらつきが 45 より大きい。**3 往復で affine 2.78% / sym 3.93% (45 は ±0.15%)。
  中央値の差 +9.48% はこの幅より十分大きいが、**同じ精度の測定ではない**。
- **`08` と `09` の t/s は参考値である。**この 2 本を取った 15:19〜15:27 は、
  **別のセッションが同じ機体でモデルを走らせていた** (`bench/mtp46/` の生成時刻と重なる)。
  バイト一致・hit 一致・accepted 分布の一致は決定的なので影響を受けないが、
  壁時計は汚染されている可能性がある。`06` はその後、静穏を確認してから取り直した。
- **通していない経路**: ストリーミングインストール (45 §6 の既知の限界で `affine` のまま)、
  vision タワー × `sym` (出荷済み sym モデルにタワーが無い)、Mac アプリの GUI。
- MTP の検定に使った sym + drafter のモデルは検証用の複製で、**確認後に削除した**。
  再現するには `--add-draft --input-moepack <sym モデル>` (236 MB、約 66 秒)。

## ついでに気づいたこと

`../greedy_equivalence.log` と `../ab_tps_32slots.log` には**使ったプロンプトが記録されていない**。
本ディレクトリのドライバには書いてある。
