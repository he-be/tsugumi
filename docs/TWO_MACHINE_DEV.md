# MBP と M6 の 2 機で開発する

2026-09-22 から、開発機が 2 台になった。**機械ごとに OS もツールチェーンも SSD も違い、揃えることができない。**
揃わないまま正しく使うための取り決めを書く。

| | **MBP** (母艦) | **M6 mini** (ヘッドレス) |
| --- | --- | --- |
| 機種 | Mac15,6 / M3 Pro / 18GB | Mac18,5 / M6 12 コア / 16GB |
| OS | macOS 15.7.5 (24G624) | macOS 27.0 (26A428) |
| ツールチェーン | Xcode 26.6 / Swift 6.3.3 / SDK 26.5 | **CLT のみ** / Swift 6.4 / SDK 27.0 |
| SSD | AP1024Z、読み 6.7 GB/s | AP0256Z、読み **3.3 GB/s** ([M6_SSD_BANDWIDTH](investigations/M6_SSD_BANDWIDTH.md)) |
| 外付け | — | TB4 / APFS `SSD256`、238 GB 空き |
| 接続 | — | `ssh m6` (Tailscale) / `m6-lan` (有線 1GbE) / `m6-tb` (TB4 直結、常設しない) |

**MBP は macOS 15 なので Xcode 27 系を載せられない。M6 は headless なので Xcode を GUI で入れる意味がない。
この差は解消しない。** 揃える努力をやめて、役割を分けて使う。

## 1. 役割 — 性能の記録値は M6 でしか取らない

| | MBP | M6 |
| --- | --- | --- |
| やる | 移植と正しさ（参照器との一致、fixtures、`swift test`）、Instruments / GPU キャプチャ、**macOS 15 の回帰** | **性能の記録値**（[m6-prefill/04-GATES](m6-prefill/04-GATES.md) の G0〜G6。証拠は `bench/m6/results/`、規則は [m6-prefill/02-EVIDENCE](m6-prefill/02-EVIDENCE.md)。旧 [M6 文書](investigations/M6_MAC_MINI_TARGETING.md) は凍結）、macOS 27 / Metal 4 tensor ops、Swift 6.4 でのビルド確認 |
| やらない | 記録に残す性能値 | GUI が要る作業 |

SSD は MBP が 2 倍速く、RAM は 2 GB 多い。**この 2 機の数字を混ぜた瞬間に比較が死ぬ。**
[33](mtp/33-M8-IO-FLOOR.md) の冒頭で「サンプル画像を差し替えたので 31 / 32 と比較できない」と断ったのと同じことが、
今度は機械の差で起きる。同じ条件どうしでしか比べないこと。

## 2. ブランチは切らない。分岐は実行時に置く

**OS ごとのブランチを作らない。** 既に答えは出ていて、`103bfbc`（Lower the deployment target to macOS 15）が
`.macOS(.v26)` を `.v15` に下げ、**MSL のバージョンを実行時に選ぶ**形にした。
[CONTRIBUTING.md](../CONTRIBUTING.md) の「Metal 4 / MSL 4.0 tensor kernels are allowed only behind their
`__HAVE_TENSOR__` shader guard plus a Swift fallback path」がその方針である。

M6-0（`supportsFamily(.apple10)` の静的ゲートをパイプライン生成の成否に置換）も同じ方向の仕事であって、
**M6 専用ブランチの代わりにやること**である。main 一本で両機が動く状態を保つ。

## 3. 同期 — MBP で編集し、M6 へ push する

作業ツリーを 2 つ育てると、docs の「どの機械のどのビルドの数字か」がすぐ崩れる。**編集は MBP だけで行う。**

```bash
# 一度だけ（設定済み）
ssh m6 'git init -b main ~/dev/tsugumi && git -C ~/dev/tsugumi config receive.denyCurrentBranch updateInstead'
git remote add m6 m6:dev/tsugumi

# 1 サイクル (手で回す形。prefill の計測は bench/m6/cycle.sh が push → build → 計測 → 証拠の引き戻しまでやる)
git push m6 HEAD
ssh m6 'cd ~/dev/tsugumi && swift build -c release && .build/release/TsugumiCLI ...'
```

- `receive.denyCurrentBranch updateInstead` は、**M6 側の作業ツリーが綺麗なときだけ** push を受けて checkout まで進める設定。
  M6 で直接編集していると push が弾かれる。それでいい（編集は MBP だけ、を機械が守ってくれる）。
- `.build` は **機械ごとに別物**（Swift 6.3.3 と 6.4）。共有もマウントもしない。
- GitHub の `origin` は真実の置き場として維持する。M6 は `origin` を見る必要がない。
- TB4 直結（`m6-tb`）はケーブルを常設しないので**当てにしない**。繋いだときだけ速い経路として使う
  （生 TCP 37.8 Gb/s、ssh は `aes128-gcm` 指定で 2.15 GB/s。既定の chacha20 だと 0.39 GB/s しか出ない）。

## 4. 測定値にはホスト札を機械的に付ける

`bench/hostinfo.sh` が 1 行を吐く。**結果ファイルの先頭に必ず貼る。**

```
$ bench/hostinfo.sh .build/release/TsugumiCLI
測定: 2026-09-22 / Mac15,6 Apple M3 Pro 18GB / macOS 15.7.5 (24G624) / Swift 6.3.3 / SDK 26.5 / Xcode 26.6 / APPLE SSD AP1024Z / commit 2903532 / minos 15.0
```

`minos` を含めるのが肝である。旧 [M6 文書](investigations/M6_MAC_MINI_TARGETING.md) §7 の #5（deployment target を戻せるか）を
やるとバイナリの性格が変わるので、後から必ず「どっちのビルドの数字か」を問われる。

## 5. ストレージ — 内蔵は測定対象、外付けは置き場

**M6 の内蔵 SSD は測定対象そのものである。**だから:

- **測定は必ず内蔵から走らせる。** 外付け `SSD256` や、TB4 越しのマウント、ネットワーク共有から重みを読んで測らない。
  読んだ瞬間に [M6_SSD_BANDWIDTH](investigations/M6_SSD_BANDWIDTH.md) の 3.34 GB/s が意味を失う。
- **外付け `SSD256`（238 GB 空き）は、退避と受け渡しに使う。** 大きいファイルを M6 へ持ち込む / 逃がすのはこちら。
  内蔵の空きが本当に足りないときの逃がし先でもある。
- **重みを M6 に常駐させない。** 測定対象だけ置いて、終わったら消す。
  MBP の `scratch` は 39 GB（gemma 15 + ornith 19）、Flash-Next を載せるなら install 78 GB + PLE 29 GB = 107 GB で、
  内蔵 147 GB 空きに全部は同居できない。

> **ヘッドレスでは外付けが自動マウントされない。**GUI ログインが無いので Disk Arbitration が上げない。
> `ssh m6 'diskutil mount disk5s1'` で `/Volumes/SSD256` に付く（sudo 不要）。再起動のたびに要る。

現状の内蔵の使い方（2026-09-22）: 空き 147 GB。`~/Library/Caches/mflux` が 17 GB、`~/dev/mflux` が 1.3 GB。
測定で数十 GB 要るときは、まずこの 18 GB を外付けへ逃がすのが順当（`HF_HOME` を `/Volumes/SSD256` に向ける）。

## 6. ビルドに Xcode は要らない

この package はシェーダを実行時にコンパイルする（`MetalContext.swift` の
「edit a shader, rebuild the Swift target, no Xcode metallib step」）。`.metal` はリソースとして同梱され、
`device.makeLibrary(source:)` で読む。したがって **CLT の Swift だけで `swift build -c release` が通る**見込みで、
`metal` コンパイラが CLT に無いことは問題にならない（**実測**: M6 の CLT で `swift build -c release --product TsugumiCLI` は通る。2026-09-22）。

Xcode が要るのは Instruments の Metal System Trace と GPU フレームキャプチャだけで、それは MBP の仕事である（§1）。
どうしても M6 に入れるなら GUI は不要で、`xcodes` CLI か、MBP の `/Applications/Xcode.app`（3.7 GB）を送ればよい。
ただし Xcode 26.6 が持つ SDK は 26.5 で、**CLT の 27.0 より古い**。`xcode-select` は CLT のままにしておくこと。
