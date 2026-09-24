# M1 Mac mini で Gemma 4 26B-A4B を動かす

2026-09-24 起草。M1 Mac mini (16 GB、ヘッドレス) でサーバが動くかを確かめた記録。
目的は「Apple が M1 比で何倍と言う、その M1 で実際に動くか」を記事のために確かめること。
**ここにある数字は動作確認の 1 回ずつの値で、性能の記録値ではない。** 記録値は M6 でしか取らない ([TWO_MACHINE_DEV](../TWO_MACHINE_DEV.md))。
M1 はこの先、GUI を使わないサーバ専用機として使う。

## 1. 機械

| | M1 mini |
| --- | --- |
| 機種 | Macmini9,1 / Apple M1 (CPU 4P+4E、GPU 8 コア) / 16 GB |
| OS | macOS 27.0 (26A428)、GUI ログイン無し |
| ツールチェーン | CLT のみ / Swift 6.4 / SDK 27.0 |
| SSD | APPLE SSD AP0256Q (251 GB) |
| 接続 | `ssh mini` (Tailscale) / `mini-lan` (192.168.0.84) |
| wired limit | `iogpu.wired_limit_mb = 0` (既定のまま) |
| repo | `~/dev/tsugumi` (MBP から `git push m1`、M6 と同じ `updateInstead`) |
| モデル | `~/dev/tsugumi/scratch/gemma4-qat-sym.gturbo` (MBP から rsync) |
| サーバのバイナリ | `~/tsugumi-bin/` (MBP でビルドしたもの。§3) |

## 2. 準備でやったこと

1. **Irodori-TTS を止めた。** `gradio_app.py` が常駐していた (RSS 約 1.9 GB)。ユーザーの許可あり。別ユーザーの `korean-notes-v2` サーバは残した。
2. **audiomxd の CPU ループを止めた。** GUI ログインが無いと BT の Smart Routing が
   `UpdateAudioState failed to start XPC: kUnexpectedErr (No user logged in)` を繰り返し、
   `audiomxd` が約 72%、`configd` が約 43% の CPU を使い続けていた (再起動しても続く)。
   M6 で 2026-09-23 に効いたのと同じ機能フラグを置いて再起動し、止まった。

   ```
   sudo mkdir -p /Library/Preferences/FeatureFlags/Domain
   sudo defaults write /Library/Preferences/FeatureFlags/Domain/BluetoothFeatures.plist SmartRoutingMacOS -dict Enabled -bool NO
   sudo shutdown -r now
   ```

   再起動後、`configd` は 0%。`audiomxd` は launchd が 5 秒おきに起動しては終わるだけで CPU はほぼ 0 (M6 も同じ出方)。
   BT の電源を切る (`ControllerPowerState 0` + `killall -HUP bluetoothd`) は効かなかったので戻した。
   元に戻すには plist を消して再起動する。GUI ログインや自動ログインで直す案は採らない。
3. **モデルを MBP から LAN で送った。** `rsync -a --exclude .DS_Store -e 'ssh -c aes128-gcm@openssh.com'`、15 GB を約 65 MB/s で約 4 分。
   送った後にファイル一覧とサイズを MBP と突き合わせて一致 (中身のハッシュは比べていない)。転送中の M1 の Swapouts は 0。

## 3. ビルド — Swift 6.4 ではサーバが通らない

| M1 上のコマンド (commit c625d97、Swift 6.4) | 結果 |
| --- | --- |
| `swift build -c release` | **失敗** |
| `swift build -c release --product TsugumiServer` | **失敗** (rc=1) |
| `swift build -c release --product TsugumiCLI` | 成功 |

失敗は 21 か所で、どれも Swift 6.4 の region isolation のエラー `sending 'X' risks causing data races`。
場所は `Sources/TsugumiServer/Core/` の `QwenServerSession.swift`、`Qwen38ServerSession.swift`、`ServerInference.swift`
(Gemma も通る共通部分)。MBP の Swift 6.3.3 では同じソースが通る。M6 も Swift 6.4 で、`--product TsugumiCLI` だけを
ビルドしているので表に出ていなかったと思われる (M6 では確かめていない)。**ソースは直していない。**

代わりに、MBP で c625d97 を `git archive` して別の場所に展開し、`swift build -c release --product TsugumiServer`
(Swift 6.3.3、116 秒) で作ったバイナリと、横に置くリソースバンドル 4 つを M1 の `~/tsugumi-bin/` に送った。
対応 OS は macOS 15 以上 (`minos 15.0`)。外部ライブラリは Swift ランタイムだけ。Metal のシェーダは
`Tsugumi_Tsugumi.bundle` のソースから起動時にその機械向けにコンパイルされるので、M1 用の変更は要らない。
`/props` の `build_info` は `TsugumiServer-ccd455f23fb8` で、commit のハッシュとは一致しない (理由は確かめていない)。

## 4. 実行

実行前のチェック: モデルを使うプロセス無し、`memory_pressure -Q` の空き 95%、Swapouts 0、空き 117 GB。

```
cd ~/dev/tsugumi
~/tsugumi-bin/TsugumiServer --model scratch/gemma4-qat-sym.gturbo --port 8080 --ctx-size 16384
```

- 起動時のログ: `context=16384 slots=32 expert_io=mmap mtp=0 reasoning_budget=-1 reasoning_format=auto`。
  既定の 32 スロットは読み込み時のチェックで弾かれなかった。
- `/health` が 200 を返すまで約 10 秒。

リクエスト 2 本 (`/v1/chat/completions`、非ストリーミング、各 n = 1):

| リクエスト | prompt | cache | 生成 | prompt_ms | predicted_ms | predicted_per_second | 終了 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A: "Reply with exactly READY."、temperature 0、max 16 | 21 | 0 | 15 | 7747.5 | 1320.8 | 10.60 | stop |
| B: M1 チップの説明を 300 字、サンプリングはサーバ既定、max 512 | 23 | 11 | 512 | 1627.0 | 40533.3 | 12.61 | length |

- A はサーバ起動直後の最初のリクエスト。答えは `READY.`。
- B は thinking on で、思考部分は 1833 B。本文は自然な日本語だが、512 トークンで打ち切られて文の途中で終わった。

メモリ (5 秒ごとの `vm_stat`、[results/2026-09-24-first-server-run/tsugumi-vmwatch.log](results/2026-09-24-first-server-run/tsugumi-vmwatch.log)):

| | 値 |
| --- | --- |
| Swapouts | 0 のまま |
| Pageouts | 0 → 19 |
| wired の最大 | 522,367 ページ (16 KiB、約 7.97 GiB) |
| free の最小 | 3,703 ページ |
| 終了後の `memory_pressure -Q` | 空き 94% |
| サーバの RSS | 約 350 MB (重みは mmap なので RSS に出ない) |

サーバのログ: [results/2026-09-24-first-server-run/tsugumi-server.log](results/2026-09-24-first-server-run/tsugumi-server.log)。
バイナリの出所: [BUILD_INFO](results/2026-09-24-first-server-run/BUILD_INFO)。

## 5. 分かっていないこと

- 反復 3 回以上の値。上の表は 1 回ずつなので、数字に解釈を付けていない。
- M1 比の記事に使う比較。比べる機械 (M3 Pro / M6) と同じプロンプト・同じ生成の長さ・thinking の扱いを揃えた測定はしていない。設計は [01](01-THREE-MACHINE-COMPARISON-PLAN.md)。
- A の prompt_ms 7.7 s のうち、シェーダのコンパイルと SSD からの初回読みの分。
- Swift 6.4 でサーバをビルドできるようにする修正 (ソースの変更になるので未着手)。
