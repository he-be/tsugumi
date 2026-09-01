# 配布 — .app の組み立てと、別の Mac での検分

最終更新: 2026-09-01。`swift build` で出る裸の実行ファイルを、**Xcode も
チェックアウトも無い Mac に持っていける `Tsugumi.app`** にするまでの手順と、
持っていった先で何を見るか。

このリポジトリには `.xcodeproj` が無い。バンドルはスクリプトが組む。

## 1. 組み立てる

```bash
Scripts/app/make_app.sh --zip
Scripts/app/verify_app.sh
```

`dist/Tsugumi.app` (47 MB) と `dist/Tsugumi-<version>.zip` ができる。
中身は 3 つだけ:

| 場所 | 中身 | なぜそこか |
| --- | --- | --- |
| `Contents/MacOS/TsugumiMac` | アプリ本体 | `CFBundleExecutable` |
| `Contents/MacOS/TsugumiDecodeService` | モデルを持つ別プロセス | アプリが**自分の隣**を探して launchd に渡す (`DecodeServiceInferenceClient.defaultServiceURL`) |
| `Contents/Resources/*.bundle` | Metal のソース・chat template・prompt preset・アイコン | 下記 |

バージョンは `AboutPanelPresentation.fallbackShortVersion` から取って
`Info.plist` に書く。About パネルの表示と `Info.plist` は同じ出所になる。

アイコンは `Sources/TsugumiApp/Mac/Resources/tsugumi-app-icon.png` の 1 枚から、
Finder/Dock 用の `.icns` も、アプリが実行時に読む画像 (`MacAppIcon`) も作る。
中身は `docs/assets/tsugumi.png` を macOS の格子 — 1024 の canvas の中央に
824 角の角丸正方形 — に収めたもので、ロゴを描き直したときは
`swift Scripts/app/round_app_icon.swift docs/assets/tsugumi.png <出力>` を
通してから置き換えること。全面を絵で埋めると角の立った四角いタイルになる。

### リソースの置き場所が Contents/Resources である理由

SwiftPM が作る `Bundle.module` は `<パッケージ名>_<ターゲット名>.bundle` を
**`Bundle.main.bundleURL` の隣**に探す。裸の実行ファイルではそれが実行ファイルの
ディレクトリで正しいが、`.app` の中ではバンドルの直下になる。そこに署名されない
ファイルを置くと codesign が「封をしていない中身」として弾くので、その配置は
取れない。

代わりに `PackagedResourceBundle` (`Sources/TsugumiBundleLocation`) が
`Bundle.main.resourceURL` = `Contents/Resources` を先に見る。`Contents/MacOS`
に置いた実行ファイルは、自分のディレクトリではなく**囲っている `.app`** を
main bundle として受け取るので、アプリからも decode service からも同じ場所が
見える (実測。`verify_app.sh` の 3 番目の項目がそれを毎回確かめている)。

Metal のシェーダは実行時に `.metal` のソースから compile する。ここが外れると
モデルの読み込みが `missingShaderResource` で落ちる — バンドル化で一番壊れやすい
のはこの 1 点なので、`verify_app.sh` は `TsugumiKernelCheck` を `.app` の複製に
入れて走らせ、実際に compile が通るところまで見ている。

## 2. 署名

既定は ad-hoc 署名。手元と、借りてきた Mac で動かすにはこれで足りる。

Developer ID を持っているなら:

```bash
TSUGUMI_CODESIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" \
  Scripts/app/make_app.sh --zip
```

このときだけ hardened runtime と `Scripts/app/Tsugumi.entitlements` が付く。
entitlements には `allow-jit` と `disable-library-validation` を入れてある —
実行時に Metal のライブラリを作って読むためで、**実機で外して通るなら外すこと**。

App Sandbox は入れていない。launchd に plist を渡して decode service を起こす
経路と、`~/Library/Application Support/Tsugumi` への直書きがあるので、
サンドボックスの中では通らない。したがって Mac App Store ではなく
**Developer ID + zip/DMG** が配布経路になる。

## 3. 借りた Mac に持っていく

`dist/Tsugumi-<version>.zip` を AirDrop なりで送る (zip は `ditto` で作ってある。
`zip(1)` では署名に必要な属性が落ちる)。受け取った側で:

```bash
unzip Tsugumi-0.4.3.zip -d /Applications
xattr -dr com.apple.quarantine /Applications/Tsugumi.app   # ad-hoc 署名のとき必要
open /Applications/Tsugumi.app
```

隔離属性を外さないと Gatekeeper が止める (`spctl --assess` は ad-hoc では必ず
`rejected` になる)。公証まで通せばこの 1 行は要らなくなる。

### 持っていった先の前提

- Apple Silicon / macOS 15 以降
- **ディスク**: Gemma 4 が 15.7 GB、Ornith が 21.0 GB。両方入れるなら 37 GB
  空けておくこと。インストーラは 1 GB を予備として要求する
- モデルの置き場所は `~/Library/Application Support/Tsugumi/`。開発
  チェックアウトの `scratch/` を見にいくのは、`Package.swift` が親を辿って
  見つかるとき — つまり配布した `.app` では起きない
- 会話は同じディレクトリの `chats.json`
- ネットワーク。ウェイトは Hugging Face の 2 リポジトリから直接落ちる
  (どちらも public。トークンは要らない)

### 16GB 機での既定

既定は 32K コンテキスト / エキスパートキャッシュ 32 スロットで、これが運用点。
128K を選ぶと wired limit を上げない限り decode が半減しうる
([SERVER_RUNBOOK](SERVER_RUNBOOK.md))。16GB での測定はまだ無いので、まず既定の
まま回すこと。

## 4. 実機で見る

`verify_app.sh` が手元で見ているのは構造・署名・リソース解決・launchd の 4 つで、
**モデルは読んでいない**。借りた機械で見るのはその先:

1. 未インストールからの通し — 起動 → Download → SHA-256 の検証 → Load → 生成。
   `chats.json` が新規に作られること
2. Ornith の MTP が実際に立っているか。パック内の `mtp-head/` が見つからないと
   **黙って OFF に落ちる** (`RealInferenceClient.mtpSidecarDirectory` は開発機の
   `~/LLM/ornith-mtp-head` を fallback にしている)。生成後の `draft` の数字が
   `0/0` なら立っていない
3. 16GB でのメモリの振る舞い。README が主張しているのは「要る RAM 1.3 GB /
   借りる RAM 最大 13.28 GiB」で、これを 16GB 実機で確かめたことはまだ無い
4. macOS 15 なら prefill が遅いこと (Metal 4 のテンソルカーネルは macOS 26 のみ)

decode service だけを叩きたいときは、GUI を通さずに:

```bash
TSUGUMI_DECODE_SERVICE=/Applications/Tsugumi.app/Contents/MacOS/TsugumiDecodeService \
TSUGUMI_MODEL_DIR="$HOME/Library/Application Support/Tsugumi" \
  python3 Scripts/app/smoke_decode.py gemma
```

## 5. まだ無いもの

- **Developer ID 署名と公証** (`notarytool` + `stapler`)。証明書が要る
- **DMG**。今は zip だけ
- **リリースのワークフロー**。CI は手動起動 (`workflow_dispatch`) だけに
  してあるので、タグ → バンドル → 署名 → 公証 → Release の自動化は入れていない。
  `Scripts/check_app_version.rb` は公開済みリリースがある前提で動くため、
  最初のリリースを切るときに版の運用 (今 0.4.3) も同時に決めることになる
- **16GB での実測**

## 付録: 気づいた落とし穴

- `swift-transformers` の `Hub` は、tokenizer config に `tokenizer_class` が
  無いときだけ `Bundle.module` から fallback config を読む。その `Bundle.module`
  は開発機の絶対パスを最後の候補にしていて、外した Mac では `fatalError` になる。
  配布している 2 つのパックの `tokenizer_config.json` はどちらも
  `tokenizer_class` を持っている (`GemmaTokenizer` / `Qwen2Tokenizer`) ので
  この経路には入らない。**パックのトークナイザを差し替えるときはここを確かめる**
- `.app` の直下に `Contents` 以外を置かないこと。codesign が弾く。
  `verify_app.sh` が毎回見ている
