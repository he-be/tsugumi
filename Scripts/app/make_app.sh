#!/usr/bin/env bash
# 配れる Tsugumi.app を組み立てる。
#
# 出来上がりの中身:
#   Tsugumi.app/Contents/MacOS/TsugumiMac            … アプリ本体
#   Tsugumi.app/Contents/MacOS/TsugumiDecodeService  … モデルを持つ別プロセス
#   Tsugumi.app/Contents/Resources/*.bundle          … Metal / Templates / prompts / アイコン
#
# 2 つの実行ファイルが隣り合っていることが要件である。アプリは
# DecodeServiceInferenceClient.defaultServiceURL() で「自分の隣の
# TsugumiDecodeService」を探し、それを launchd に渡して起こす。
#
# リソースが Contents/Resources に入っているのも要件である。SwiftPM が作る
# Bundle.module は .app の直下を見にいくが、そこに署名されないファイルを置くと
# codesign が弾く。PackagedResourceBundle が Contents/Resources を先に見て
# いるので、この配置で両方のプロセスから見える (Contents/MacOS の実行ファイルは
# 自分のディレクトリではなく、囲っている .app を main bundle として受け取る)。
#
# 署名は既定で ad-hoc。手元と、貸してもらった Mac で動かすにはこれで足りる
# (受け取り側で隔離属性を外す必要はある。docs/DISTRIBUTION.md を見ること)。
# 配布用に Developer ID で署名するときは環境変数で識別名を渡す:
#
#   TSUGUMI_CODESIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" \
#     Scripts/app/make_app.sh --zip
#
# そのときだけ hardened runtime と Scripts/app/Tsugumi.entitlements が付く。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIRECTORY="$REPO_ROOT/dist"
BUNDLE_IDENTIFIER="io.github.he-be.tsugumi"
MAKE_ZIP=0
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_DIRECTORY="$2"; shift 2 ;;
        --zip) MAKE_ZIP=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        -h|--help)
            sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$REPO_ROOT"

# バージョンはアプリが表示する定数から取る。Info.plist を手で書くと
# About パネルの表示と食い違うので、出所を 1 つにしておく。
VERSION_SOURCE="Sources/TsugumiApp/MacPresentation/AboutPanelPresentation.swift"
VERSION="$(sed -n 's/.*fallbackShortVersion *= *"\([^"]*\)".*/\1/p' "$VERSION_SOURCE")"
if [[ -z "$VERSION" ]]; then
    echo "fallbackShortVersion を $VERSION_SOURCE から読めなかった" >&2
    exit 1
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    echo "== build (release) =="
    swift build -c release --product TsugumiMac
    swift build -c release --product TsugumiDecodeService
fi

BIN_PATH="$(swift build -c release --show-bin-path)"
APP="$OUTPUT_DIRECTORY/Tsugumi.app"
CONTENTS="$APP/Contents"

echo "== assemble $APP (version $VERSION) =="
rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

for executable in TsugumiMac TsugumiDecodeService; do
    if [[ ! -x "$BIN_PATH/$executable" ]]; then
        echo "$BIN_PATH/$executable が無い (--skip-build を外して実行すること)" >&2
        exit 1
    fi
    cp "$BIN_PATH/$executable" "$CONTENTS/MacOS/$executable"
done

# 名前は <パッケージ名>_<ターゲット名>.bundle。Package.swift でどちらかを
# 変えると PackagedResourceBundle の引数と揃わなくなるので、ここで名指しして
# 落とす — 揃っていないと初回のモデル読み込みで missingShaderResource になる。
RESOURCE_BUNDLES=(
    Tsugumi_Tsugumi            # Metal/ と Templates/ (シェーダは実行時に compile する)
    Tsugumi_TsugumiAppCore     # app-prompts.json
    Tsugumi_TsugumiMac         # アプリアイコン
)
for bundle in "${RESOURCE_BUNDLES[@]}"; do
    if [[ ! -d "$BIN_PATH/$bundle.bundle" ]]; then
        echo "$BIN_PATH/$bundle.bundle が無い" >&2
        exit 1
    fi
    cp -R "$BIN_PATH/$bundle.bundle" "$CONTENTS/Resources/$bundle.bundle"
done

echo "== icon =="
ICON_SOURCE="$REPO_ROOT/Sources/TsugumiApp/Mac/Resources/tsugumi-app-icon.png"
ICONSET="$(mktemp -d)/Tsugumi.iconset"
mkdir -p "$ICONSET"
for size in 16 32 64 128 256 512; do
    sips -z "$size" "$size" "$ICON_SOURCE" \
        --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -z "$((size * 2))" "$((size * 2))" "$ICON_SOURCE" \
        --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
# 1024 の @2x は 2048 になってしまうので、512@2x までで打ち止めにしてある。
iconutil --convert icns "$ICONSET" --output "$CONTENTS/Resources/Tsugumi.icns"
rm -rf "$(dirname "$ICONSET")"

cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Tsugumi</string>
    <key>CFBundleDisplayName</key><string>Tsugumi</string>
    <key>CFBundleExecutable</key><string>TsugumiMac</string>
    <key>CFBundleIdentifier</key><string>$BUNDLE_IDENTIFIER</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>CFBundleIconFile</key><string>Tsugumi</string>
    <key>LSMinimumSystemVersion</key><string>15.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSHumanReadableCopyright</key><string>Apache License 2.0</string>
    <key>NSSupportsAutomaticTermination</key><false/>
    <key>NSSupportsSuddenTermination</key><false/>
</dict>
</plist>
PLIST

echo "== sign =="
IDENTITY="${TSUGUMI_CODESIGN_IDENTITY:--}"
SIGN_OPTIONS=(--force --sign "$IDENTITY" --timestamp=none)
if [[ "$IDENTITY" != "-" ]]; then
    # Developer ID のときだけ hardened runtime。Metal のシェーダは実行時に
    # compile して読み込むので、library validation を切っていないと
    # モデル読み込みで落ちる (entitlements のコメントを見ること)。
    SIGN_OPTIONS=(--force --sign "$IDENTITY" --timestamp --options runtime
                  --entitlements "$REPO_ROOT/Scripts/app/Tsugumi.entitlements")
fi
# 内側から署名する。順番を逆にすると外側の署名が中身の変更で壊れる。
codesign "${SIGN_OPTIONS[@]}" "$CONTENTS/MacOS/TsugumiDecodeService"
codesign "${SIGN_OPTIONS[@]}" "$APP"
codesign --verify --strict --verbose=2 "$APP"

if [[ "$MAKE_ZIP" -eq 1 ]]; then
    ARCHIVE="$OUTPUT_DIRECTORY/Tsugumi-$VERSION.zip"
    echo "== zip $ARCHIVE =="
    rm -f "$ARCHIVE"
    # ditto でないと署名に必要な属性が落ちる。zip(1) は使わないこと。
    ditto -c -k --keepParent "$APP" "$ARCHIVE"
fi

echo
echo "$APP"
du -sh "$APP" | awk '{print "  " $1}'
if [[ "$IDENTITY" == "-" ]]; then
    echo "  ad-hoc 署名。別の Mac では Gatekeeper に止められる —"
    echo "  受け取った側で: xattr -dr com.apple.quarantine /Applications/Tsugumi.app"
fi
