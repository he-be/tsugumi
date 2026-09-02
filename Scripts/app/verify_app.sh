#!/usr/bin/env bash
# 組み上がった Tsugumi.app を、モデルを読まずに検分する。
#
# 見るのは 3 つ。
#   1. 構造 — 2 つの実行ファイルが隣り合っているか、リソースが
#      Contents/Resources にあるか、Info.plist の版がアプリの定数と合うか
#   2. 署名 — codesign が通るか、Gatekeeper が何と言うか
#   3. リソースが実際に見えるか — TsugumiKernelCheck を .app の
#      Contents/MacOS に複製して走らせる。あの実行ファイルはアプリと同じ
#      TsugumiResources 経由で Metal のソースを読み、実行時に compile する。
#      通れば「配布した .app の中でシェーダが見つかって通った」ことになる。
#      これが初回のモデル読み込みで一番壊れやすい箇所である。
#
# 3 番目は .app の複製に対して行う。渡す本体には手を触れない。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="${1:-$REPO_ROOT/dist/Tsugumi.app}"
CONTENTS="$APP/Contents"
failures=0

fail() { echo "  NG    $1"; failures=$((failures + 1)); }
pass() { echo "  ok    $1"; }

if [[ ! -d "$APP" ]]; then
    echo "$APP が無い。先に Scripts/app/make_app.sh を走らせること" >&2
    exit 1
fi

echo "== 構造 =="
for path in \
    "MacOS/TsugumiMac" \
    "MacOS/TsugumiDecodeService" \
    "Resources/Tsugumi_Tsugumi.bundle" \
    "Resources/Tsugumi_TsugumiAppCore.bundle" \
    "Resources/Tsugumi_TsugumiMac.bundle" \
    "Resources/Tsugumi.icns" \
    "Resources/en.lproj" \
    "Resources/ja.lproj" \
    "Resources/Tsugumi_TsugumiMac.bundle/ja.lproj/Localizable.strings" \
    "Resources/Tsugumi_TsugumiAppCore.bundle/ja.lproj/Localizable.strings" \
    "Info.plist"
do
    if [[ -e "$CONTENTS/$path" ]]; then pass "$path"; else fail "$path が無い"; fi
done

# .app の直下に Contents 以外があると codesign が「封をしていない中身」として
# 弾く。Bundle.module が期待する場所がまさにそこなので、明示的に見ておく。
stray="$(ls -A "$APP" | grep -v '^Contents$' || true)"
if [[ -z "$stray" ]]; then
    pass "bundle root は Contents だけ"
else
    fail "bundle root に余計なものがある: $stray"
fi

shader_count="$(find "$CONTENTS/Resources/Tsugumi_Tsugumi.bundle" -name '*.metal' | wc -l | tr -d ' ')"
if [[ "$shader_count" -gt 0 ]]; then
    pass "Metal のソース $shader_count 本"
else
    fail "Metal のソースが 1 本も入っていない"
fi

echo "== バージョン =="
plist_version="$(defaults read "$CONTENTS/Info.plist" CFBundleShortVersionString)"
source_version="$(sed -n 's/.*fallbackShortVersion *= *"\([^"]*\)".*/\1/p' \
    "$REPO_ROOT/Sources/TsugumiApp/MacPresentation/AboutPanelPresentation.swift")"
if [[ "$plist_version" == "$source_version" ]]; then
    pass "Info.plist と AboutPanelPresentation がどちらも $plist_version"
else
    fail "Info.plist は $plist_version、ソースは $source_version"
fi

echo "== 署名 =="
if codesign --verify --deep --strict "$APP" 2>/dev/null; then
    pass "codesign --verify --deep --strict"
else
    fail "codesign --verify が通らない"
fi
authority="$(codesign -dv "$APP" 2>&1 | sed -n 's/^Authority=//p' | head -1)"
echo "  --    署名者: ${authority:-ad-hoc (署名者なし)}"
assessment="$(spctl --assess --type execute "$APP" 2>&1 || true)"
echo "  --    spctl: ${assessment:-通った}"

echo "== .app の中からリソースが見えるか =="
BIN_PATH="$(cd "$REPO_ROOT" && swift build -c release --show-bin-path)"
if [[ ! -x "$BIN_PATH/TsugumiKernelCheck" ]]; then
    fail "TsugumiKernelCheck が無い (swift build -c release を先に)"
else
    scratch="$(mktemp -d)"
    cp -R "$APP" "$scratch/Tsugumi.app"
    cp "$BIN_PATH/TsugumiKernelCheck" "$scratch/Tsugumi.app/Contents/MacOS/"
    if output="$("$scratch/Tsugumi.app/Contents/MacOS/TsugumiKernelCheck" 2>&1)"; then
        pass "Contents/MacOS の実行ファイルから Metal を読んで compile できた"
        echo "  --    $(echo "$output" | tail -1)"
    else
        fail "Contents/MacOS から走らせると落ちる"
        echo "$output" | tail -5 | sed 's/^/        /'
    fi
    rm -rf "$scratch"
fi

echo "== launchd が .app の中の decode service を起こせるか =="
# アプリは自分の隣の TsugumiDecodeService を plist にして launchctl bootstrap
# gui/$uid に渡す。ここでやっているのはその手順そのもので、署名や隔離属性の
# せいで launchd が拒む場合はここで分かる (アプリ側では「モデルの読み込みに
# 失敗」としか見えない)。ソケットが現れたら起きたということ。
label="com.tsugumi.decode.verify.$$"
socket_path="/private/tmp/tsugumi-verify-$$.sock"
plist_path="$(mktemp -d)/$label.plist"
cat > "$plist_path" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>$CONTENTS/MacOS/TsugumiDecodeService</string>
        <string>--socket</string><string>$socket_path</string>
        <string>--launch-label</string><string>$label</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><false/>
    <key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
PLIST
if launchctl bootstrap "gui/$(id -u)" "$plist_path" 2>/dev/null; then
    started=0
    for _ in $(seq 1 100); do
        if [[ -S "$socket_path" ]]; then started=1; break; fi
        sleep 0.05
    done
    if [[ "$started" -eq 1 ]]; then
        pass "launchd が起こして unix socket を張った"
    else
        fail "launchd は受け付けたが decode service が socket を張らなかった"
    fi
    launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
else
    fail "launchctl bootstrap が拒んだ"
fi
rm -rf "$(dirname "$plist_path")"
rm -f "$socket_path"

echo
if [[ "$failures" -eq 0 ]]; then
    echo "全部通った"
else
    echo "$failures 件"
    exit 1
fi
