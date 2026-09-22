#!/bin/sh
# One-line host stamp for benchmark results (docs/TWO_MACHINE_DEV.md §4).
#
#   bench/hostinfo.sh                       # 機械とツールチェーンだけ
#   bench/hostinfo.sh .build/release/TsugumiCLI   # 測ったバイナリの minos も付ける
#
# 2 機並走 (MBP macOS 15 / M6 macOS 27) では、どちらで取った数字かを
# 後から復元できないと比較が死ぬ。結果ファイルの先頭にこの 1 行を貼ること。
set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)

model=$(sysctl -n hw.model)
chip=$(sysctl -n machdep.cpu.brand_string)
ram=$(( $(sysctl -n hw.memsize) / 1073741824 ))
os=$(sw_vers -productVersion)
build=$(sw_vers -buildVersion)
swiftv=$(swift --version 2>/dev/null | sed -n 's/.*Apple Swift version \([0-9.]*\).*/\1/p' | head -1)
sdk=$(xcrun --sdk macosx --show-sdk-version 2>/dev/null || echo "?")
devdir=$(xcode-select -p 2>/dev/null)
case "$devdir" in
    *CommandLineTools*) toolchain="CLT" ;;
    *Xcode*) toolchain="Xcode $(xcodebuild -version 2>/dev/null | sed -n '1s/Xcode //p')" ;;
    *) toolchain="$devdir" ;;
esac
ssd=$(system_profiler SPNVMeDataType 2>/dev/null | sed -n 's/.*Model: *\(APPLE SSD [A-Z0-9]*\).*/\1/p' | head -1)
commit=$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo "?")
if ! git -C "$repo" diff --quiet HEAD 2>/dev/null; then
    commit="$commit+dirty"
fi

stamp="測定: $(date '+%Y-%m-%d') / ${model} ${chip} ${ram}GB / macOS ${os} (${build})"
stamp="${stamp} / Swift ${swiftv} / SDK ${sdk} / ${toolchain} / ${ssd} / commit ${commit}"

if [ $# -ge 1 ]; then
    minos=$(otool -l "$1" 2>/dev/null | awk '/LC_BUILD_VERSION/,/sdk/ { if ($1 == "minos") print $2 }' | head -1)
    [ -n "$minos" ] && stamp="${stamp} / minos ${minos}"
fi

echo "$stamp"
