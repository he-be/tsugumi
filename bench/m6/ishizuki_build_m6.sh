#!/bin/sh
# Build ishizuki's headless tools on the M6 itself, from the he-be/ishizuki fork
# (docs/qwen38-27b/16 §17). The M6 has Xcode 27.0 since 2026-09-28; before that it had only the
# CLT, which cannot compile mlx-swift's Metal kernels, and ishizuki_ship.sh built on the MBP.
#   bench/m6/ishizuki_build_m6.sh [m6 host]     # default: m6-lan if reachable, else m6
# Environment:
#   ISHIZUKI_FORK_COMMIT  fork commit to build (default below; the fork holds api / bind-host /
#                         mtp as commits and the four executables as targets)
#   ISHIZUKI_BIN_DIR      ~/LLM/ref/<dir> to install into (default ishizuki-bin-m6); pick a new
#                         one to build beside a running server
# What it does on the M6:
#   1. clone https://github.com/he-be/ishizuki into ~/LLM/ref/ishizuki-src (once), fetch,
#      check out the commit (Package.resolved's local rewrite by SwiftPM is discarded first)
#   2. with DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer (xcode-select still points
#      at the CLT and switching it needs sudo), swift build --build-system swiftbuild -c release
#      for ishizuki-serve, ishizuki-prefill, ishizuki-verify-bench, ishizuki-decode-probe.
#      Needs Xcode's Metal Toolchain component (`xcodebuild -downloadComponent MetalToolchain`,
#      839 MB, installed 2026-09-28): Swift Build compiles mlx-swift's .metal sources into
#      mlx-swift_Cmlx.bundle/default.metallib, which MLX loads at start-up
#   3. copy the executables and the bundles to ~/LLM/ref/$ISHIZUKI_BIN_DIR with build-info.txt
# The server still reads the pack through ~/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit, a link into
# /Volumes/SSD256: after a reboot, `diskutil mount disk5s1` first.
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
commit=${ISHIZUKI_FORK_COMMIT:-98e22a4}
dest=${ISHIZUKI_BIN_DIR:-ishizuki-bin-m6}
host=${1:-}
if [ -z "$host" ]; then
    if ssh -o ConnectTimeout=3 m6-lan true 2>/dev/null; then host=m6-lan; else host=m6; fi
fi
ssh "$host" COMMIT="$commit" DEST="$dest" TF="$(git -C "$repo" rev-parse --short HEAD)" sh -s <<'EOF'
set -eu
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun -find metal >/dev/null
xcodebuild -showComponent MetalToolchain 2>/dev/null | grep -q 'Status: installed' || {
    echo "Metal Toolchain not installed: xcodebuild -downloadComponent MetalToolchain"; exit 1; }
src=~/LLM/ref/ishizuki-src
[ -d "$src/.git" ] || git clone -q https://github.com/he-be/ishizuki "$src"
cd "$src"
git remote set-url origin https://github.com/he-be/ishizuki
git checkout -q -- Package.resolved 2>/dev/null || true
git fetch -q origin
git checkout -q "$COMMIT"
[ -z "$(git status --porcelain)" ] || { echo "ishizuki-src is not clean:"; git status --short; exit 1; }
for product in ishizuki-serve ishizuki-prefill ishizuki-verify-bench ishizuki-decode-probe; do
    swift build --build-system swiftbuild -c release --product "$product" 2>&1 | grep -E 'error:|Build complete' || true
done
products=.build/out/Products/Release
[ -f "$products/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ] || { echo "no default.metallib"; exit 1; }
out=~/LLM/ref/$DEST
rm -rf "$out"; mkdir -p "$out"
cp "$products/ishizuki-serve" "$products/ishizuki-prefill" "$products/ishizuki-verify-bench" "$products/ishizuki-decode-probe" "$out/"
cp -R "$products"/*.bundle "$out/"
{
    echo "built on $(hostname -s) $(date '+%Y-%m-%d %H:%M') / $(sysctl -n machdep.cpu.brand_string) / macOS $(sw_vers -productVersion) / $(xcodebuild -version | tr '\n' ' ')/ $(swift --version 2>&1 | head -1)"
    echo "he-be/ishizuki $(git rev-parse --short HEAD) / turbo-fieldfare $TF"
    otool -l "$out/ishizuki-serve" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print "minos " $2; exit}'
} > "$out/build-info.txt"
cat "$out/build-info.txt"
echo "installed to $out"
EOF
