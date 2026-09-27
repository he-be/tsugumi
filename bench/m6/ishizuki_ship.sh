#!/bin/sh
# Build the ishizuki prefill driver on the MBP and ship it to the M6
# (docs/qwen38-27b/15-M6-PREFILL-PLAN.md §3-2, route B). ishizuki cannot be built on the
# M6: mlx-swift compiles its Metal kernels at build time with `metal`, which the CLT lacks.
# The MBP has Xcode 26.6 (SDK 26.5); ishizuki's platforms are .macOS(.v26), so the binary
# carries minos 26 and runs on the M6's macOS 27 but not on the MBP's macOS 15.
#   bench/m6/ishizuki_ship.sh [m6 host]     # default: m6-lan if reachable, else m6
# What it does, each step idempotent:
#   1. clone struffl/ishizuki into scratch/ishizuki (pinned commit below) and lower the
#      manifest's tools-version 6.4 -> 6.3 (the MBP has Swift 6.3.3; nothing else changes)
#   2. add an executable target IshizukiPrefill whose one source is
#      bench/m6/ishizuki-driver/main.swift (copied in, the repo copy stays the original)
#   3. swift build -c release --product ishizuki-prefill
#   4. rsync the executable and every resource bundle (mlx-swift's metallib lives in
#      mlx-swift_Cmlx.bundle) to <host>:~/LLM/ref/ishizuki-bin/, with build-info.txt
#      (this machine's hostinfo line, the ishizuki commit, the binary's minos)
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
src="$repo/scratch/ishizuki"
pin=${ISHIZUKI_COMMIT:-459ee00}
host=${1:-}
if [ -z "$host" ]; then
    if ssh -o ConnectTimeout=3 m6-lan true 2>/dev/null; then host=m6-lan; else host=m6; fi
fi
if [ ! -d "$src/.git" ]; then
    git clone -q https://github.com/struffl/ishizuki "$src"
fi
git -C "$src" checkout -q "$pin" 2>/dev/null || { git -C "$src" fetch -q --depth 50 origin; git -C "$src" checkout -q "$pin"; }
sed -i '' '1s/swift-tools-version: 6\.4/swift-tools-version: 6.3/' "$src/Package.swift"
mkdir -p "$src/Sources/IshizukiPrefill"
cp "$repo/bench/m6/ishizuki-driver/main.swift" "$src/Sources/IshizukiPrefill/main.swift"
if ! grep -q 'IshizukiPrefill' "$src/Package.swift"; then
    # Append the product and the target inside the existing arrays (both end with "],").
    python3 - "$src/Package.swift" <<'EOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = s.replace('.library(name: "IshizukiLink", targets: ["IshizukiLink"]),',
              '.library(name: "IshizukiLink", targets: ["IshizukiLink"]),\n    .executable(name: "ishizuki-prefill", targets: ["IshizukiPrefill"]),', 1)
s = s.replace('    .testTarget(\n      name: "IshizukiLinkTests",',
              '    .executableTarget(\n      name: "IshizukiPrefill",\n      dependencies: ["IshizukiKit", .product(name: "MLX", package: "mlx-swift")]\n    ),\n    .testTarget(\n      name: "IshizukiLinkTests",', 1)
open(p, "w").write(s)
EOF
fi
grep -q 'name: "IshizukiPrefill"' "$src/Package.swift" || { echo "Package.swift patch did not apply"; exit 1; }
# Agent/IshizukiLanguageModel.swift adopts Foundation Models' custom LanguageModel API,
# which exists in SDK 27.0 (the M6's CLT) and not in the MBP's SDK 26.5. Nothing else in
# the package references it (grep -rl IshizukiLanguageModel Sources App is empty), so it
# is left out of the IshizukiKit target here. The driver never touches the agent layer.
if ! grep -q 'Agent/IshizukiLanguageModel.swift' "$src/Package.swift"; then
    python3 - "$src/Package.swift" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
# PackageDescription takes the labels in declaration order: dependencies, then exclude.
s = s.replace('        .product(name: "OrderedCollections", package: "swift-collections"),\n      ]\n    ),',
              '        .product(name: "OrderedCollections", package: "swift-collections"),\n      ],\n      exclude: ["Agent/IshizukiLanguageModel.swift"]\n    ),', 1)
open(p, "w").write(s)
EOF
fi
grep -q 'Agent/IshizukiLanguageModel.swift' "$src/Package.swift" || { echo "Package.swift exclude patch did not apply"; exit 1; }
# Swift Build, not the classic build system: only Swift Build compiles the .metal sources
# of mlx-swift's Cmlx target into mlx-swift_Cmlx.bundle/default.metallib, which MLX loads
# at start-up (device.cpp: load_default_library). The classic `swift build` links fine and
# then fails on the M6 with "Failed to load the default metallib". The metal compiler is
# Xcode's Metal Toolchain component (`xcodebuild -downloadComponent MetalToolchain`).
# arm64 only: mlx-swift has no Float16 for x86_64, so a universal build fails.
xcrun -find metal >/dev/null 2>&1 || { echo "no metal compiler: xcodebuild -downloadComponent MetalToolchain"; exit 1; }
(cd "$src" && swift build --build-system swiftbuild --arch arm64 -c release --product ishizuki-prefill 2>&1 | tail -3)
products="$src/.build/out/Products/Release"
bin="$products/ishizuki-prefill"
[ -x "$bin" ] || { echo "no $bin"; exit 1; }
[ -f "$products/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ] || [ -f "$products/mlx-swift_Cmlx.bundle/default.metallib" ] || { echo "no default.metallib in $products/mlx-swift_Cmlx.bundle"; exit 1; }
out="$src/.build/ship"
rm -rf "$out"; mkdir -p "$out"
cp "$bin" "$out/"
for b in "$products"/*.bundle; do [ -d "$b" ] && cp -R "$b" "$out/"; done
{
    sh "$repo/bench/hostinfo.sh" "$bin"
    echo "ishizuki $(git -C "$src" rev-parse --short HEAD) (tools-version 6.4 -> 6.3, + IshizukiPrefill target) / driver bench/m6/ishizuki-driver/main.swift @ $(git -C "$repo" rev-parse --short HEAD)"
    echo "bundles: $(ls "$out" | grep '\.bundle$' | tr '\n' ' ')"
    ls -l "$out/ishizuki-prefill" | awk '{print "size " $5}'
} > "$out/build-info.txt"
cat "$out/build-info.txt"
ssh "$host" 'mkdir -p ~/LLM/ref/ishizuki-bin'
rsync -a --delete -e "ssh -c aes128-gcm@openssh.com" "$out/" "$host:LLM/ref/ishizuki-bin/"
echo "shipped to $host:~/LLM/ref/ishizuki-bin/"
