#!/bin/sh
# TsugumiKernelCheck --router-bench (docs/m6-prefill/33), filed as evidence.
#   bench/m6/cycle.sh router_bench.sh <label>
# Output: bench/m6/results/<date>-<host>-<label>/{hostinfo.txt,command.txt,output.txt}
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}
cd "$repo"
swift build -c release --product TsugumiKernelCheck 2>&1 | tail -1
bin="$repo/.build/release/TsugumiKernelCheck"
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" "$bin" > "$out/hostinfo.txt"
echo "$bin --group-size 32 --router-bench" > "$out/command.txt"
rc=0
"$bin" --group-size 32 --router-bench > "$out/output.txt" 2>&1 || rc=$?
echo "exit=$rc" >> "$out/output.txt"
cat "$out/output.txt"
echo "evidence: ${out#$repo/}"
exit "$rc"
