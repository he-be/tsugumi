#!/bin/sh
# TsugumiKernelCheck at group 64 and 32, with the output filed as evidence
# (docs/m6-prefill/04-GATES.md G1 (a)). On M6 the mpp-qmm cases must run, not SKIP.
#   bench/m6/kernel_check.sh <label>
#   bench/m6/cycle.sh kernel_check.sh g1-kernel-check
# Output: bench/m6/results/<date>-<host>-<label>/{hostinfo.txt,command.txt,output.txt}
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}
cd "$repo"
swift build -c release --product TsugumiKernelCheck 2>&1 | tail -1
bin="$repo/.build/release/TsugumiKernelCheck"
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" "$bin" > "$out/hostinfo.txt"
echo "$bin --group-size 64 ; $bin --group-size 32" > "$out/command.txt"
rc=0
for g in 64 32; do
    "$bin" --group-size "$g" >> "$out/output.txt" 2>&1 || rc=$?
    echo "exit($g)=$rc" >> "$out/output.txt"
done
grep -E 'SKIP|FAIL|mpp-qmm|^PASS|exit\(' "$out/output.txt"
echo "evidence: ${out#$repo/}"
exit "$rc"
