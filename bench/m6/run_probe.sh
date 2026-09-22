#!/bin/sh
# Run one of the swift probes in bench/m6 and file its output as evidence.
#   bench/m6/run_probe.sh gpu_family_probe
#   bench/m6/run_probe.sh gpu_peak_probe 4096
# Output: bench/m6/results/<date>-<host>-<probe>/{hostinfo.txt,output.txt}
# The result directory is the evidence (docs/m6-prefill/02-EVIDENCE.md).
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
probe=$1; shift
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$probe"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" > "$out/hostinfo.txt"
echo "swift $repo/bench/m6/$probe.swift $*" > "$out/command.txt"
cd "$repo"
swift "bench/m6/$probe.swift" "$@" 2>&1 | tee "$out/output.txt"
echo "evidence: ${out#$repo/}"
