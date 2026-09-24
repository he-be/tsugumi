#!/bin/sh
# Internal SSD and an external one read at the same time: does the sum grow?
#   bench/m6/dual_ssd_probe.sh <label> <internal dir> <external dir> [seconds] [rounds]
# Each dir holds layer_*.bin that no process has read since boot (bench/gen_dataset.c
# writes them with F_NOCACHE). rec_probe reads 3,358,720-byte records (one expert,
# expertStride of the shipped pack) at random 16K-aligned offsets with F_NOCACHE.
# Per round and QD: internal alone, external alone, both at once (two processes).
# Output: bench/m6/results/<date>-<host>-<label>/{hostinfo.txt,command.txt,output.tsv}
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; int=${2:?internal dir}; ext=${3:?external dir}
secs=${4:-15}; rounds=${5:-2}
rec=3358720
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" > "$out/hostinfo.txt"
echo "bench/m6/dual_ssd_probe.sh $*" > "$out/command.txt"
probe=$(mktemp -d)/rec_probe
clang -O2 -o "$probe" "$repo/bench/rec_probe.c" -lpthread
{ df -h "$int" "$ext"; ls -l "$int" "$ext"; } > "$out/datasets.txt"
# rec_probe columns: MB qd GB/s count IOPS p10 p50 mean p99 p999 (ms)
printf 'round\tqd\tarm\tdisk\tMB\tqd2\tGBps\tcount\tIOPS\tp10\tp50\tmean\tp99\tp999\n' > "$out/output.tsv"
r=1
while [ "$r" -le "$rounds" ]; do
    for qd in 4 8; do
        line=$("$probe" "$int" $rec $qd "$secs")
        printf '%s\t%s\talone\tinternal\t%s\n' "$r" "$qd" "$line" | tee -a "$out/output.tsv"
        line=$("$probe" "$ext" $rec $qd "$secs")
        printf '%s\t%s\talone\texternal\t%s\n' "$r" "$qd" "$line" | tee -a "$out/output.tsv"
        "$probe" "$int" $rec $qd "$secs" > "$out/.int" &
        pi=$!
        "$probe" "$ext" $rec $qd "$secs" > "$out/.ext" &
        pe=$!
        wait $pi $pe
        printf '%s\t%s\tboth\tinternal\t%s\n' "$r" "$qd" "$(cat "$out/.int")" | tee -a "$out/output.tsv"
        printf '%s\t%s\tboth\texternal\t%s\n' "$r" "$qd" "$(cat "$out/.ext")" | tee -a "$out/output.tsv"
        rm -f "$out/.int" "$out/.ext"
    done
    r=$((r + 1))
done
echo "evidence: ${out#$repo/}"
