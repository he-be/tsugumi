#!/bin/sh
# Default vs. TF_EXPERT_MIRROR (docs/m6-prefill/44), natural prompts rotated so
# no two consecutive runs share a prompt (docs/m6-prefill/43 §3).
#   bench/m6/mirror_bench.sh <label> <mirror dir> [rounds] [share]
# Round r walks the five prompts; the arm alternates run by run and the
# starting arm flips each round, so every (prompt, arm) cell gets rounds / 2 runs.
# Fixed geometry as prefill_bench.sh. Each run records vm_stat and the
# cumulative iostat MB of every disk before and after, so the bytes each SSD
# served during the run are the difference.
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt run-N.{log,pre.txt,post.txt} summary.tsv greedy-{def,mir}.log
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; mirror=${2:?mirror dir}
rounds=${3:-6}; share=${4:-0.32}
bin="$repo/.build/release/TsugumiCLI"
model="${MODEL:-$repo/scratch/gemma4-qat-sym.gturbo}"
[ -x "$bin" ] || { echo "no $bin"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
if env | grep -q '^TF_'; then echo "unset every TF_ variable; the script sets the arm"; exit 1; fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" "$bin" > "$out/hostinfo.txt"
prompts="ja-twomachine ja-m6readme en-readme en-server code-runner"
fixed="--expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 --max-context 4096 --verification trusted-install"
{
    echo "bench/m6/mirror_bench.sh $*"
    echo "def: $bin --model $model --messages-file bench/m6/prompts/<prompt>.json $fixed"
    echo "mir: TF_EXPERT_MIRROR=$mirror TF_EXPERT_MIRROR_SHARE=$share <def>"
} > "$out/command.txt"
snap() {
    date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null | tail -1 || true
    vm_stat; echo "iostat:"; iostat -d -I -c 1
}
# iostat -I prints a header row of disk names, then KB/t xfrs MB per disk.
disk_mb() {
    awk '/^iostat:/ { getline names; getline; getline vals; n = split(names, d, " ");
         split(vals, v, " "); for (i = 1; i <= n; i++) printf "%s=%s ", d[i], v[3 * i] }' "$1"
}
printf 'run\tround\tprompt\tarm\tprefill_s\tprefill_io_s\tpageins_MiB\tswapouts\tdisk_MB\texit\n' > "$out/summary.tsv"
run=1; r=1
while [ "$r" -le "$rounds" ]; do
    k=0
    for p in $prompts; do
        if [ $(( (k + r) % 2 )) -eq 0 ]; then arm=def; else arm=mir; fi
        k=$((k + 1))
        snap > "$out/run-$run.pre.txt"
        rc=0
        if [ "$arm" = mir ]; then
            TF_EXPERT_MIRROR="$mirror" TF_EXPERT_MIRROR_SHARE="$share" \
                "$bin" --model "$model" --messages-file "$repo/bench/m6/prompts/$p.json" $fixed \
                > "$out/run-$run.log" 2>&1 || rc=$?
        else
            "$bin" --model "$model" --messages-file "$repo/bench/m6/prompts/$p.json" $fixed \
                > "$out/run-$run.log" 2>&1 || rc=$?
        fi
        snap > "$out/run-$run.post.txt"
        pre=$(disk_mb "$out/run-$run.pre.txt"); post=$(disk_mb "$out/run-$run.post.txt")
        disks=$(printf '%s\n%s\n' "$pre" "$post" | awk '
            NR == 1 { for (i = 1; i <= NF; i++) { split($i, kv, "="); a[kv[1]] = kv[2] } }
            NR == 2 { for (i = 1; i <= NF; i++) { split($i, kv, "="); printf "%s:%.0f ", kv[1], kv[2] - a[kv[1]] } }')
        awk -v run="$run" -v r="$r" -v p="$p" -v arm="$arm" -v rc="$rc" -v disks="$disks" \
            -v pre="$out/run-$run.pre.txt" -v post="$out/run-$run.post.txt" '
            function vm(file, key,   line, v) { while ((getline line < file) > 0) if (index(line, key) == 1) { v = line; sub(/.*: */, "", v); sub(/\.$/, "", v) } close(file); return v + 0 }
            /^\[load=/ { for (k = 1; k <= NF; k++) { split($k, kv, "="); sub(/^\[/, "", kv[1]); v = kv[2]; sub(/s.*$|GB.*$|\]$/, "", v); f[kv[1]] = v } }
            /^\[expert prefill/ { for (k = 1; k <= NF; k++) if ($k ~ /^io=/) { v = $k; sub(/^io=/, "", v); sub(/s$/, "", v); pio = v; break } }
            END { pin = (vm(post, "Pageins:") - vm(pre, "Pageins:")) * 16384 / 1048576
                  so = vm(post, "Swapouts:") - vm(pre, "Swapouts:")
                  printf "%s\t%s\t%s\t%s\t%s\t%s\t%.0f\t%d\t%s\t%s\n", run, r, p, arm, f["prefill"], pio, pin, so, disks, rc }
        ' "$out/run-$run.log" >> "$out/summary.tsv"
        tail -1 "$out/summary.tsv"
        run=$((run + 1))
    done
    r=$((r + 1))
done
# Same text either way: greedy 64 tokens on one prompt, each arm once.
g="--model $model --messages-file $repo/bench/m6/prompts/ja-twomachine.json --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 64 --temperature 0 --max-context 4096 --verification trusted-install"
"$bin" $g > "$out/greedy-def.log" 2>&1 || echo "greedy def exit $?"
TF_EXPERT_MIRROR="$mirror" TF_EXPERT_MIRROR_SHARE="$share" "$bin" $g > "$out/greedy-mir.log" 2>&1 || echo "greedy mir exit $?"
echo "evidence: ${out#$repo/}"
