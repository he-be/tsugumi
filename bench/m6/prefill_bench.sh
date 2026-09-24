#!/bin/sh
# Prefill benchmark with the evidence filed automatically.
#   bench/m6/prefill_bench.sh <label> [n] [-- extra TsugumiCLI args]
#   TF_PREFILL_GPU_PROFILE=1 bench/m6/prefill_bench.sh g0-baseline 6
# Fixed geometry (docs/m6-prefill/04-GATES.md G0-3): gemma4-qat-sym, bench/l.json
# (2478 prompt tokens), 32 slots, chunk 2048, max-new 1, max-context 4096,
# trusted-install (sizes against verified-install.json, no SHA-256), so prefill
# carries no integrity check and the page cache is left as the run finds it
# (docs/m6-prefill/35). Change nothing here without a new label; pp is 2478 / prefill s.
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt env.txt run-N.log run-N.pre.txt run-N.post.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; shift
n=6
if [ $# -ge 1 ] && [ "$1" != "--" ]; then n=$1; shift; fi
[ $# -ge 1 ] && [ "$1" = "--" ] && shift
# A ledger-* label is the production default and nothing else (docs/m6-prefill/06-LEDGER.md):
# no TF_ variable, no extra argument. Other labels are candidates and may carry both.
case "$label" in
    ledger*)
        if env | grep -q '^TF_' || [ $# -ge 1 ]; then
            echo "label '$label' is a ledger row: unset every TF_ variable and pass no extra args"; exit 1
        fi ;;
esac
tokens=${PROMPT_TOKENS:-2478}
bin="$repo/.build/release/TsugumiCLI"
model="${MODEL:-$repo/scratch/gemma4-qat-sym.gturbo}"
[ -x "$bin" ] || { echo "no $bin (swift build -c release --product TsugumiCLI)"; exit 1; }
[ -e "$model" ] || { echo "no model at $model"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" "$bin" > "$out/hostinfo.txt"
# TF_BENCH_MESSAGES swaps the prompt for a measurement that is not a ledger row
# (the ledger guard refuses any TF_ variable).
messages="${TF_BENCH_MESSAGES:-$repo/bench/l.json}"
set -- "$bin" --model "$model" --messages-file "$messages" \
    --expert-cache-slots 32 --prefill-chunk-tokens 2048 --max-new 1 \
    --max-context 4096 --verification trusted-install "$@"
echo "$*" > "$out/command.txt"
{ env | grep '^TF_' || true; echo "PROMPT_TOKENS=$tokens"; } > "$out/env.txt"
printf 'run\tprefill_s\tpp_tok_s\tprefill_io_s\tlayerVerify_s\tload_s\tttft_s\tpeak_gb\texit\n' > "$out/summary.tsv"
i=1
while [ "$i" -le "$n" ]; do
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$i.pre.txt"
    rc=0
    "$@" > "$out/run-$i.log" 2>&1 || rc=$?
    # After the run too, so Swapins/Swapouts during it are the difference of the two.
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$i.post.txt"
    awk -v run="$i" -v tokens="$tokens" -v rc="$rc" '
        /^\[load=/ { for (k = 1; k <= NF; k++) { split($k, kv, "="); sub(/^\[/, "", kv[1]); v = kv[2]; sub(/s.*$|GB.*$|\]$/, "", v); f[kv[1]] = v } }
        /^\[expert prefill/ { for (k = 1; k <= NF; k++) if ($k ~ /^io=/) { v = $k; sub(/^io=/, "", v); sub(/s$/, "", v); pio = v; break } }
        END { pp = (f["prefill"] > 0) ? tokens / f["prefill"] : 0;
              printf "%s\t%s\t%.1f\t%s\t%s\t%s\t%s\t%s\t%s\n", run, f["prefill"], pp, pio, f["layerVerify"], f["load"], f["ttft"], f["peak"], rc }
    ' "$out/run-$i.log" >> "$out/summary.tsv"
    tail -1 "$out/summary.tsv"
    i=$((i + 1))
done
echo "evidence: ${out#$repo/}"
cat "$out/summary.tsv"
