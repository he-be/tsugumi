#!/bin/sh
# Three-machine comparison runs for the article (docs/m1/01-THREE-MACHINE-COMPARISON-PLAN.md).
#   bench/compare/compare_bench.sh <label> <tasks> [rounds] [-- extra TsugumiCLI args]
#   bench/compare/compare_bench.sh probe  C    1 -- --max-new 16   # §7: MTP + vision work
#   bench/compare/compare_bench.sh warmup A    1                   # §5-2
#   bench/compare/compare_bench.sh main   ABCD 3                   # §5-3, tasks interleaved per round
# Fixed: gemma4-qat-sym, --thinking on --draft-block-size 4 --max-new 1024, everything
# else default, no TF_ variable. MAX_CONTEXT=8192 adds --max-context to every task (§4).
# COOL_S (default 20) seconds between runs. IMAGE overrides the task C photo (sha256 checked).
# A swap watchdog kills the run if Swapouts grow by 1 GiB in total or 512 MiB within 60 s.
# Output: bench/compare/results/<date>-<host>-<label>-<task>/
#   hostinfo.txt command.txt env.txt [image.sha256] run-N.{log,out,pre.txt,post.txt} summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; tasks=${2:?tasks (subset of ABCD)}; shift 2
rounds=3
if [ $# -ge 1 ] && [ "$1" != "--" ]; then rounds=$1; shift; fi
[ $# -ge 1 ] && [ "$1" = "--" ] && shift
if env | grep -q '^TF_'; then echo "unset every TF_ variable first"; exit 1; fi
cool=${COOL_S:-20}
bin="$repo/.build/release/TsugumiCLI"
model="${MODEL:-$repo/scratch/gemma4-qat-sym.gturbo}"
image="${IMAGE:-$HOME/Pictures/sample_imgs/NO_FUSION_0408_001.png}"
image_sha=968bb14b8aff43631eb5cdfd8dfa12886b1014d1d5978646a1bbf3b6f1a192ec
[ -x "$bin" ] || { echo "no $bin (swift build -c release --product TsugumiCLI)"; exit 1; }
[ -e "$model" ] || { echo "no model at $model"; exit 1; }
case "$tasks" in *C*)
    [ -e "$image" ] || { echo "no image at $image"; exit 1; }
    [ "$(shasum -a 256 "$image" | cut -d' ' -f1)" = "$image_sha" ] || { echo "image sha256 differs: $image"; exit 1; } ;;
esac
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
day=$(date '+%Y-%m-%d')
page=$(vm_stat | sed -n 's/.*page size of \([0-9]*\) bytes.*/\1/p')
swapouts() { vm_stat | awk '/^Swapouts:/ { gsub(/\./, "", $2); print $2 }'; }
now() { perl -MTime::HiRes=time -e 'printf "%.3f", time'; }

prompt_of() {
    case "$1" in
        A) echo "$repo/bench/compare/prompts/chat-ja.json" ;;
        B) echo "$repo/docs/benchmark-prompts/real-generation-v1/long-synthesis.json" ;;
        C) echo "$repo/bench/compare/prompts/photo-ja.json" ;;
        D) echo "$repo/bench/compare/prompts/code-en.json" ;;
        *) echo "unknown task $1" >&2; exit 1 ;;
    esac
}

# The command of one task, one argument per line.
task_args() {
    t=$1
    printf '%s\n' "$bin" --model "$model" --messages-file "$(prompt_of "$t")"
    [ "$t" = C ] && printf '%s\n' --image "$image"
    printf '%s\n' --thinking on --draft-block-size 4 --max-new 1024
    [ -n "${MAX_CONTEXT:-}" ] && printf '%s\n' --max-context "$MAX_CONTEXT"
    return 0
}

extra_file=$(mktemp)
trap 'rm -f "$extra_file"' EXIT
for a in "$@"; do printf '%s\n' "$a"; done > "$extra_file"

outdir() { echo "$repo/bench/compare/results/$day-$host-$label-$1"; }

t_list=$(echo "$tasks" | sed 's/./& /g')
for t in $t_list; do
    out=$(outdir "$t")
    [ -e "$out" ] && { echo "$out exists; pick a new label"; exit 1; }
    mkdir -p "$out"
    sh "$repo/bench/hostinfo.sh" "$bin" > "$out/hostinfo.txt"
    { task_args "$t"; cat "$extra_file"; } | tr '\n' ' ' | sed 's/ $//' > "$out/command.txt"; echo >> "$out/command.txt"
    { env | grep '^TF_' || true; echo "COOL_S=$cool"; echo "MAX_CONTEXT=${MAX_CONTEXT:-}"; } > "$out/env.txt"
    [ "$t" = C ] && shasum -a 256 "$image" > "$out/image.sha256"
    printf 'run\tstop\tprefill_tok\tnew_tok\tdecode_s\ttok_s\tload_s\tlayerVerify_s\tprefill_s\tttft_s\tpeak_gb\tmtp_accept\tmtp_rounds\ttower_s\tsoft_tok\twall_s\tswapouts_delta\tkilled\texit\n' > "$out/summary.tsv"
done

first=1
r=1
while [ "$r" -le "$rounds" ]; do
    for t in $t_list; do
        [ "$first" = 1 ] || sleep "$cool"
        first=0
        out=$(outdir "$t")
        { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$r.pre.txt"
        s0=$(swapouts); t0=$(now)
        # Arguments one per line; the paths hold no newline.
        old_ifs=$IFS; IFS='
'
        set -- $(task_args "$t"; cat "$extra_file")
        IFS=$old_ifs
        "$@" > "$out/run-$r.out" 2> "$out/run-$r.log" &
        pid=$!
        killed=0
        (   base=$s0; win=$s0; ticks=0
            while kill -0 "$pid" 2>/dev/null; do
                sleep 5
                cur=$(swapouts); ticks=$((ticks + 1))
                if [ $(((cur - base) * page)) -ge 1073741824 ] || [ $(((cur - win) * page)) -ge 536870912 ]; then
                    echo "watchdog: Swapouts $base -> $cur, killing $pid" >> "$out/run-$r.log"
                    kill "$pid" 2>/dev/null; exit 0
                fi
                [ $((ticks % 12)) -eq 0 ] && win=$cur
            done ) &
        wpid=$!
        rc=0; wait "$pid" || rc=$?
        kill "$wpid" 2>/dev/null || true; wait "$wpid" 2>/dev/null || true
        t1=$(now); s1=$(swapouts)
        grep -q '^watchdog:' "$out/run-$r.log" && killed=1
        { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$r.post.txt"
        awk -v run="$r" -v rc="$rc" -v killed="$killed" -v wall="$(echo "$t1 - $t0" | bc)" -v dswap=$((s1 - s0)) '
            function num(v) { sub(/^\[/, "", v); sub(/\]$/, "", v); return match(v, /^[0-9.]+/) ? substr(v, 1, RLENGTH) : v }
            /^\[stop=|^\[load=|^\[mtp |^\[vision / {
                p = ($1 ~ /^\[mtp/) ? "mtp_" : ($1 ~ /^\[vision/) ? "vis_" : ($1 ~ /^\[load=/) ? "L_" : ""
                for (k = 1; k <= NF; k++) { n = index($k, "="); if (n == 0) continue
                    key = substr($k, 1, n - 1); sub(/^\[/, "", key); val = substr($k, n + 1)
                    f[p key] = (key == "stop") ? val : num(val) } }
            END { sub(/\]$/, "", f["stop"])
                  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%s\t%s\t%s\n", run,
                    f["stop"], f["prefill"], f["new"], f["decode"], f["tok/s"], f["L_load"], f["L_layerVerify"],
                    f["L_prefill"], f["L_ttft"], f["L_peak"], f["mtp_accept"], f["mtp_rounds"], f["vis_tower"], f["vis_soft"],
                    wall, dswap, killed, rc }
        ' "$out/run-$r.log" >> "$out/summary.tsv"
        echo "$t $(tail -1 "$out/summary.tsv")"
    done
    r=$((r + 1))
done
for t in $t_list; do echo "evidence: $(outdir "$t" | sed "s|^$repo/||")"; done
