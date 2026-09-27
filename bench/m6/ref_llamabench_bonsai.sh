#!/bin/sh
# External reference on the same machine: the prism fork's llama-bench on Ternary
# Bonsai 2 27B (PQ2_0), prompt processing at fixed synthetic lengths (pp512, pp2048),
# filed as evidence (docs/qwen38-27b/15-M6-PREFILL-PLAN.md). This is the form other
# projects publish (ishizuki's engine table, prism's README), so it is what a number
# from outside can be set beside; it is not the natural-text prompt of
# ref_llamacpp_bonsai.sh, and llama-bench's prompt is synthetic tokens, not text.
#   bench/m6/ref_llamabench_bonsai.sh <label> [reps]
#   GGML_METAL_TENSOR_DISABLE=1 bench/m6/ref_llamabench_bonsai.sh bonsai-llamabench-tensor-off 3
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt env.txt run-1.log run-1.pre.txt run-1.post.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; shift
reps=${1:-5}
llama=${LLAMA_DIR:-$HOME/LLM/ref/prism-b10743}
gguf=${GGUF:-$HOME/LLM/ref/Ternary-Bonsai-2-27B-PQ2_0.gguf}
lengths=${PP:-512,2048}
ubatch=${UBATCH:-2048}
bin="$llama/llama-bench"
[ -x "$bin" ] || bin="$llama/build/bin/llama-bench"
[ -x "$bin" ] || { echo "no llama-bench under $llama (or its build/bin)"; exit 1; }
[ -f "$gguf" ] || { echo "no GGUF at $gguf"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|llama-' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" > "$out/hostinfo.txt"
echo "$("$llama/llama-completion" --version 2>&1 | head -1) / $(basename "$gguf") $(stat -f %z "$gguf") bytes" >> "$out/hostinfo.txt"
# -n 0: prompt processing only. -r: llama-bench's own repetitions inside one process,
# so every pp row is one warm process; the mean and stddev it prints are of those.
set -- "$bin" -m "$gguf" -p "$lengths" -n 0 -b "$ubatch" -ub "$ubatch" -fa 1 -r "$reps" -o md
echo "$*" > "$out/command.txt"
{ env | grep '^GGML_\|^LLAMA_' || true; echo "GGUF=$gguf"; echo "PP=$lengths"; echo "UBATCH=$ubatch"; echo "REPS=$reps"; sysctl iogpu.wired_limit_mb 2>/dev/null || true; } > "$out/env.txt"
{ date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-1.pre.txt"
rc=0
"$@" > "$out/run-1.log" 2>&1 || rc=$?
{ date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-1.post.txt"
printf 'test\tpp_tok_s_mean\tpp_tok_s_sd\treps\texit\n' > "$out/summary.tsv"
# llama-bench -o md prints "| ... | pp512 | 123.45 ± 1.23 |"; keep the test and the two numbers.
awk -v reps="$reps" -v rc="$rc" -F'|' '
    $0 ~ /\| *pp[0-9]+ *\|/ { test = $(NF-2); rate = $(NF-1); gsub(/ /, "", test); n = split(rate, a, "±"); gsub(/ /, "", a[1]); gsub(/ /, "", a[2]); printf "%s\t%s\t%s\t%s\t%s\n", test, a[1], a[2], reps, rc }
' "$out/run-1.log" >> "$out/summary.tsv"
echo "evidence: ${out#$repo/}"
cat "$out/summary.tsv"
