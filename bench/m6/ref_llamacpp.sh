#!/bin/sh
# External reference on the same machine: llama.cpp prompt eval on bench/l.json
# (the ledger's 2478-token prompt), one new process per run, filed as evidence.
#   bench/m6/ref_llamacpp.sh <label> [n]
#   LLAMA_DIR=~/LLM/ref/llama.cpp GGUF=~/LLM/ref/x.gguf bench/m6/ref_llamacpp.sh ref-llamacpp-iq3xxs 6
# This is a calibration of a foreign runtime, not a ledger row (docs/m6-prefill/06):
# the numbers are what llama.cpp reports as "prompt eval time" (its own warm process,
# no integrity hash), so they compare with the (b)-like quantity, not with (c).
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt env.txt prompt.txt run-N.log run-N.pre.txt run-N.post.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; shift
n=${1:-6}
llama=${LLAMA_DIR:-$HOME/LLM/ref/llama.cpp}
gguf=${GGUF:-$HOME/LLM/ref/gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf}
ubatch=${UBATCH:-2048}
ctx=${CTX:-4096}
bin="$llama/build/bin/llama-completion"
[ -x "$bin" ] || bin="$llama/build/bin/llama-cli"
[ -x "$bin" ] || { echo "no llama-completion/llama-cli under $llama/build/bin"; exit 1; }
[ -f "$gguf" ] || { echo "no GGUF at $gguf"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|llama-' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" > "$out/hostinfo.txt"
echo "llama.cpp $(git -C "$llama" rev-parse --short HEAD 2>/dev/null || echo '?') / $(basename "$gguf") $(stat -f %z "$gguf") bytes" >> "$out/hostinfo.txt"
# Same messages as the ledger prompt, wrapped in Gemma's chat turn. llama.cpp adds <bos> itself.
python3 - "$repo/bench/l.json" > "$out/prompt.txt" <<'EOF'
import json, sys
msgs = json.load(open(sys.argv[1]))
parts = []
for m in msgs:
    role = "model" if m["role"] == "assistant" else m["role"]
    parts.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n")
sys.stdout.write("".join(parts) + "<start_of_turn>model\n")
EOF
# -fit off: everything on the GPU or fail loudly. With -fit on, llama.cpp silently
# moves MoE expert tensors to the CPU when the wired budget is short (seen on 16 GB
# with the default iogpu.wired_limit_mb), which would measure the wrong thing.
# -lv 4 (trace) keeps the Metal backend's device lines in run-N.log.
set -- "$bin" -m "$gguf" -f "$out/prompt.txt" -n 1 -c "$ctx" -b "$ubatch" -ub "$ubatch" \
    -fa on --temp 0 -no-cnv --no-display-prompt -fit off -lv 4
echo "$*" > "$out/command.txt"
{ env | grep '^GGML_\|^LLAMA_' || true; echo "GGUF=$gguf"; echo "UBATCH=$ubatch"; echo "CTX=$ctx";   sysctl iogpu.wired_limit_mb 2>/dev/null || true; } > "$out/env.txt"
printf 'run\tprompt_tok\tprompt_ms\tpp_tok_s\tload_ms\ttensor_api\texit\n' > "$out/summary.tsv"
i=1
while [ "$i" -le "$n" ]; do
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$i.pre.txt"
    rc=0
    "$@" > "$out/run-$i.log" 2>&1 || rc=$?
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$i.post.txt"
    awk -v run="$i" -v rc="$rc" '
        /prompt eval time/ { for (k = 1; k <= NF; k++) { if ($k == "=") { ms = $(k + 1) } if ($k == "/") { tok = $(k + 1) } } }
        /llama_perf_context_print: *load time/ || /load time *=/ { for (k = 1; k <= NF; k++) if ($k == "=") { load = $(k + 1) } }
        /has tensor|tensor API|use tensor|GGML_METAL_HAS_TENSOR|tensor *= *(true|false|1|0)/ { t = $0; sub(/^.*tensor/, "tensor", t); tensor = t }
        END { pp = (ms > 0) ? tok / (ms / 1000.0) : 0
              printf "%s\t%s\t%s\t%.1f\t%s\t%s\t%s\n", run, tok, ms, pp, load, tensor, rc }
    ' "$out/run-$i.log" >> "$out/summary.tsv"
    tail -1 "$out/summary.tsv"
    i=$((i + 1))
done
echo "evidence: ${out#$repo/}"
cat "$out/summary.tsv"
