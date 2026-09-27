#!/bin/sh
# External reference on the same machine: the prism fork of llama.cpp reading
# Ternary Bonsai 2 27B (PQ2_0) — prompt eval on one natural-text prompt, one new
# process per run, filed as evidence (docs/qwen38-27b/15-M6-PREFILL-PLAN.md).
#   bench/m6/ref_llamacpp_bonsai.sh <label> [n]
#   PROMPT=bench/m6/prompts/ja-twomachine.json bench/m6/ref_llamacpp_bonsai.sh bonsai-llamacpp-ja 6
#   GGML_METAL_TENSOR_DISABLE=1 bench/m6/ref_llamacpp_bonsai.sh bonsai-llamacpp-tensor-off 3
# Calibration of a foreign runtime, not a ledger row (docs/m6-prefill/06): the number is
# what llama.cpp reports as "prompt eval time" for its own warm process (weights
# mmapped, no integrity hash), so it compares with the (b)-like quantity, not with (c).
# The prompt is wrapped in the Qwen chat turn with thinking off, the way the pack's
# chat_template.jinja renders a user message with enable_thinking=false.
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt env.txt prompt.txt run-N.log run-N.pre.txt run-N.post.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; shift
n=${1:-6}
llama=${LLAMA_DIR:-$HOME/LLM/ref/prism-b10743}
gguf=${GGUF:-$HOME/LLM/ref/Ternary-Bonsai-2-27B-PQ2_0.gguf}
prompt=${PROMPT:-$repo/bench/m6/prompts/en-readme.json}
ubatch=${UBATCH:-2048}
ctx=${CTX:-4096}
bin="$llama/llama-completion"
[ -x "$bin" ] || bin="$llama/build/bin/llama-completion"
[ -x "$bin" ] || { echo "no llama-completion under $llama (or its build/bin)"; exit 1; }
[ -f "$gguf" ] || { echo "no GGUF at $gguf"; exit 1; }
[ -f "$prompt" ] || { echo "no prompt at $prompt"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|llama-' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" > "$out/hostinfo.txt"
echo "$("$bin" --version 2>&1 | head -1) / $(basename "$gguf") $(stat -f %z "$gguf") bytes" >> "$out/hostinfo.txt"
# The messages file is a list of {role, content}; the user turns are wrapped in the
# Qwen chat template with the empty think block that enable_thinking=false renders.
python3 - "$prompt" > "$out/prompt.txt" <<'EOF'
import json, sys
msgs = json.load(open(sys.argv[1]))
parts = []
for m in msgs:
    parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
sys.stdout.write("".join(parts) + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
EOF
# -fit off: everything on the GPU or fail loudly (docs/m6-prefill/29). -lv 4 keeps the
# Metal backend's device lines in run-N.log.
set -- "$bin" -m "$gguf" -f "$out/prompt.txt" -n 1 -c "$ctx" -b "$ubatch" -ub "$ubatch" \
    -fa on --temp 0 -no-cnv --no-display-prompt -fit off -lv 4
echo "$*" > "$out/command.txt"
{ env | grep '^GGML_\|^LLAMA_' || true; echo "GGUF=$gguf"; echo "PROMPT=${prompt#$repo/}"; echo "UBATCH=$ubatch"; echo "CTX=$ctx"; sysctl iogpu.wired_limit_mb 2>/dev/null || true; } > "$out/env.txt"
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
