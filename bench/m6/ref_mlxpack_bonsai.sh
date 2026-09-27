#!/bin/sh
# External reference on the same machine: the MLX 2-bit pack of Ternary Bonsai 2 27B
# read through the runtime the pack ships (runtime/artifact.py; stock mlx_lm cannot
# load it), one natural-text prompt pushed through the model in one call, one new
# process per run, filed as evidence (docs/qwen38-27b/15-M6-PREFILL-PLAN.md §5-2).
#   bench/m6/ref_mlxpack_bonsai.sh <label> [n]
#   PROMPT=bench/m6/prompts/ja-twomachine.json bench/m6/ref_mlxpack_bonsai.sh bonsai-mlxpack-ja 6
# Calibration of a foreign runtime, not a ledger row (docs/m6-prefill/06). The number
# is the wall clock from the prompt array to mx.eval of the last-position logits, in a
# warm process (weights loaded and evaluated first), so it compares with the (b)-like
# quantity, not with (c). No chunking: the pack's quickstart.py runs the prompt in one
# call, and so does this.
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt env.txt prompt.txt run-N.log run-N.pre.txt run-N.post.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; shift
n=${1:-6}
venv=${MLX_VENV:-$HOME/LLM/ref/mlx-venv}
pack=${MLX_PACK:-$HOME/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit}
prompt=${PROMPT:-$repo/bench/m6/prompts/en-readme.json}
py="$venv/bin/python"
[ -x "$py" ] || { echo "no python at $py"; exit 1; }
[ -f "$pack/model.safetensors" ] || { echo "no model.safetensors under $pack"; exit 1; }
[ -f "$pack/runtime/artifact.py" ] || { echo "no runtime/artifact.py under $pack"; exit 1; }
[ -f "$prompt" ] || { echo "no prompt at $prompt"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|llama-' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" > "$out/hostinfo.txt"
echo "$($py -c 'import mlx.core as mx, mlx_lm, transformers, tokenizers; print("mlx", mx.__version__, "mlx_lm", mlx_lm.__version__, "transformers", transformers.__version__, "tokenizers", tokenizers.__version__)') / $(basename "$pack") model.safetensors $(stat -f %z "$pack/model.safetensors") bytes" >> "$out/hostinfo.txt"
# The same wrapping as ref_llamacpp_bonsai.sh: the Qwen chat turn with the empty think
# block that the pack's chat_template.jinja renders for enable_thinking=false.
python3 - "$prompt" > "$out/prompt.txt" <<'EOF'
import json, sys
msgs = json.load(open(sys.argv[1]))
parts = []
for m in msgs:
    parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
sys.stdout.write("".join(parts) + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
EOF
cat > "$out/prefill.py" <<'EOF'
import sys, time, json
import mlx.core as mx
pack, prompt_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, pack + "/runtime")
from artifact import load_model
from tokenizers import Tokenizer
t0 = time.perf_counter()
model, config = load_model(pack)
ids = Tokenizer.from_file(pack + "/tokenizer.json").encode(open(prompt_path).read(), add_special_tokens=False).ids
# Touch the weights once so the timed pass does not include the first page-in of the file.
cache = model.make_cache()
mx.eval(model.lm_head(model.model(mx.array([ids[:8]]), cache=cache)[:, -1:, :]))
load_s = time.perf_counter() - t0
mx.reset_peak_memory()
cache = model.make_cache()
t1 = time.perf_counter()
logits = model.lm_head(model.model(mx.array([ids]), cache=cache)[:, -1:, :])
mx.eval(logits)
prefill_s = time.perf_counter() - t1
tok = int(mx.argmax(logits[0, -1]).item())
print(json.dumps({"prompt_tok": len(ids), "prefill_s": round(prefill_s, 4), "pp_tok_s": round(len(ids) / prefill_s, 1),
                  "load_s": round(load_s, 3), "peak_gb": round(mx.get_peak_memory() / 1e9, 3), "argmax": tok}))
EOF
set -- "$py" "$out/prefill.py" "$pack" "$out/prompt.txt"
echo "$*" > "$out/command.txt"
{ env | grep '^MLX_' || true; echo "MLX_PACK=$pack"; echo "PROMPT=${prompt#$repo/}"; sysctl iogpu.wired_limit_mb 2>/dev/null || true; } > "$out/env.txt"
printf 'run\tprompt_tok\tprefill_s\tpp_tok_s\tload_s\tpeak_gb\targmax\texit\n' > "$out/summary.tsv"
i=1
while [ "$i" -le "$n" ]; do
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$i.pre.txt"
    rc=0
    "$@" > "$out/run-$i.log" 2>&1 || rc=$?
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$i.post.txt"
    python3 - "$i" "$rc" "$out/run-$i.log" >> "$out/summary.tsv" <<'EOF'
import json, sys
run, rc, path = sys.argv[1], sys.argv[2], sys.argv[3]
row = {}
for line in open(path, errors="replace"):
    line = line.strip()
    if line.startswith("{") and "prefill_s" in line:
        try:
            row = json.loads(line)
        except ValueError:
            pass
print("\t".join(str(row.get(k, "")) for k in ("prompt_tok", "prefill_s", "pp_tok_s", "load_s", "peak_gb", "argmax")).join([run + "\t", "\t" + rc]))
EOF
    tail -1 "$out/summary.tsv"
    i=$((i + 1))
done
echo "evidence: ${out#$repo/}"
cat "$out/summary.tsv"
