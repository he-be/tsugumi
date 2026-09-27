#!/bin/sh
# External reference on the same machine: the MLX 2-bit pack of Ternary Bonsai 2 27B
# read through the loader the pack ships (runtime/vision_artifact.py on mlx-vlm; the
# text-only runtime/artifact.py rejects the pack's schema 2 config, and stock mlx_lm
# cannot load it at all), one natural-text prompt, one new process per run, filed as
# evidence (docs/qwen38-27b/15-M6-PREFILL-PLAN.md §5-2).
#   bench/m6/ref_mlxpack_bonsai.sh <label> [n]
#   PROMPT=bench/m6/prompts/ja-twomachine.json bench/m6/ref_mlxpack_bonsai.sh bonsai-mlxpack-ja 6
# Calibration of a foreign runtime, not a ledger row (docs/m6-prefill/06). The number is
# what mlx_vlm.generate reports as prompt_tps / prompt_tokens for its own warm process
# (weights loaded, no hash), so it compares with the (b)-like quantity, not with (c).
# The venv must carry the pack's pinned versions (runtime/requirements.txt: mlx 0.32.0,
# mlx-vlm 0.6.3, transformers 5.5.0); the Gemma venv (~/LLM/ref/mlx-venv) has no mlx-vlm.
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt env.txt prompt.txt run-N.log run-N.pre.txt run-N.post.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; shift
n=${1:-6}
venv=${MLX_VENV:-$HOME/LLM/ref/bonsai-venv}
pack=${MLX_PACK:-$HOME/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit}
prompt=${PROMPT:-$repo/bench/m6/prompts/en-readme.json}
py="$venv/bin/python"
[ -x "$py" ] || { echo "no python at $py"; exit 1; }
[ -f "$pack/model.safetensors" ] || { echo "no model.safetensors under $pack"; exit 1; }
[ -f "$pack/runtime/vision_artifact.py" ] || { echo "no runtime/vision_artifact.py under $pack"; exit 1; }
[ -f "$prompt" ] || { echo "no prompt at $prompt"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|mlx_vlm|llama-' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" > "$out/hostinfo.txt"
echo "$($py -c 'import mlx.core as mx, mlx_vlm, mlx_lm, transformers; print("mlx", mx.__version__, "mlx_vlm", mlx_vlm.__version__, "mlx_lm", mlx_lm.__version__, "transformers", transformers.__version__)') / $(basename "$pack") model.safetensors $(stat -f %z "$pack/model.safetensors") bytes" >> "$out/hostinfo.txt"
# The same wrapping as ref_llamacpp_bonsai.sh: the Qwen chat turn with the empty think
# block that the pack's chat_template.jinja renders for enable_thinking=false. It is
# handed to mlx_vlm.generate as the prompt text, so no template is applied again.
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
from vision_artifact import load_vl_model
from mlx_vlm import generate
t0 = time.perf_counter()
model, processor, config = load_vl_model(pack)
load_s = time.perf_counter() - t0
prompt = open(prompt_path).read()
mx.reset_peak_memory()
r = generate(model, processor, prompt, max_tokens=1, temperature=0.0, verbose=False)
print(json.dumps({"prompt_tok": r.prompt_tokens, "prompt_tps": round(r.prompt_tps, 1),
                  "prefill_s": round(r.prompt_tokens / r.prompt_tps, 4) if r.prompt_tps else None,
                  "load_s": round(load_s, 3), "peak_gb": round(r.peak_memory, 3), "cached_tokens": r.cached_tokens,
                  "text": r.text}))
EOF
set -- "$py" "$out/prefill.py" "$pack" "$out/prompt.txt"
echo "$*" > "$out/command.txt"
# mlx_vlm's prefill chunk is its generate_step default; record it from the installed source.
dispatch=$($py -c 'import inspect, mlx_vlm; print(inspect.getsourcefile(mlx_vlm.generate))')
step=$(grep -m1 'prefill_step_size: int = ' "$dispatch" | sed 's/.*= *//; s/,.*//')
{ env | grep '^MLX_' || true; echo "MLX_PACK=$pack"; echo "PROMPT=${prompt#$repo/}"; echo "prefill_step_size=$step (mlx_vlm generate default, $dispatch)"; sysctl iogpu.wired_limit_mb 2>/dev/null || true; } > "$out/env.txt"
printf 'run\tprompt_tok\tprefill_s\tpp_tok_s\tload_s\tpeak_gb\tcached_tok\texit\n' > "$out/summary.tsv"
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
    if line.startswith("{") and "prompt_tps" in line:
        try:
            row = json.loads(line)
        except ValueError:
            pass
cols = [str(row.get(k, "")) for k in ("prompt_tok", "prefill_s", "prompt_tps", "load_s", "peak_gb", "cached_tokens")]
print("\t".join([run] + cols + [rc]))
EOF
    tail -1 "$out/summary.tsv"
    i=$((i + 1))
done
echo "evidence: ${out#$repo/}"
cat "$out/summary.tsv"
