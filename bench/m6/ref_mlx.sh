#!/bin/sh
# External reference on the same machine: mlx_lm prompt processing on bench/l.json
# (the ledger's 2478-token prompt), one new process per run, filed as evidence.
#   bench/m6/ref_mlx.sh <label> [n]
#   MLX_VENV=~/LLM/ref/mlx-venv MLX_MODEL=~/LLM/ref/gemma-4-26b-a4b-it-UD-MLX-3bit bench/m6/ref_mlx.sh ref-mlx-3bit 6
# Calibration of a foreign runtime, not a ledger row (docs/m6-prefill/06). mlx_lm reports
# "Prompt: N tokens, X tokens-per-sec" for its own warm process (weights loaded, no hash),
# so it compares with the (b)-like quantity, not with (c).
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt env.txt prompt.txt run-N.log run-N.pre.txt run-N.post.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; shift
n=${1:-6}
venv=${MLX_VENV:-$HOME/LLM/ref/mlx-venv}
model=${MLX_MODEL:-$HOME/LLM/ref/gemma-4-26b-a4b-it-UD-MLX-3bit}
# The CLI has no prefill-step flag; generate_step's default (2048, the same as the
# ledger's chunk) is recorded from the installed source into env.txt.
py="$venv/bin/python"
[ -x "$py" ] || { echo "no python at $py"; exit 1; }
[ -f "$model/config.json" ] || { echo "no model at $model"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|llama-' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" > "$out/hostinfo.txt"
echo "$($py -c 'import mlx.core as mx, mlx_lm; print("mlx", mx.__version__, "mlx_lm", mlx_lm.__version__)') / $(basename "$model") $(du -sk "$model" | cut -f1) KiB" >> "$out/hostinfo.txt"
# The raw user text; mlx_lm applies the model's chat template itself.
python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.stdout.write("".join(x["content"] for x in m if x["role"]=="user"))' "$repo/bench/l.json" > "$out/prompt.txt"
set -- "$py" -m mlx_lm generate --model "$model" --prompt - --max-tokens 1 --temp 0
echo "$* < prompt.txt" > "$out/command.txt"
step=$(grep -m1 'prefill_step_size: int = ' "$($py -c 'import os, mlx_lm; print(os.path.join(os.path.dirname(mlx_lm.__file__), "generate.py"))')" | sed 's/.*= *//; s/,.*//')
{ env | grep '^MLX_' || true; echo "MLX_MODEL=$model"; echo "prefill_step_size=$step (mlx_lm generate_step default)";   sysctl iogpu.wired_limit_mb 2>/dev/null || true; } > "$out/env.txt"
printf 'run\tprompt_tok\tpp_tok_s\tpeak_gb\texit\n' > "$out/summary.tsv"
i=1
while [ "$i" -le "$n" ]; do
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$i.pre.txt"
    rc=0
    "$@" < "$out/prompt.txt" > "$out/run-$i.log" 2>&1 || rc=$?
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$i.post.txt"
    awk -v run="$i" -v rc="$rc" '
        /^Prompt:/ { tok = $2; for (k = 1; k <= NF; k++) if ($k ~ /tokens-per-sec/) pp = $(k - 1) }
        /^Peak memory:/ { peak = $3 }
        END { printf "%s\t%s\t%s\t%s\t%s\n", run, tok, pp, peak, rc }
    ' "$out/run-$i.log" >> "$out/summary.tsv"
    tail -1 "$out/summary.tsv"
    i=$((i + 1))
done
echo "evidence: ${out#$repo/}"
cat "$out/summary.tsv"
