#!/bin/sh
# External reference on the same machine: ishizuki (struffl/ishizuki, MLX Swift) reading
# the MLX 2-bit pack of Ternary Bonsai 2 27B through the driver built on the MBP by
# bench/m6/ishizuki_ship.sh, one natural-text prompt, one new process per run, filed as
# evidence (docs/qwen38-27b/15-M6-PREFILL-PLAN.md §3-2 / §7-5).
#   bench/m6/ref_ishizuki_bonsai.sh <label> [n]
#   PROMPT=bench/m6/prompts/ja-twomachine.json bench/m6/ref_ishizuki_bonsai.sh bonsai-ishizuki-ja 6
# Calibration of a foreign runtime, not a ledger row (docs/m6-prefill/06). The number is
# ishizuki's own Generator.promptSeconds (the one its EngineBenchProbe reads) in a warm
# process, so it compares with the (b)-like quantity, not with (c).
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt build-info.txt command.txt env.txt prompt.txt run-N.log run-N.pre.txt run-N.post.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; shift
n=${1:-6}
bindir=${ISHIZUKI_BIN:-$HOME/LLM/ref/ishizuki-bin}
pack=${MLX_PACK:-$HOME/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit}
prompt=${PROMPT:-$repo/bench/m6/prompts/en-readme.json}
bin="$bindir/ishizuki-prefill"
[ -x "$bin" ] || { echo "no $bin (run bench/m6/ishizuki_ship.sh on the MBP)"; exit 1; }
[ -f "$bindir/build-info.txt" ] || { echo "no build-info.txt beside $bin"; exit 1; }
[ -f "$pack/model.safetensors" ] || { echo "no model.safetensors under $pack"; exit 1; }
[ -f "$prompt" ] || { echo "no prompt at $prompt"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|mlx_vlm|llama-|ishizuki-prefill' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" "$bin" > "$out/hostinfo.txt"
echo "$(basename "$pack") model.safetensors $(stat -f %z "$pack/model.safetensors") bytes" >> "$out/hostinfo.txt"
# The binary was built elsewhere; its build machine, ishizuki commit and minos travel with it.
cp "$bindir/build-info.txt" "$out/build-info.txt"
# The same wrapping as ref_llamacpp_bonsai.sh and ref_mlxpack_bonsai.sh.
python3 - "$prompt" > "$out/prompt.txt" <<'EOF'
import json, sys
msgs = json.load(open(sys.argv[1]))
parts = []
for m in msgs:
    parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
sys.stdout.write("".join(parts) + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
EOF
set -- "$bin" "$pack" "$out/prompt.txt"
echo "$*" > "$out/command.txt"
{ env | grep '^ISHIZUKI_\|^MLX_' || true; echo "MLX_PACK=$pack"; echo "PROMPT=${prompt#$repo/}"; sysctl iogpu.wired_limit_mb 2>/dev/null || true; } > "$out/env.txt"
printf 'run\tprompt_tok\tprefill_s\tpp_tok_s\tload_s\tpeak_gb\tprefill_chunk\tfirst_token\texit\n' > "$out/summary.tsv"
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
cols = [str(row.get(k, "")) for k in ("prompt_tok", "prefill_s", "pp_tok_s", "load_s", "peak_gb", "prefill_chunk", "first_token")]
print("\t".join([run] + cols + [rc]))
EOF
    tail -1 "$out/summary.tsv"
    i=$((i + 1))
done
echo "evidence: ${out#$repo/}"
cat "$out/summary.tsv"
