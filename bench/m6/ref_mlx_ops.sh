#!/bin/sh
# MLX kernels on Tsugumi's prefill shapes, per stage (bench/m6/ref_mlx_ops.py), one new
# process per run, filed as evidence. Calibration of a foreign runtime, not a ledger row.
#   bench/m6/ref_mlx_ops.sh <label> [n] [dtype]
#   bench/m6/ref_mlx_ops.sh ref-mlx-ops-fp16 3 float16
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt env.txt run-N.log run-N.pre.txt run-N.post.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; shift
n=${1:-3}
dtype=${2:-float16}
venv=${MLX_VENV:-$HOME/LLM/ref/mlx-venv}
py="$venv/bin/python"
[ -x "$py" ] || { echo "no python at $py"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|llama-' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" > "$out/hostinfo.txt"
echo "$($py -c 'import mlx.core as mx; print("mlx", mx.__version__)')" >> "$out/hostinfo.txt"
set -- "$py" "$repo/bench/m6/ref_mlx_ops.py" 5 "$dtype"
echo "$*" > "$out/command.txt"
{ env | grep '^MLX_' || true; sysctl iogpu.wired_limit_mb 2>/dev/null || true; } > "$out/env.txt"
i=1
while [ "$i" -le "$n" ]; do
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$i.pre.txt"
    rc=0
    "$@" > "$out/run-$i.log" 2>&1 || rc=$?
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$i.post.txt"
    echo "run $i exit $rc"
    i=$((i + 1))
done
# summary.tsv: one row per stage, the per-run medians side by side.
python3 - "$out" "$n" <<'EOF' > "$out/summary.tsv"
import sys
out, n = sys.argv[1], int(sys.argv[2])
runs = []
for i in range(1, n + 1):
    rows = {}
    for s in open(f'{out}/run-{i}.log'):
        f = s.rstrip('\n').split('\t')
        if len(f) >= 2 and f[0] not in ('stage',):
            try:
                rows[f[0]] = float(f[1])
            except ValueError:
                pass
    runs.append(rows)
print('stage\t' + '\t'.join(f'run{i}' for i in range(1, n + 1)))
for k in runs[0]:
    print(k + '\t' + '\t'.join(f'{r.get(k, float("nan")):.4f}' for r in runs))
EOF
echo "evidence: ${out#$repo/}"
cat "$out/summary.tsv"
