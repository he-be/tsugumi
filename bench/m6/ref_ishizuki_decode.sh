#!/bin/sh
# Decode of Ternary Bonsai 2 27B under ishizuki with and without the MTP head, filed as
# evidence: one ishizuki-decode-probe process (docs/qwen38-27b/16 §14〜§16) on a rendered
# prompt, the server's official thinking-off sampling, three seeds per setting. Built for
# the M1 mini / M6 comparison: the same binary (ishizuki_ship.sh) and the same pack on both.
#   bench/m6/ref_ishizuki_decode.sh <label> [reps 10] [gen 300]
#   ISHIZUKI_BIN=~/LLM/ref/ishizuki-bin-final bench/m6/ref_ishizuki_decode.sh bonsai-decode-ja
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt build-info.txt command.txt env.txt prompt.txt run.log pre.txt post.txt
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}
reps=${2:-10}
gen=${3:-300}
bindir=${ISHIZUKI_BIN:-$HOME/LLM/ref/ishizuki-bin-final}
pack=${MLX_PACK:-$HOME/LLM/ref/Ternary-Bonsai-2-27B-mlx-2bit-mtp}
prompt=${PROMPT:-$repo/bench/m6/prompts/ja-twomachine.rendered.txt}
bin="$bindir/ishizuki-decode-probe"
[ -x "$bin" ] || { echo "no $bin (run bench/m6/ishizuki_ship.sh on the MBP)"; exit 1; }
[ -f "$bindir/build-info.txt" ] || { echo "no build-info.txt beside $bin"; exit 1; }
[ -f "$pack/mtp.safetensors" ] || { echo "no mtp.safetensors under $pack"; exit 1; }
[ -f "$prompt" ] || { echo "no prompt at $prompt"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|mlx_vlm|llama-|ishizuki-' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" "$bin" > "$out/hostinfo.txt"
echo "$(basename "$pack") base $(stat -L -f %z "$pack/base.safetensors") mtp $(stat -f %z "$pack/mtp.safetensors") bytes" >> "$out/hostinfo.txt"
cp "$bindir/build-info.txt" "$out/build-info.txt"
cp "$prompt" "$out/prompt.txt"
set -- "$bin" "$pack" "$out/prompt.txt" "$reps" "$gen"
echo "$*" > "$out/command.txt"
{ env | grep '^ISHIZUKI_\|^MLX_' || true; echo "MLX_PACK=$pack"; echo "PROMPT=${prompt#$repo/} $(shasum -a 256 "$prompt" | cut -d' ' -f1)"; sysctl iogpu.wired_limit_mb 2>/dev/null || true; } > "$out/env.txt"
{ date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/pre.txt"
rc=0
"$@" > "$out/run.log" 2>&1 || rc=$?
{ date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/post.txt"
echo "exit $rc" >> "$out/run.log"
echo "evidence: ${out#$repo/}"
grep -E '^mtp head|^prefilled|^fold|^generate|^exit' "$out/run.log" || true
printf 'swapouts %s -> %s\n' "$(awk '/Swapouts/ {print $2}' "$out/pre.txt")" "$(awk '/Swapouts/ {print $2}' "$out/post.txt")"
exit "$rc"
