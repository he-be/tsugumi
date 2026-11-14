#!/bin/zsh
# Wired memory while Gemma 4 runs, filed as evidence (docs/m6-prefill/46).
#
# Each run starts a fresh model process, samples `vm_stat` every 0.25 s for the whole run
# (wired / file-backed / free pages), sends a fixed request mix, reads the internal SSD's
# counters before and after, and stops the process. Two kinds:
#   server  TsugumiServer with the production arguments of the exhibit (--reasoning-budget 0
#           --draft-block-size 4) and bench/m6/wired_client.py: one picture, 3 rounds of
#           one streamed 128-token description + three 32-token follow-ups.
#   cli     TsugumiCLI, one 128-token answer to a text prompt (the CLI takes --image or
#           --prompt, not both), the same slots / draft / context as the server.
# An arm is chosen with TF_ variables in the environment; they are written to env.txt.
#
#   IMAGE=~/wired-probe/05_m.jpg bench/m6/wired_probe.sh wired-server-mmap server 3
#   TF_EXPERT_MMAP=0 IMAGE=… bench/m6/wired_probe.sh wired-server-pread server 3
#   bench/m6/wired_probe.sh wired-cli-mmap cli 3
#
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt env.txt run-N.log run-N.client.txt run-N.wired.tsv
#   run-N.pre.txt run-N.post.txt run-N.io.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}
kind=${2:?server or cli}
reps=${3:-3}
model=scratch/gemma4-qat-sym.gturbo
cd "$repo"
case "$kind" in
    server) bin=.build/release/TsugumiServer
            image=${IMAGE:?IMAGE=<picture> is required for the server kind}
            [ -f "$image" ] || { echo "no picture at $image"; exit 1; } ;;
    cli) bin=.build/release/TsugumiCLI ;;
    *) echo "kind is server or cli"; exit 1 ;;
esac
[ -x "$bin" ] || { echo "no $bin (swift build -c release)"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|mlx_vlm|llama-|ishizuki-' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out"
sh "$repo/bench/hostinfo.sh" "$bin" > "$out/hostinfo.txt"
prompt="この写真に写っているものを、日本語で 3 文で説明してください。"
if [ "$kind" = server ]; then
    set -- "$bin" --model "$model" --port 8081 --model-id gemma4-26b-a4b-lattice \
        --reasoning-budget 0 --draft-block-size 4
    echo "$* & python3 -I bench/m6/wired_client.py <IMAGE> 3" > "$out/command.txt"
else
    set -- "$bin" --model "$model" --prompt "$prompt" --max-new 128 \
        --expert-cache-slots 32 --draft-block-size 4 --max-context 16384
    echo "$*" > "$out/command.txt"
fi
{
    env | grep '^TF_' || true
    [ "$kind" = server ] && echo "IMAGE=$(basename "$image") $(shasum -a 256 "$image" | cut -d' ' -f1)"
    sysctl iogpu.wired_limit_mb
    mount | grep -q '/Volumes/SSD256' && echo "SSD256 mounted" || echo "SSD256 not mounted (expert mirror off)"
} > "$out/env.txt"

pages() { vm_stat | awk '/wired down/{w=$4} /File-backed/{f=$3} /Pages free/{r=$3} END{gsub(/\./,"",w); gsub(/\./,"",f); gsub(/\./,"",r); print w, f, r}'; }
disk0() { iostat -d -I disk0 | awk 'END{print $3}'; }

printf 'run\twired_idle_gib\twired_max_gib\tfile_min_gib\twired_plus_file_min_gib\twired_plus_file_max_gib\twired_after_5s_gib\tdisk0_read_gb\twall_s\tswapouts\texit\n' > "$out/summary.tsv"
for n in $(seq 1 "$reps"); do
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$n.pre.txt"
    ( while true; do echo "$(date +%s.%N) $(pages)"; sleep 0.25; done ) > "$out/run-$n.wired.tsv" &
    sampler=$!
    rc=0
    if [ "$kind" = server ]; then
        "$@" > "$out/run-$n.log" 2>&1 &
        pid=$!
        for i in $(seq 1 240); do grep -q 'ready at' "$out/run-$n.log" && break; sleep 1; done
        sleep 3
        echo "idle $(date +%s.%N)" > "$out/run-$n.client.txt"
        before=$(disk0); started=$(date +%s.%N)
        python3 -I "$repo/bench/m6/wired_client.py" "$image" 3 >> "$out/run-$n.client.txt" 2>&1 || rc=$?
    else
        sleep 2
        echo "idle $(date +%s.%N)" > "$out/run-$n.client.txt"
        before=$(disk0); started=$(date +%s.%N)
        "$@" > "$out/run-$n.log" 2>&1 || rc=$?
    fi
    ended=$(date +%s.%N); after=$(disk0)
    echo "done $ended" >> "$out/run-$n.client.txt"
    sleep 5
    echo "after5 $(date +%s.%N)" >> "$out/run-$n.client.txt"
    if [ "$kind" = server ]; then kill "$pid"; wait "$pid" 2>/dev/null || true; fi
    sleep 1
    kill "$sampler"
    { date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; } > "$out/run-$n.post.txt"
    echo "disk0_mb_before $before after $after" > "$out/run-$n.io.txt"
    swap=$(( $(awk '/Swapouts/ {gsub(/\./,"",$2); print $2}' "$out/run-$n.post.txt") - $(awk '/Swapouts/ {gsub(/\./,"",$2); print $2}' "$out/run-$n.pre.txt") ))
    python3 -I - "$out" "$n" "$before" "$after" "$started" "$ended" "$swap" "$rc" >> "$out/summary.tsv" <<'EOF'
import sys
out, n, before, after, started, ended, swap, rc = sys.argv[1:]
gib = 16384 / 2**30
marks = dict(line.split()[:2] for line in open(f"{out}/run-{n}.client.txt") if line.split()[0] in ("idle", "done", "after5"))
rows = [tuple(float(x) for x in line.split()) for line in open(f"{out}/run-{n}.wired.tsv") if len(line.split()) == 4]
idle, done, after5 = float(marks["idle"]), float(marks["done"]), float(marks["after5"])
run = [r for r in rows if idle < r[0] <= done]
at_idle = [r for r in rows if r[0] <= idle][-1]
at_after = [r for r in rows if r[0] <= after5][-1]
both = [(r[1] + r[2]) * gib for r in run]
print("\t".join([n, f"{at_idle[1]*gib:.2f}", f"{max(r[1] for r in run)*gib:.2f}", f"{min(r[2] for r in run)*gib:.2f}",
                 f"{min(both):.2f}", f"{max(both):.2f}", f"{at_after[1]*gib:.2f}",
                 f"{(float(after) - float(before)) / 1024:.2f}", f"{float(ended) - float(started):.2f}", swap, rc]))
EOF
    sleep 3
done
echo "evidence: ${out#$repo/}"
cat "$out/summary.tsv"
