#!/bin/zsh
# kajimachine's requests on this machine, one runtime per arm, filed as evidence.
#
# Starts the server given after `--` (it must listen on 127.0.0.1:$PORT and print READY_PATTERN
# when it can take requests), replays the recorded payloads with bench/m6/kj_replay.py
# <rounds> times, samples vm_stat every 0.5 s, and stops the server. The payloads and the
# answers carry visitors' photographs and stay outside the repository (~/diet-probe).
#
#   PORT=8081 READY='ready at' bench/m6/kj_replay_probe.sh kj-tsugumi-qat 2 -- \
#       .build/release/TsugumiServer --model scratch/gemma4-qat-sym.gturbo --port 8081 \
#       --model-id gemma4-26b-a4b-lattice --reasoning-budget 0 --draft-block-size 4
#
# Output: bench/m6/results/<date>-<host>-<label>/
#   hostinfo.txt command.txt env.txt server.log plays.jsonl mem.tsv pre.txt post.txt summary.tsv
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
label=${1:?label}; rounds=${2:?rounds}; shift 2
[ "$1" = "--" ] && shift
port=${PORT:?PORT}
ready=${READY:?READY}
payloads=${PAYLOADS:-$HOME/diet-probe/kj_payloads.jsonl}
cd "$repo"
[ -f "$payloads" ] || { echo "no payloads at $payloads"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiPackageTests|mlx_lm|mlx-lm|mlx_vlm|llama-|ishizuki-|omlx' >/dev/null; then
    echo "another model process is running; refusing to measure"; exit 1
fi
host=$(sysctl -n hw.model | tr ',' '-')
out="$repo/bench/m6/results/$(date '+%Y-%m-%d')-$host-$label"
mkdir -p "$out" "$HOME/diet-probe/answers"
sh "$repo/bench/hostinfo.sh" > "$out/hostinfo.txt"
[ -n "${RUNTIME_NOTE:-}" ] && echo "$RUNTIME_NOTE" >> "$out/hostinfo.txt"
echo "$* & python3 -I bench/m6/kj_replay.py http://127.0.0.1:$port <payloads> $rounds <answers>" > "$out/command.txt"
{
    env | grep -E '^(TF_|LLAMA_|GGML_|MLX_)' || true
    echo "payloads $(shasum -a 256 "$payloads" | cut -d' ' -f1) $(wc -l < "$payloads" | tr -d ' ') requests, rounds=$rounds"
    sysctl iogpu.wired_limit_mb
    mount | grep -q '/Volumes/SSD256' && echo "SSD256 mounted" || echo "SSD256 not mounted"
} > "$out/env.txt"
mem() { vm_stat | awk '/Pages free/{r=$3} /wired down/{w=$4} /File-backed/{f=$3} /Anonymous/{a=$3} /occupied by compressor/{c=$5} END{gsub(/\./,"",r);gsub(/\./,"",w);gsub(/\./,"",f);gsub(/\./,"",a);gsub(/\./,"",c); print r, w, f, a, c}'; }
{ date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; sysctl vm.swapusage; } > "$out/pre.txt"
( echo "t free wired file anon compressor"; while true; do echo "$(date +%s.%N) $(mem)"; sleep 0.5; done ) > "$out/mem.tsv" &
sampler=$!
"$@" > "$out/server.log" 2>&1 &
server=$!
cleanup() { kill $server $sampler 2>/dev/null || true; }
trap cleanup EXIT
for i in $(seq 1 300); do grep -q "$ready" "$out/server.log" && break; kill -0 $server 2>/dev/null || break; sleep 1; done
grep -q "$ready" "$out/server.log" || { echo "server not ready"; tail -20 "$out/server.log"; exit 1; }
sleep 3
python3 -I bench/m6/kj_replay.py "http://127.0.0.1:$port" "$payloads" "$rounds" "$HOME/diet-probe/answers/$label.jsonl" > "$out/plays.jsonl" 2> "$out/client.err" || echo "client exit $?" >> "$out/client.err"
ps -o rss= -p $server | awk '{printf "server_rss_gib %.2f\n", $1/1048576}' > "$out/rss.txt" || true
{ date '+%H:%M:%S'; memory_pressure -Q 2>/dev/null || true; vm_stat; sysctl vm.swapusage; } > "$out/post.txt"
cleanup; trap - EXIT
wait $server 2>/dev/null || true
python3 -I - "$out" > "$out/summary.tsv" <<'PY'
import json, statistics as st, sys
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/plays.jsonl") if l.startswith("{")]
mem = [l.split() for l in open(f"{out}/mem.tsv")][1:]
gib = 16384 / 2**30
med = lambda xs: round(st.median(xs), 2) if xs else "-"
print("round\tplays\twall_s_median\twall_s_min\twall_s_max\tdisk_gb_median\tpageins_gib_median\tswapouts_sum\tread_s\titem_s\tpick_s\tdeed_s\tread_first_s\tprefill_tps\tdecode_tps\tinvalid_json\terrors\twired_max_gib")
for n in sorted({r["round"] for r in rows}):
    part = [r for r in rows if r["round"] == n]
    calls = [c for r in part for c in r["calls"]]
    by = lambda name: med([c["s"] for c in calls if c["name"] == name])
    names = list(dict.fromkeys(c["name"] for c in calls))
    pre = [c["prompt_n"] / (c["prompt_ms"] / 1000) for c in calls if c.get("prompt_ms") and c.get("prompt_n", 0) >= 100]
    dec = [c["predicted_n"] / (c["predicted_ms"] / 1000) for c in calls if c.get("predicted_ms") and c.get("predicted_n", 0) >= 16]
    print("\t".join(str(x) for x in [n, len(part), med([r["wall_s"] for r in part]), min(r["wall_s"] for r in part),
          max(r["wall_s"] for r in part), med([r["disk_gb"] for r in part]), med([r["pageins_gib"] for r in part]),
          sum(r["swapouts"] for r in part), *[by(x) for x in (names + ["-"] * 4)[:4]],
          med([c["first_s"] for c in calls if c.get("first_s")]), med(pre), med(dec),
          sum(1 for c in calls if not c["valid"]), sum(1 for c in calls if c["error"]),
          round(max(float(m[2]) for m in mem) * gib, 2)]))
PY
echo "evidence: ${out#$repo/}"
cat "$out/summary.tsv"
