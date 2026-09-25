#!/bin/zsh
# Q0 / ledger bench for Qwen3.8-Flash-Next (docs/qfn-prefill/04 Q0, 06). Evidence is filed automatically.
#   bench/qfn/pair_bench.sh <label> [n] --solo [-- extra TsugumiServer args]
# --solo is the MBP alone. The two-machine form (M6 started over bridge0) is added with Q1; without --solo this refuses.
#
# Two bodies, the same script for both columns (06 §0):
#   本文 1: the natural prompts of bench/m6/prompts/ (5, 2345-2499 Gemma tokens), each in a fresh TsugumiServer, one
#           request with max_completion_tokens 1 and cache_prompt false; timings.prompt_ms (m6-prefill/43 §1.2).
#           n rounds, the order of the five turned by one each round so no body runs twice in a row (43 §3).
#   本文 2: one recorded turn replayed through the app's engine in one process (ToolLoopCheck --replay --replay-model):
#           QFN_REPLAY_RUN (knowledge-sources ple-ab bf16 offline rep1), QFN_REPLAY_ONLY (N5-passport, 7 rounds).
#           The turn's prefill total is the sum of the rounds' prefill_s; wall is the sum of each round's engine call.
# Operating point: 32K, KV Q8_0 and chunk 2048 (the manifest's), MTP on (--draft-block-size 2, the app's), thinking off.
# Each process runs under Scripts/qwen38/guarded.sh with caffeinate -i; COOL_S (20) between processes.
# QFN_BODIES=1|2|12 (12) picks the bodies. A ledger-* label refuses Q38_/TF_ variables and extra arguments.
# Output: bench/qfn/results/<date>-Mac15-6-<label>/ (hostinfo.mbp.txt command.txt env.txt summary-b1.tsv summary-b2.tsv
#   b1-<body>-rN.{server.log,response.json,pre.txt,post.txt,guard.txt}, b2-rN/ (ToolLoopCheck out) b2-rN.{pre,post}.txt)
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; shift
n=6
if [[ $# -ge 1 && $1 != --* ]]; then n=$1; shift; fi
solo=0
[[ $# -ge 1 && $1 == --solo ]] && { solo=1; shift; }
[[ $# -ge 1 && $1 == -- ]] && shift
if (( ! solo )); then echo "only --solo exists yet (the two-machine form comes with Q1)"; exit 1; fi
case $label in
    ledger*)
        if env | grep -qE '^(Q38_|TF_)' || [[ $# -ge 1 ]]; then
            echo "label '$label' is a ledger row: unset every Q38_/TF_ variable and pass no extra args"; exit 1
        fi ;;
esac
model=${QFN_MODEL:-$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2}
bodies=${QFN_BODIES:-12}
replay_run=${QFN_REPLAY_RUN:-$repo/docs/experiments/knowledge-sources/ple-ab/bf16/offline/rep1}
replay_only=${QFN_REPLAY_ONLY:-N5-passport}
questions=$repo/docs/experiments/knowledge-sources/questions-v2-stage2.json
port=${QFN_PORT:-8093}
cool=${COOL_S:-20}
server=$repo/.build/release/TsugumiServer
loop=$repo/.build/release/TsugumiToolLoopCheck
prompts=(${=QFN_PROMPTS:-ja-twomachine ja-m6readme en-readme en-server code-runner})
for b in $server $loop; do [[ -x $b ]] || { echo "no $b (swift build -c release --product ${b:t})"; exit 1; }; done
[[ -f $model/manifest.json ]] || { echo "no model at $model"; exit 1; }
busy() { pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiToolLoopCheck|TsugumiKernelCheck|TsugumiPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm|llama-' || true; }
if [[ -n $(busy) ]]; then busy; echo "another model process is running; refusing to measure"; exit 1; fi

host=$(sysctl -n hw.model | tr ',' '-')
out=$repo/bench/qfn/results/$(date +%F)-$host-$label
mkdir -p $out
sh $repo/bench/hostinfo.sh $server > $out/hostinfo.mbp.txt
{ env | grep -E '^(Q38_|TF_|QFN_|COOL_S)' || true; } > $out/env.txt
server_cmd=($server --model $model --host 127.0.0.1 --port $port --ctx-size 32768 --draft-block-size 2 "$@")
loop_cmd=($loop --model $model --conversations $questions --only $replay_only --network offline --max-rounds 6
          --thinking off --context 32768 --replay $replay_run --replay-model)
{ echo "本文 1: ${server_cmd[*]}"
  echo "本文 1 request: POST /v1/chat/completions {messages: bench/m6/prompts/<body>.json, max_completion_tokens: 1, cache_prompt: false, chat_template_kwargs: {enable_thinking: false}}"
  echo "本文 2: ${loop_cmd[*]} --out <b2-rN>"
  echo "each under: caffeinate -i $repo/Scripts/qwen38/guarded.sh <log> <cmd>"; } > $out/command.txt
snap() { { date '+%H:%M:%S'; memory_pressure -Q | tail -1; vm_stat; iostat -d -I; } > $1; }
swapouts() { awk '/Swapouts/ {gsub(/\./,"",$2); print $2}' $1; }
pageins() { awk '/Pageins/ {gsub(/\./,"",$2); print $2}' $1; }

body1() {  # body round
    local b=$1 r=$2 tag=b1-$1-r$2 gpid spid rc=0
    snap $out/$tag.pre.txt
    GUARD_LOG=$out/$tag.guard.txt caffeinate -i $repo/Scripts/qwen38/guarded.sh $out/$tag.server.log "${server_cmd[@]}" &
    gpid=$!
    local waited=0
    until curl -sf http://127.0.0.1:$port/health > /dev/null 2>&1; do
        sleep 1; waited=$(( waited + 1 ))
        if ! kill -0 $gpid 2>/dev/null || (( waited > 600 )); then rc=1; break; fi
    done
    if (( rc == 0 )); then
        python3 - $repo/bench/m6/prompts/$b.json > $out/$tag.request.json <<'EOF'
import json, sys
print(json.dumps({"messages": json.load(open(sys.argv[1])), "max_completion_tokens": 1, "cache_prompt": False,
                  "chat_template_kwargs": {"enable_thinking": False}}, ensure_ascii=False))
EOF
        curl -sf -m 900 -H 'Content-Type: application/json' --data-binary @$out/$tag.request.json \
            http://127.0.0.1:$port/v1/chat/completions > $out/$tag.response.json || rc=$?
    fi
    # Only the server this run started: its exact command line (guarded.sh's own line also names TsugumiServer).
    spid=$(pgrep -f "^$server --model $model --host 127.0.0.1 --port $port" || true)
    if [[ -n $spid ]]; then kill $spid 2>/dev/null || true; fi
    wait $gpid 2>/dev/null || true
    snap $out/$tag.post.txt
    grep -q '^GUARD: Swapouts' $out/$tag.server.log && rc=9
    python3 - $out/$tag.response.json $b $r $rc \
        $(( $(swapouts $out/$tag.post.txt) - $(swapouts $out/$tag.pre.txt) )) \
        $(( ($(pageins $out/$tag.post.txt) - $(pageins $out/$tag.pre.txt)) * 16384 / 1048576 )) >> $out/summary-b1.tsv <<'EOF'
import json, sys
path, body, rnd, rc, so, pin = sys.argv[1:]
try: t = json.load(open(path)).get("timings", {})
except Exception: t = {}
ms = t.get("prompt_ms")
print("\t".join([body, rnd, str(t.get("prompt_n", "")), str(t.get("cache_n", "")),
                 f"{ms / 1000:.3f}" if ms is not None else "", pin, so, rc]))
EOF
    tail -1 $out/summary-b1.tsv
}

body2() {  # round
    local r=$1 tag=b2-r$1 rc=0
    snap $out/$tag.pre.txt
    mkdir -p $out/$tag
    GUARD_LOG=$out/$tag.guard.txt caffeinate -i $repo/Scripts/qwen38/guarded.sh $out/$tag.log "${loop_cmd[@]}" --out $out/$tag || rc=$?
    snap $out/$tag.post.txt
    grep -q '^GUARD: Swapouts' $out/$tag.log && rc=9
    python3 - $out/$tag $r $rc $(( $(swapouts $out/$tag.post.txt) - $(swapouts $out/$tag.pre.txt) )) \
        $(( ($(pageins $out/$tag.post.txt) - $(pageins $out/$tag.pre.txt)) * 16384 / 1048576 )) >> $out/summary-b2.tsv <<'EOF'
import json, sys
d, rnd, rc, so, pin = sys.argv[1:]
try: rows = [json.loads(l) for l in open(d + "/rounds.jsonl")]
except Exception: rows = []
pre = sum(x.get("prefill_s", 0) for x in rows)
wall = sum((x["ended"] or x["started"]) - x["started"] for x in rows)
# The cached count of a round that ended in the call parser's failure is not in rounds.jsonl (ReplayInferenceClient);
# the session's own line before each round has it: "prompt cache qwen38 <state> agreed=A prompt=P".
import re
seen = [(int(m[0]), int(m[1])) for m in re.findall(r"prompt cache qwen38 \S+ agreed=(\d+) prompt=(\d+)", open(d + ".log").read())]
new = [p - a for a, p in seen]
per = ",".join(f"{x.get('prefill_s', 0):.3f}" for x in rows)
print("\t".join([rnd, str(len(rows)), f"{pre:.3f}", f"{wall:.3f}", ",".join(map(str, new)), per, pin, so, rc]))
EOF
    tail -1 $out/summary-b2.tsv
}

first=1
pause() { (( first )) && first=0 || sleep $cool; }
if [[ $bodies == *1* ]]; then
    printf 'body\tround\tprompt_n\tcache_n\tprefill_s\tpageins_MB\tswapouts\texit\n' > $out/summary-b1.tsv
    for r in $(seq 1 $n); do
        np=${#prompts}; k=$(( (r - 1) % np ))
        for i in $(seq 0 $(( np - 1 ))); do pause; body1 ${prompts[$(( (i + k) % np + 1 ))]} $r; done
    done
fi
if [[ $bodies == *2* ]]; then
    printf 'round\trounds\tprefill_sum_s\twall_sum_s\tnew_tokens\tprefill_s_by_round\tpageins_MB\tswapouts\texit\n' > $out/summary-b2.tsv
    for r in $(seq 1 $n); do pause; body2 $r; done
fi
echo "evidence: ${out#$repo/}"
