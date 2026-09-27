#!/bin/sh
# The app's tool loop (TsugumiToolLoopCheck) against ishizuki-serve on the M6, reached over the
# ssh forward (docs/qwen38-27b/16-M6-ISHIZUKI-AGENT-PLAN.md S3-S4). The arguments are those of
# 05 §6 (offline) and 11 §6 (online); only the endpoint differs. Nothing is loaded on the MBP.
#   bench/m6/ishizuki_toolloop_run.sh offline|online OUT [extra TsugumiToolLoopCheck args]
# Every 10 s it records the M6's Swapouts (vm_stat, 16 KiB pages, relative to the start) and the
# server's footprint (top MEM) in OUT/watch.log, and stops the check at +512 MiB in 60 s or
# +1 GiB in total.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
mode=$1; OUT=$2; shift 2
EP=${ISHIZUKI_ENDPOINT:-http://127.0.0.1:8128}
HOST=${M6_HOST:-m6-lan}
case $mode in
offline) ARGS="--conversations $R/docs/experiments/knowledge-sources/questions-v2-stage2.json --network offline" ;;
online) ARGS="--conversations $R/Scripts/qwen38/tool_loop_conversations.json --network online --web-store $R/scratch/bonsai27b/web --pin-search --search-budget 0" ;;
*) echo "usage: $0 offline|online OUT [args]"; exit 2 ;;
esac
mkdir -p "$OUT"
curl -s -m 5 "$EP/health" | grep -q ok || { echo "no /health at $EP (ssh -f -N -L 8128:127.0.0.1:8128 $HOST)"; exit 1; }
so() { ssh "$HOST" "vm_stat | awk '/Swapouts/ {gsub(\"\\\\.\",\"\",\$2); print \$2}'"; }
mem() { ssh "$HOST" 'top -l 1 -stats command,mem | awk "/ishizuki-serve/ {print \$2}"'; }
ssh "$HOST" 'tail -1 ~/LLM/ref/ishizuki-logs/serve.log' > "$OUT/server-start.txt" 2>/dev/null
S0=$(so); echo "start swapouts $S0" > "$OUT/watch.log"
# shellcheck disable=SC2086
caffeinate -i "$R/.build/release/TsugumiToolLoopCheck" --out "$OUT" --model ~/LLM/Ternary-Bonsai-2-27B-MTP \
  $ARGS --max-rounds 6 --thinking off --context 32768 --repeats 1 \
  --endpoint "$EP" --remote-model ternary-bonsai-2-27b --remote-direct "$@" > "$OUT/check.log" 2>&1 &
CHK=$!
t=0; hist=""
while kill -0 $CHK 2>/dev/null; do
  sleep 10; t=$((t + 10)); N=$(so); M=$(mem)
  echo "t=$t swapouts +$((N - S0)) mem $M" >> "$OUT/watch.log"
  hist="$hist $N"; old=$(echo $hist | awk '{ if (NF > 6) print $(NF-6); else print $1 }')
  if [ $((N - S0)) -gt 65536 ] || [ $((N - old)) -gt 32768 ]; then
    echo "guard stop at t=$t" >> "$OUT/watch.log"; pkill -P $CHK; kill $CHK 2>/dev/null
  fi
done
wait $CHK; rc=$?; echo "exit=$rc" >> "$OUT/watch.log"
grep -E '^[A-Za-z0-9-]+#[0-9]+ turn [0-9]+: ' "$OUT/check.log"; tail -1 "$OUT/check.log"
exit $rc
