#!/bin/bash
# Bonsai 2 27B をアプリと同じ経路 (KindRoutingInferenceClient → 子サーバの llama-server) で流す。
# サーバは TsugumiToolLoopCheck が自分で立てて自分で止める (docs/qwen38-27b/10 の S4〜S6)。
#   OUT=…   出力先 (既定 scratch/bonsai27b/runs/child-set1)
#   ONLY=…  会話 id をカンマ区切りで絞る
#   EXTRA=… TsugumiToolLoopCheck への追加引数
#   SERVER_ARGS=… 子サーバの llama-server に足す引数 (例 "--cache-ram 0")
# Swapouts が 60 秒で +512 MiB か合計 +1 GiB になったら、検査を子孫ごと止める。
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-$R/scratch/bonsai27b/runs/child-set1}"
mkdir -p "$OUT"

swapouts () { vm_stat | awk '/Swapouts/ {gsub("\\.","",$2); print $2}'; }
killtree () { for c in $(pgrep -P "$1"); do killtree "$c"; done; kill "$1" 2>/dev/null; }
cleanup () { [ -n "${CHK:-}" ] && killtree "$CHK"; }
trap cleanup EXIT

S0=$(swapouts); echo "swapouts(start) $S0" | tee "$OUT/watch.log"
caffeinate -i "$R/.build/release/TsugumiToolLoopCheck" --out "$OUT" --model ~/LLM/Ternary-Bonsai-2-27B-MTP \
  --conversations "$R/Scripts/qwen38/tool_loop_conversations.json" ${ONLY:+--only $ONLY} \
  --network online --max-rounds 6 --thinking off --context 32768 --repeats 1 \
  --web-store "$R/scratch/bonsai27b/web" --pin-search --search-budget 0 ${EXTRA:-} \
  ${SERVER_ARGS:+--server-args "$SERVER_ARGS"} > "$OUT/check.log" 2>&1 &
CHK=$!

# 5 秒刻みで残し、判定は直近 60 秒 (12 標本) の増分と合計で行う
HIST=()
while kill -0 $CHK 2>/dev/null; do
  sleep 5
  NOW=$(swapouts)
  HIST+=("$NOW"); [ ${#HIST[@]} -gt 12 ] && HIST=("${HIST[@]:1}")
  WIN=$((NOW-HIST[0]))
  RSS=$(ps -axo rss,comm | awk '/llama-server$/ {printf "%.2f", $1/1048576}')
  FREE=$(memory_pressure | awk '/free percentage/ {print $NF}')
  ROUND=$(grep -c 'round [0-9]* start' "$OUT/check.log" 2>/dev/null)
  echo "$(date +%H:%M:%S) swapouts +$((NOW-S0)) pages (60s +$WIN) rss ${RSS} GB free ${FREE} rounds $ROUND" >> "$OUT/watch.log"
  # 誰がメモリを持っているかは、増えているときだけ残す (止まったときに原因を推測で書かないため)
  if [ $WIN -gt 0 ]; then
    ps -axo rss,comm | sort -rn | head -5 | awk '{n=split($2,a,"/"); printf "    %.2f GB %s\n", $1/1048576, a[n]}' >> "$OUT/watch.log"
  fi
  # 16 KiB ページ: 512 MiB = 32768、1 GiB = 65536
  if [ $WIN -gt 32768 ] || [ $((NOW-S0)) -gt 65536 ]; then
    echo "SWAP LIMIT: stopping" | tee -a "$OUT/watch.log"; cleanup; exit 2
  fi
done
wait $CHK; RC=$?
echo "check exit $RC, swapouts total +$(( $(swapouts) - S0 )) pages" | tee -a "$OUT/watch.log"
LEFT=$(pgrep -f 'build/bin/llama-server' | tr '\n' ' ')
echo "llama-server left after exit: ${LEFT:-none}" | tee -a "$OUT/watch.log"
exit $RC
