#!/bin/bash
# Bonsai 2 27B (PQ2_0 + MTP) を llama-server で立て、TsugumiToolLoopCheck の Online を固定検索で最後まで流す。
#   OUT=…   出力先 (既定 scratch/bonsai27b/runs/online-mtp-set1)
#   SPEC=…  投機の引数 (既定 n_max 1。投機なしは SPEC="--spec-type none")
#   ONLY=…  会話 id をカンマ区切りで絞る
# Swapouts が 60 秒で +512 MiB か合計 +1 GiB になったら、サーバも検査も子孫ごと止める。
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-$R/scratch/bonsai27b/runs/online-mtp-set1}"
SPEC="${SPEC:---spec-type draft-mtp --spec-draft-n-max 1}"
BIN=/Users/mh/LLM/prism-llamacpp/src-b10709/build/bin/llama-server
M=/Users/mh/LLM/Ternary-Bonsai-2-27B-MTP/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf
PORT=8080; RELAY=8081
mkdir -p "$OUT"

swapouts () { vm_stat | awk '/Swapouts/ {gsub("\\.","",$2); print $2}'; }
killtree () { for c in $(pgrep -P "$1"); do killtree "$c"; done; kill "$1" 2>/dev/null; }
cleanup () { for p in ${CHK:-} ${RLY:-} ${SRV:-}; do killtree "$p"; done; }
trap cleanup EXIT

caffeinate -i "$BIN" -m "$M" -ngl 99 -fa on -c 32768 -np 1 --jinja $SPEC \
    --host 127.0.0.1 --port $PORT > "$OUT/server.log" 2>&1 &
SRV=$!
until curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok; do
  sleep 3
  kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 "$OUT/server.log"; exit 1; }
done
python3 "$R/Scripts/qwen38_27b/upstream_relay.py" $RELAY $PORT > "$OUT/relay.log" 2>&1 &
RLY=$!
sleep 1
S0=$(swapouts); echo "swapouts(start) $S0" | tee "$OUT/watch.log"

"$R/.build/release/TsugumiToolLoopCheck" --out "$OUT" --model ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2 \
  --conversations "$R/Scripts/qwen38/tool_loop_conversations.json" ${ONLY:+--only $ONLY} \
  --network online --max-rounds 6 --thinking off --context 32768 --repeats 1 \
  --web-store "$R/scratch/bonsai27b/web" --pin-search --search-budget 0 \
  --endpoint http://127.0.0.1:$RELAY --remote-model bonsai > "$OUT/check.log" 2>&1 &
CHK=$!

PREV=$S0
while kill -0 $CHK 2>/dev/null; do
  sleep 60
  NOW=$(swapouts)
  RSS=$(ps -axo rss,comm | awk '/llama-server$/ {printf "%.2f", $1/1048576}')
  FREE=$(memory_pressure | awk '/free percentage/ {print $NF}')
  echo "$(date +%H:%M:%S) swapouts +$((NOW-S0)) pages (60s +$((NOW-PREV))) rss ${RSS} GB free ${FREE}" >> "$OUT/watch.log"
  # 誰がメモリを持っているかを残す (止まったときに原因を推測で書かないため)
  ps -axo rss,comm | sort -rn | head -6 | awk '{n=split($2,a,"/"); printf "    %.2f GB %s\n", $1/1048576, a[n]}' >> "$OUT/watch.log"
  # 16 KiB ページ: 512 MiB = 32768、1 GiB = 65536
  if [ $((NOW-PREV)) -gt 32768 ] || [ $((NOW-S0)) -gt 65536 ]; then
    echo "SWAP LIMIT: stopping" | tee -a "$OUT/watch.log"; cleanup; exit 2
  fi
  PREV=$NOW
done
wait $CHK; RC=$?
echo "check exit $RC, swapouts total +$(( $(swapouts) - S0 )) pages" | tee -a "$OUT/watch.log"
exit $RC
