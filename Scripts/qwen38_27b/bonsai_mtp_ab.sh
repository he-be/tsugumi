#!/bin/bash
# Bonsai 2 27B の MTP A/B を A,B,B,A の順で回す。ブロック間は 20 秒のクールダウン。
# 出力先は OUT= で変えられる (既定 scratch/bonsai27b/mtp-ab)。
set -u
S="${OUT:-$PWD/scratch/bonsai27b/mtp-ab}"
mkdir -p "$S"
H="$(cd "$(dirname "$0")" && pwd)"
BIN=/Users/mh/LLM/prism-llamacpp/src-b10709/build/bin/llama-server
M=/Users/mh/LLM/Ternary-Bonsai-2-27B-MTP/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf
PORT=8099

block () {  # $1=tag  $2=spec 引数
  echo "=== $1 ==="
  caffeinate -i "$BIN" -m "$M" -ngl 99 -fa on -c 32768 -np 1 $2 \
      --host 127.0.0.1 --port $PORT > "$S/srv-$1.log" 2>&1 &
  SRV=$!
  until curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1; do
    sleep 3
    kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 "$S/srv-$1.log"; return 1; }
  done
  ps -o rss= -p $SRV | awk '{printf "  RSS(load) %.2f GB\n", $1/1048576}'
  python3 "$H/bonsai_mtp_ab.py" $PORT "$1" "$S/ab-$1.json"
  ps -o rss= -p $SRV | awk '{printf "  RSS(end)  %.2f GB\n", $1/1048576}'
  kill $SRV; wait $SRV 2>/dev/null
  sleep 20
}

block none-1 "--spec-type none"
block mtp-1  "--spec-type draft-mtp --spec-draft-n-max 2"
block mtp-2  "--spec-type draft-mtp --spec-draft-n-max 2"
block none-2 "--spec-type none"
echo ALLDONE
