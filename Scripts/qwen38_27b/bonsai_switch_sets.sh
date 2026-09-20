#!/bin/bash
# 会話の切り替え (ja-news → en-swift) を、子サーバの引数を 1 つずつ変えて順に流す (docs/qwen38-27b/10 §3-1 2)。
# 各実行は en-swift の 1 ターン目が終わったら止める (切り替えの後 2 分以上が入る)。間は 20 秒空ける。
#   SET=…      出力先の接尾辞 (既定 set2)
#   VARIANTS=… 空白区切り (既定 "default cacheram0 ctxcp0")
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"
SET="${SET:-set2}"
for V in ${VARIANTS:-default cacheram0 ctxcp0}; do
  case $V in
    default) ARGS="" ;;
    cacheram0) ARGS="--cache-ram 0" ;;
    ctxcp0) ARGS="--ctx-checkpoints 0" ;;
    *) echo "unknown variant $V"; exit 2 ;;
  esac
  OUT="$R/scratch/bonsai27b/runs/s6-switch-$V-$SET"
  rm -rf "$OUT"; mkdir -p "$OUT"
  OUT="$OUT" ONLY=ja-news,en-swift SERVER_ARGS="$ARGS" "$R/Scripts/qwen38_27b/bonsai_child_server_run.sh" > "$OUT/run.log" 2>&1 &
  RUN=$!
  while kill -0 $RUN 2>/dev/null; do
    grep -q 'en-swift#1 turn 1: ' "$OUT/check.log" 2>/dev/null && break
    sleep 2
  done
  if kill -0 $RUN 2>/dev/null; then kill $RUN; wait $RUN 2>/dev/null; echo "$V: stopped after en-swift turn 1"
  else wait $RUN; echo "$V: run ended by itself with exit $?"; fi
  grep -E 'SWAP LIMIT' "$OUT/watch.log" | head -1
  until ! pgrep -f 'build/bin/llama-server' > /dev/null; do sleep 1; done
  sleep 20
done
