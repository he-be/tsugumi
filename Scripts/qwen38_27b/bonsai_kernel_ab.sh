#!/bin/bash
# mul_mv_ext の有無 × (投機なし / MTP n_max 1 / n_max 2) を c=8192 で回す。
# ブロック間は 20 秒のクールダウン。出力先は OUT= で変えられる。
# 実行は caffeinate -i ./Scripts/qwen38_27b/bonsai_kernel_ab.sh で。実行中にこのファイルを編集しない。
set -u
S="${OUT:-$PWD/scratch/bonsai27b/kernel-ab}"
mkdir -p "$S"
H="$(cd "$(dirname "$0")" && pwd)"
BIN=/Users/mh/LLM/prism-llamacpp/src-b10709/build/bin/llama-server
M=/Users/mh/LLM/Ternary-Bonsai-2-27B-MTP/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf
PORT=8099
CTX="${CTX:-8192}"

if [ -n "$(lsof -ti :$PORT 2>/dev/null)" ]; then echo "port $PORT busy"; exit 1; fi

block () {  # $1=tag  $2=spec 引数  $3=ext (0/1)
  echo "=== $1 ==="
  if [ "$3" = 1 ]; then export GGML_METAL_PQ2_0_EXT_ENABLE=1; else unset GGML_METAL_PQ2_0_EXT_ENABLE || true; fi
  "$BIN" -m "$M" -ngl 99 -fa on -c $CTX -np 1 $2 \
      --host 127.0.0.1 --port $PORT > "$S/srv-$1.log" 2>&1 &
  SRV=$!
  until [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; do
    sleep 3
    kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 "$S/srv-$1.log"; return 1; }
  done
  ps -o rss= -p $SRV | awk '{printf "  RSS(load) %.2f GB\n", $1/1048576}'
  python3 "$H/bonsai_mtp_ab.py" $PORT "$1" "$S/ab-$1.json"
  ps -o rss= -p $SRV | awk '{printf "  RSS(end)  %.2f GB\n", $1/1048576}'
  kill $SRV
  for _ in $(seq 1 20); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
  kill -9 $SRV 2>/dev/null
  wait $SRV 2>/dev/null
  sleep 20
}

# 引数を渡すと "tag:ext:spec 引数" の並びをその順で回す。無ければ既定の 6 ブロック。
if [ $# -gt 0 ]; then
  for a in "$@"; do
    tag="${a%%:*}"; rest="${a#*:}"; ext="${rest%%:*}"; spec="${rest#*:}"
    block "$tag" "$spec" "$ext"
  done
else
  block ext1-none "--spec-type none" 1
  block ext1-n1 "--spec-type draft-mtp --spec-draft-n-max 1" 1
  block ext1-n2 "--spec-type draft-mtp --spec-draft-n-max 2" 1
  block ext0-none "--spec-type none" 0
  block ext0-n1 "--spec-type draft-mtp --spec-draft-n-max 1" 0
  block ext0-n2 "--spec-type draft-mtp --spec-draft-n-max 2" 0
fi
echo ALLDONE
