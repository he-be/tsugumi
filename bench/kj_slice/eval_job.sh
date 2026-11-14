#!/bin/bash
# eval_job.sh <arm>=<gguf> ...   — 各腕: 確かめる用の 48 枚を greedy と通常サンプリングで 1 回ずつ
cd /root/diet
PY=/root/xolotl/.venv/bin/python
for spec in "$@"; do
  arm=${spec%%=*}; gguf=${spec#*=}
  ./serve.sh 9302 $gguf > serve-$arm.log 2>&1 &
  sp=$!
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:9302/health | grep -q ok && break; sleep 2; done
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -1 > vram-$arm.txt
  $PY run_arm.py http://127.0.0.1:9302 kj_payloads96.jsonl ans_${arm}_greedy.jsonl --greedy --plays every2:1
  $PY run_arm.py http://127.0.0.1:9302 kj_payloads96.jsonl ans_${arm}_sample.jsonl --plays every2:1
  kill $sp; wait $sp
done
echo EVAL_JOB_DONE
