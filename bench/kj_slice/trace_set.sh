#!/bin/bash
# trace_set.sh <set> <port>   — kj_payloads<set>.jsonl をフルモデルで流し、ルーターの確率を記録して stats_<set>.npz にする (GPU は環境変数)
set=$1; port=$2
cd /root/diet
Q4=/root/models/gemma-4-26b-a4b-qat-2/Huihui-gemma-4-26B-A4B-it-qat-abliterated-lattice-Q4_0.gguf
rm -f trace_$set.bin
./serve.sh $port $Q4 /root/diet/trace_$set.bin > serve-trace-$set.log 2>&1 &
sp=$!
for i in $(seq 150); do curl -s -m 2 http://127.0.0.1:$port/health | grep -q ok && break; sleep 2; done
/root/xolotl/.venv/bin/python run_arm.py http://127.0.0.1:$port kj_payloads$set.jsonl arm_trace_$set.jsonl --trace /root/diet/trace_$set.bin
kill $sp; wait $sp
/root/xolotl/.venv/bin/python trace_stats.py trace_$set.bin arm_trace_$set.jsonl stats_$set.npz && rm -f trace_$set.bin
echo TRACE_SET_DONE $set
