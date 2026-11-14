#!/bin/bash
cd /root/diet
Q4=/root/models/gemma-4-26b-a4b-qat-2/Huihui-gemma-4-26B-A4B-it-qat-abliterated-lattice-Q4_0.gguf
rm -f trace_full.bin
./serve.sh 9301 $Q4 /root/diet/trace_full.bin > serve-9301.log 2>&1 &
sp=$!
for i in $(seq 90); do curl -s -m 2 http://127.0.0.1:9301/health | grep -q ok && break; sleep 2; done
/root/xolotl/.venv/bin/python run_arm.py http://127.0.0.1:9301 kj_payloads96.jsonl arm_trace_full.jsonl --trace /root/diet/trace_full.bin
kill $sp; wait $sp
/root/xolotl/.venv/bin/python trace_stats.py trace_full.bin arm_trace_full.jsonl stats_full.npz
echo TRACE_JOB_DONE
