#!/bin/bash
cd ~/diet
until grep -q MITIG_JOB_DONE mitig.log; do sleep 5; done
GPU=0 CTX=4096 BIN=$HOME/LLM/llama.cpp-diet/build/bin/llama-server MMPROJ=$HOME/diet/mmproj-model-Q8_0.gguf ./serve.sh 9310 $HOME/diet/lattice-keep_80.gguf > serve-maxlen.log 2>&1 &
sp=$!
for i in $(seq 150); do curl -s -m 2 http://127.0.0.1:9310/health | grep -q ok && break; sleep 2; done
python3 run_arm.py http://127.0.0.1:9310 kj_payloadsA_maxlen.jsonl item_maxlen100.jsonl --only item --rounds 5
kill $sp; wait $sp
echo MAXLEN_DONE
