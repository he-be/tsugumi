#!/bin/bash
# serve.sh <port> <gguf> [trace-file]   — 3090 だけ、M6 の再生と同じ引数 (kajimachine 用)
port=$1; model=$2; trace=${3:-}
export CUDA_VISIBLE_DEVICES=GPU-629ffdef-f38e-d212-7f89-0f74f1ee94e5
export LLAMA_ARG_CHAT_TEMPLATE_KWARGS="{\"enable_thinking\":false}"
bin=/root/llama.cpp/build/bin/llama-server
if [ -n "$trace" ]; then export LLAMA_MOE_TRACE=$trace; export LD_LIBRARY_PATH=/root/diet/bin-trace; bin=/root/diet/bin-trace/llama-server; fi
exec $bin -m $model --mmproj /root/models/gemma-4-26b-a4b-qat-2/mmproj-model-Q8_0.gguf --port $port --host 127.0.0.1 \
  -ngl 99 -fit off -fa on -b 1024 -ub 512 -np 1 -cram 0 -c 8192 --reasoning off --image-max-tokens 280 --temp 1.0 --top-k 64 --min-p 0 -t 8
