#!/bin/bash
# serve.sh <port> <gguf> [trace-file]   — kajimachine の再生と同じ引数で llama-server を起こす
# 環境変数: GPU (CUDA_VISIBLE_DEVICES に渡す値。既定は llm-server の RTX 3090)、MMPROJ、BIN、CTX
port=$1; model=$2; trace=${3:-}
export CUDA_VISIBLE_DEVICES=${GPU:-GPU-629ffdef-f38e-d212-7f89-0f74f1ee94e5}
export LLAMA_ARG_CHAT_TEMPLATE_KWARGS="{\"enable_thinking\":false}"
bin=${BIN:-/root/llama.cpp/build/bin/llama-server}
mmproj=${MMPROJ:-/root/models/gemma-4-26b-a4b-qat-2/mmproj-model-Q8_0.gguf}
if [ -n "$trace" ]; then export LLAMA_MOE_TRACE=$trace; export LD_LIBRARY_PATH=/root/diet/bin-trace; bin=/root/diet/bin-trace/llama-server; fi
exec $bin -m $model --mmproj $mmproj --port $port --host 127.0.0.1 \
  -ngl 99 -fit off -fa on -b 1024 -ub 512 -np 1 -cram 0 -c ${CTX:-8192} --reasoning off --image-max-tokens 280 --temp 1.0 --top-k 64 --min-p 0 -t 8
