#!/bin/zsh
cd ~/dev/tsugumi
R=$HOME/LLM/ref
U=$R/unsloth-llama-b12588/llama-b12588-mix-c0c3de8
export RUNTIME_NOTE="llama.cpp 61dbf49 mainline / lattice-keep_80.gguf / mmproj-lattice-Q8_0 / same args as 2026-11-15 k80"
LLAMA_ARG_CHAT_TEMPLATE_KWARGS="{\"enable_thinking\":false}" PORT=8082 READY="listening on" bench/m6/kj_replay_probe.sh kj-llamacpp-lattice-k80-vsink 2 -- \
  $R/llama.cpp-latest/build/bin/llama-server -m $R/lattice-keep_80.gguf --mmproj $R/mmproj-lattice-Q8_0.gguf \
  --port 8082 --host 127.0.0.1 -ngl 99 -fit off -fa on -b 1024 -ub 512 -np 1 -cram 0 -c 4096 --reasoning off \
  --image-max-tokens 280 --temp 1.0 --top-k 64 --min-p 0
sleep 10
export RUNTIME_NOTE="unsloth llama.cpp b12588-mix-c0c3de8 / Inkling-2-Mini-UD-IQ3_XXS.gguf / mmproj-F16 / assistant prefill {"
PAYLOADS=$HOME/diet-probe/kj_payloads_ink_prefill.jsonl PORT=8083 READY="listening on" bench/m6/kj_replay_probe.sh kj-inkling2-iq3xxs-prefill 2 -- \
  $U/llama-server -m $R/inkling2/Inkling-2-Mini-UD-IQ3_XXS.gguf --mmproj $R/inkling2/mmproj-F16.gguf \
  --port 8083 --host 127.0.0.1 -ngl 99 -b 1024 -ub 512 -np 1 -cram 0 -c 4096 --jinja \
  --image-max-tokens 280 --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0
echo ALL_DONE
