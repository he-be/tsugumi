#!/bin/zsh
# ログインして Chrome と Web カメラ (Iriun) を起こした状態で、80 個版を 47 と同じ引数で再生する。引数: ラベルの末尾
cd ~/dev/tsugumi
R=$HOME/LLM/ref
export LLAMA_ARG_CHAT_TEMPLATE_KWARGS="{\"enable_thinking\":false}"
m=$R/lattice-keep_80.gguf
export RUNTIME_NOTE="llama.cpp 61dbf49 / lattice-keep_80.gguf $(stat -f %z $m) bytes / mmproj-lattice-Q8_0.gguf 806408032 bytes / GUI にログイン (console $(stat -f %Su /dev/console))、Chrome と IriunWebcam が起動中 / 起動から $(uptime | sed "s/.*up //;s/, [0-9]* user.*//")"
ps -axo rss,comm | sort -rn | head -25 > /tmp/gui_ps_pre.txt
PORT=8082 READY="listening on" bench/m6/kj_replay_probe.sh kj-llamacpp-lattice-k80-$1 2 -- \
  $R/llama.cpp-latest/build/bin/llama-server -m $m --mmproj $R/mmproj-lattice-Q8_0.gguf \
  --port 8082 --host 127.0.0.1 -ngl 99 -fit off -fa on -b 1024 -ub 512 -np 1 -cram 0 -c 4096 --reasoning off \
  --image-max-tokens 280 --temp 1.0 --top-k 64 --min-p 0
echo "rc=$?"
d=$(ls -d bench/m6/results/*kj-llamacpp-lattice-k80-$1 | tail -1); cp /tmp/gui_ps_pre.txt $d/ps_pre.txt
echo GUI_RUN_DONE
