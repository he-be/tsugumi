#!/bin/zsh
# fetch_full.sh: ds4 Q2 GGUF の per_layer_token_embd.weight (BF16、102,400,491,520 B = ファイル末尾まで) を 1 本の接続で
# ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/ple-bf16/ に引き、ヘッダ (head.bin) + 本体 + メタテンソル (tail.bin) の GGUF にする。
# 中断したら再実行でファイルサイズから続きを取る。
D=~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/ple-bf16
U=https://huggingface.co/antirez/qwen3.8-flash-next-gguf/resolve/main/Qwen3.8-Flash-Next-Q2.gguf
R0=44806635520; BODY=102400491520; HEAD=3712
P=$D/Qwen3.8-Flash-Next-PLE-BF16.gguf.part
[[ -f $P ]] || cp $D/head.bin $P
for try in {1..50}; do
  have=$(( $(stat -f %z $P) - HEAD ))
  (( have >= BODY )) && break
  echo "$(date +%T) try $try from $have" 
  curl -sSL --fail --retry 3 -r $((R0 + have))-$((R0 + BODY - 1)) $U >> $P
  echo "$(date +%T) curl exit $?"
  sleep 5
done
have=$(( $(stat -f %z $P) - HEAD ))
if (( have == BODY )); then cat $D/tail.bin >> $P && mv $P $D/Qwen3.8-Flash-Next-PLE-BF16.gguf && echo DONE; else echo "INCOMPLETE $have"; fi
