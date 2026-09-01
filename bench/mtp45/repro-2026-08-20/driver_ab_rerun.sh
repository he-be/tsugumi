#!/bin/zsh
set -u
CLI=./.build/release/TsugumiCLI
AFF=scratch/gemma4-qat.moepack
SYM=scratch/gemma4-qat-sym.moepack
OUT=$1
P="The capital of France is"
for r in 1 2 3; do
  for m in affine:$AFF sym:$SYM; do
    lab=${m%%:*}; path=${m#*:}
    date +%H:%M:%S >> $OUT/timeline.txt
    pgrep -fl 'TsugumiCLI' | grep -v "$$" >> $OUT/timeline.txt
    $CLI --model $path --prompt "$P" --temperature 0 --max-new 256 \
         --expert-cache-slots 32 > $OUT/gen_p1_${lab}_r${r}.txt 2>&1
    echo "$lab r$r exit=$?"
    /bin/sleep 20
  done
done
