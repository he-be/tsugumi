#!/bin/zsh
set -u
CLI=./.build/release/TurboFieldfareCLI
AFF=scratch/gemma4-qat.gturbo
SYM=scratch/e2e-sym-check.gturbo
OUT=$1
COOL=20
typeset -a P
P[1]="The capital of France is"
P[2]="Explain, step by step, how a mixture-of-experts transformer routes a token through its experts."
P[3]="Write a Python function that merges two sorted lists and explain how it works."

run() { # model_label model_path prompt_idx round
  local label=$1 path=$2 idx=$3 round=$4
  local f=$OUT/gen_p${idx}_${label}_r${round}.txt
  $CLI --model $path --prompt "${P[$idx]}" --temperature 0 --max-new 256 \
       --expert-cache-slots 32 > $f 2>&1
  echo "exit=$? $f"
  /bin/sleep $COOL
}

echo "### prompts"
for i in 1 2 3; do echo "prompt $i: ${P[$i]}"; done

# warm both models once (pays layerVerify) then A/B
for r in 1 2 3; do
  run affine $AFF 1 $r
  run sym    $SYM 1 $r
done
for i in 2 3; do
  run affine $AFF $i 1
  run sym    $SYM $i 1
done

# derivation check: affine data + sym library
TF_FORCE_AFFINE_SCHEME=sym $CLI --model $AFF --prompt "${P[1]}" --temperature 0 \
  --max-new 256 --expert-cache-slots 32 > $OUT/gen_p1_forcedsym_r1.txt 2>&1
echo "exit=$? forcedsym"
