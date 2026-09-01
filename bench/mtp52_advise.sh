#!/bin/bash
# 52c — E2: mmap の腕にも F_RDADVISE を出す (TF_EXPERT_MMAP_ADVISE=1)。
# **両腕とも mmap** なので pread による ページキャッシュ汚染は最初から無い。
# ブロック ABBA (adv/base/base/adv)、クールダウン 2s、run 順を全部残す。
set -u
COOL=${COOL:-2}
mkdir -p bench/mtp52
OUT=bench/mtp52/advise_256_math_slots32.log
: > "$OUT"
run() {
  local adv="$1" blk="$2" pos="$3" label
  label=$([ "$adv" = 1 ] && echo adv || echo base)
  TF_EXPERT_MMAP=1 TF_EXPERT_MMAP_ADVISE="$adv" \
  ./.build/release/TsugumiCLI \
    --model scratch/gemma4-qat-sym.moepack --messages-file bench/math.json \
    --temperature 0 --max-new 256 --expert-cache-slots 32 \
    --verification trusted-install > /tmp/tf_adv.txt 2>&1
  { echo "### block=$blk pos=$pos arm=$label"
    grep -E "expert prefill|stop=maxTokens|load=|expert mmap" /tmp/tf_adv.txt; } >> "$OUT"
  echo "block=$blk pos=$pos arm=$label" \
    "$(grep -oE 'prefill=[0-9.]+s ttft=[0-9.]+s peak=[0-9.]+GB' /tmp/tf_adv.txt)" \
    "$(grep -oE 'tok/s=[0-9.]+' /tmp/tf_adv.txt)" \
    "$(grep -oE 'expert prefill hit=[0-9.]+% [0-9]+/[0-9]+ io=[0-9.]+s' /tmp/tf_adv.txt)" \
    "$(grep -oE 'decode hit=[0-9.]+% [0-9]+/[0-9]+ io=[0-9.]+s' /tmp/tf_adv.txt)" \
    "sha=$(grep -A400 'assistant' /tmp/tf_adv.txt | shasum | cut -c1-12)"
}
blk=0
for adv in 1 0 0 1; do
  blk=$((blk+1)); for pos in 1 2 3 4 5 6; do run "$adv" "$blk" "$pos"; sleep "$COOL"; done
done
