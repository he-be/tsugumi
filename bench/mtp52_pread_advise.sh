#!/bin/bash
# 52d — 対照: 同じ F_RDADVISE を **pread の腕** に出しても速くなるか。
# 速くなるなら 52c の取り分の一部は D ではなく advise のものである (40 §4-20)。
# 両腕とも pread なのでページキャッシュの汚染は構造的に無い。
set -u
COOL=${COOL:-2}
mkdir -p bench/mtp52
OUT=bench/mtp52/pread_advise_256_math_slots32.log
: > "$OUT"
run() {
  local adv="$1" blk="$2" pos="$3" label
  label=$([ "$adv" = 1 ] && echo pread+adv || echo pread)
  TF_EXPERT_MMAP=0 TF_EXPERT_MMAP_ADVISE="$adv" \
  ./.build/release/TsugumiCLI \
    --model scratch/gemma4-qat-sym.moepack --messages-file bench/math.json \
    --temperature 0 --max-new 256 --expert-cache-slots 32 \
    --verification trusted-install > /tmp/tf_pa.txt 2>&1
  { echo "### block=$blk pos=$pos arm=$label"
    grep -E "expert prefill|stop=maxTokens|load=" /tmp/tf_pa.txt; } >> "$OUT"
  echo "block=$blk pos=$pos arm=$label" \
    "$(grep -oE 'prefill=[0-9.]+s ttft=[0-9.]+s peak=[0-9.]+GB' /tmp/tf_pa.txt)" \
    "$(grep -oE 'tok/s=[0-9.]+' /tmp/tf_pa.txt)" \
    "$(grep -oE 'expert prefill hit=[0-9.]+% [0-9]+/[0-9]+ io=[0-9.]+s' /tmp/tf_pa.txt)" \
    "$(grep -oE 'decode hit=[0-9.]+% [0-9]+/[0-9]+ io=[0-9.]+s' /tmp/tf_pa.txt)"
}
blk=0
for adv in 1 0 0 1; do
  blk=$((blk+1)); for pos in 1 2 3 4 5 6; do run "$adv" "$blk" "$pos"; sleep "$COOL"; done
done
