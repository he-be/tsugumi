#!/bin/bash
# 52b — 腕を交互にしないで測る。40 §4-15 (プローブが自分で機械を汚す) の 2 度目。
# ABBA では mmap の run が必ず pread の run の直後に来る。pread は私有スロットに
# 3.2 GB を確保して prompt の 6.4 GB をページキャッシュから追い出すので、
# mmap の prefill は「冷えたページを深度 1 でフォールトする」条件で測られていた。
# ブロック ABBA (mmap*6 / pread*6 / pread*6 / mmap*6) で汚染を外す。
# クールダウンは 2s (ユーザー指定)。ドリフト検定のため run 順を全部残す。
set -u
COOL=${COOL:-2}
mkdir -p bench/mtp52
OUT=bench/mtp52/single_arm_256_math_slots32.log
: > "$OUT"
run() {  # $1=0/1  $2=block  $3=pos
  local on="$1" blk="$2" pos="$3" label
  label=$([ "$on" = 1 ] && echo mmap || echo pread)
  # ADVISE=0: このログは advise が製品に入る前の条件で取られている
  # (既定は 2026-08-20 に mmap の腕だけ on になった。52 §8)。
  TF_EXPERT_MMAP="$on" TF_EXPERT_MMAP_ADVISE=0 ./.build/release/TsugumiCLI \
    --model scratch/gemma4-qat-sym.moepack --messages-file bench/math.json \
    --temperature 0 --max-new 256 --expert-cache-slots 32 \
    --verification trusted-install > /tmp/tf_sa.txt 2>&1
  {
    echo "### block=$blk pos=$pos arm=$label"
    grep -E "expert prefill|stop=maxTokens|load=|expert mmap" /tmp/tf_sa.txt
  } >> "$OUT"
  echo "block=$blk pos=$pos arm=$label" \
    "$(grep -oE 'prefill=[0-9.]+s ttft=[0-9.]+s peak=[0-9.]+GB' /tmp/tf_sa.txt)" \
    "$(grep -oE 'tok/s=[0-9.]+' /tmp/tf_sa.txt)" \
    "$(grep -oE 'expert prefill hit=[0-9.]+% [0-9]+/[0-9]+ io=[0-9.]+s' /tmp/tf_sa.txt)" \
    "$(grep -oE 'decode hit=[0-9.]+% [0-9]+/[0-9]+ io=[0-9.]+s' /tmp/tf_sa.txt)"
}
blk=0
for on in 1 0 0 1; do
  blk=$((blk+1))
  for pos in 1 2 3 4 5 6; do
    run "$on" "$blk" "$pos"
    sleep "$COOL"
  done
done
