#!/bin/bash
# 52 — D (TF_EXPERT_MMAP=1) の prefill がスロット数に依存するかを振る。
# 32 スロットは 51 (n=6) が既にあるので 16 と 64 だけ足す。
# 40 §4-3: 切り替え直後はページインを払うので、設定ごとに暖機を 1 本入れる。
set -u
mkdir -p bench/mtp52
# 注意: **64 は現在の CLI が受け付けない** (上限 32。52 §9 / 2026-08-20)。
# このスクリプトは 52 §1 を取ったときの形をそのまま残してある。回すなら
# `RuntimeConfiguration.allowedExpertCacheSlots` を一時的に広げること。
for SLOTS in 16 64; do
  echo "=== warmup slots=$SLOTS ==="
  TF_EXPERT_MMAP=0 ./.build/release/TsugumiCLI \
    --model scratch/gemma4-qat-sym.moepack --messages-file bench/math.json \
    --temperature 0 --max-new 32 --expert-cache-slots "$SLOTS" \
    --verification trusted-install >/dev/null 2>&1
  sleep 15
  echo "=== sweep slots=$SLOTS ==="
  # --advise off: このログは advise が製品に入る前 (どちらの腕も出さない) の
  # 条件で取られている。既定は 2026-08-20 に mmap の腕だけ on になった (52 §8)。
  ./bench/mtp51_mmap_ab.py --rounds 2 --slots "$SLOTS" --max-new 256 --advise off \
    --out "bench/mtp52/mmap_ab_256_math_slots${SLOTS}.log"
  sleep 20
done
