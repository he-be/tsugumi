#!/bin/bash
# N2 の付録 — ブロックの `attn` バケットの中身を割る (帰属専用、計時ではない)。
#
# `TF_PREFILL_GPU_PROFILE=2` は層の前半を 1 グループ 1 コマンドバッファに切って
# embed / norm / qkv / rope / kvcopy / attn.swa / attn.full / oproj / post の
# 9 つに割る (PrefillGPUProfile.swift の Detail)。**切ると GPU がグループ間で
# 遊ぶので壁時計が膨らむ** — doc comment が「帰属のためであって run の計時では
# ない」と明記している。したがって本スクリプトの run は
# **N2b の計時表に混ぜてはいけない**。ログを別ディレクトリに置くのはそのため。
#
# 細分が効くのはチャンク経路だけなので、`--draft-block-size 2` (39 §2 の最適) で
# 回して verify ブロックを通す。decode (bs=0) 側は細分の口が無い (N2 §5 の限界)。
#
# 使い方:  ./bench/mtp40_n2c_driver.sh        # 7 run (約 6 分)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUT="$ROOT/bench/logs/40_n2c"
mkdir -p "$OUT"
LOG="$OUT/driver.log"
COOL_S="${COOL_S:-20}"
MODEL="${MODEL:-scratch/gemma4-qat.moepack}"
IMGDIR="${TF_SAMPLE_IMGS:-$HOME/Pictures/sample_imgs}"
IMG="${TF_SWEEP_IMAGE:-$IMGDIR/NO_FUSION_0401_001.png}"
MAXNEW="${MAXNEW:-192}"
SLOTS="${SLOTS:-48}"
BS="${BS:-2}"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

prompt_for() {
  case "$1" in
    ja_prose)   echo "bench/mtp_goal_prompt.json" ;;
    en_bullets) echo "bench/mtp_register_en_bullets.json" ;;
    *) echo "unknown register: $1" >&2; exit 2 ;;
  esac
}

guard() {
  local found
  found=$(pgrep -fl 'Tsugumi|llama-server|llama-bench|mlx' \
          | grep -v -e pgrep -e '/bin/bash' -e '/bin/sh' -e '/bin/zsh' -e 'mtp40_n2')
  if [ -n "$found" ]; then say "ABORT: 他の推論プロセスが居る: $found"; exit 3; fi
}

run_one() {
  local name="$1" reg="$2" seed="$3"
  local p; p="$(prompt_for "$reg")"
  local -a cmd=(
    env TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_GPU_PROFILE=2
    "$ROOT/.build/release/TsugumiCLI"
    --model "$MODEL" --messages-file "$p"
    --image "$IMG" --image-tokens 280
    --seed "$seed" --max-new "$MAXNEW" --max-context 4096
    --expert-cache-slots "$SLOTS"
    --thinking off --draft-block-size "$BS"
    --verification trusted-install --prefill on
  )
  if [ "${DRY:-0}" = "1" ]; then
    echo "${cmd[*]}  > $OUT/$name.out 2> $OUT/$name.err"; return 0
  fi
  guard
  say "start $name (reg=$reg seed=$seed bs=$BS slots=$SLOTS detail=2)"
  "${cmd[@]}" > "$OUT/$name.out" 2> "$OUT/$name.err"
  say "done  $name rc=$?  $(grep -o 'tok/s=[0-9.]*' "$OUT/$name.err" | tail -1)"
  sleep "$COOL_S"
}

say "=== 40 N2c (detail=2, 帰属専用) start $(date) ==="
say "image = $IMG ; slots = $SLOTS ; bs = $BS"
md5 "$IMG" | tee -a "$LOG"
say "commit $(git rev-parse --short HEAD)"

run_one warm             ja_prose   9999
run_one ja_prose_r1      ja_prose   1234
run_one en_bullets_r1    en_bullets 1234
run_one en_bullets_r2    en_bullets 1235
run_one ja_prose_r2      ja_prose   1235
run_one ja_prose_r3      ja_prose   1236
run_one en_bullets_r3    en_bullets 1236

say "=== 40 N2c ALL DONE $(date) ==="
