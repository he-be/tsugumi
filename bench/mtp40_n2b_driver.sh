#!/bin/bash
# N2 の後半 — 運用点は投機オンなので、ブロック経路でも同じ分解を取る。
#
# 37 §1a は「投機オン (bs>0) では `decode/tok` の 4 項が全部 0.00 になる」を
# 穴として挙げた。その穴は **計器が無いのではなく、別の計器が持っている**:
# ブロック経路は `TF_PREFILL_HOST_PROFILE=1` で 1 ブロック 1 対の
#   [prefill gpu attn=.. shared=.. moe=.. tail=.. head=.. total=..]
#   [prefill host enc.front=.. wait.front=.. ... call=..s | tiles=.. experts=.. io=..]
#   [prefill gpuq busy=..s(..%) idle=..s(..%) span=..s buffers=..]
# を吐く (36 §14 がこれで 32 スロット・bs=3・temp 0 の台帳を作った)。
# 本スクリプトはそれを **48 スロット・運用点 (temp 既定)・3 反復**で取り直し、
# bs=2 (39 §2 の最適) と bs=3 (36 §14 と同じ幅) の 2 枚を並べる。
#
# `Sources/` は 1 行も触らない。
# 使い方:  ./bench/mtp40_n2b_driver.sh        # 14 run (約 10 分)
#          DRY=1 ./bench/mtp40_n2b_driver.sh
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUT="$ROOT/bench/logs/40_n2b"
mkdir -p "$OUT"
LOG="$OUT/driver.log"
COOL_S="${COOL_S:-20}"
MODEL="${MODEL:-scratch/gemma4-qat.gturbo}"
IMGDIR="${TF_SAMPLE_IMGS:-$HOME/Pictures/sample_imgs}"
IMG="${TF_SWEEP_IMAGE:-$IMGDIR/NO_FUSION_0401_001.png}"   # 39 (N1) の画像 A
MAXNEW="${MAXNEW:-192}"
SLOTS="${SLOTS:-48}"

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
  found=$(pgrep -fl 'TurboFieldfare|llama-server|llama-bench|mlx' \
          | grep -v -e pgrep -e '/bin/bash' -e '/bin/sh' -e '/bin/zsh' -e 'mtp40_n2')
  if [ -n "$found" ]; then
    say "ABORT: 他の推論プロセスが居る: $found"
    exit 3
  fi
}

# $1=名前 $2=register $3=bs $4=seed
run_one() {
  local name="$1" reg="$2" bs="$3" seed="$4"
  local p; p="$(prompt_for "$reg")"
  local -a cmd=(
    env TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_GPU_PROFILE=1
    "$ROOT/.build/release/TurboFieldfareCLI"
    --model "$MODEL"
    --messages-file "$p"
    --image "$IMG" --image-tokens 280
    --seed "$seed" --max-new "$MAXNEW" --max-context 4096
    --expert-cache-slots "$SLOTS"
    --thinking off --draft-block-size "$bs"
    --verification trusted-install --prefill on
  )
  if [ "${DRY:-0}" = "1" ]; then
    echo "${cmd[*]}  > $OUT/$name.out 2> $OUT/$name.err"
    return 0
  fi
  guard
  say "start $name (reg=$reg bs=$bs seed=$seed slots=$SLOTS)"
  "${cmd[@]}" > "$OUT/$name.out" 2> "$OUT/$name.err"
  local rc=$?
  say "done  $name rc=$rc  $(grep -o 'tok/s=[0-9.]*' "$OUT/$name.err" | tail -1)"
  sleep "$COOL_S"
}

say "=== 40 N2b start $(date) ==="
say "image = $IMG ; slots = $SLOTS"
md5 "$IMG" | tee -a "$LOG"
say "sysctl $(sysctl -n iogpu.wired_limit_mb) MB wired limit"
say "commit $(git rev-parse --short HEAD)"

run_one warm             ja_prose   2 9999
run_one ja_prose_bs2_r1  ja_prose   2 1234
run_one en_bullets_bs2_r1 en_bullets 2 1234
run_one en_bullets_bs3_r1 en_bullets 3 1234
run_one ja_prose_bs3_r1  ja_prose   3 1234
run_one ja_prose_bs3_r2  ja_prose   3 1235
run_one en_bullets_bs3_r2 en_bullets 3 1235
run_one en_bullets_bs2_r2 en_bullets 2 1235
run_one ja_prose_bs2_r2  ja_prose   2 1235
run_one ja_prose_bs2_r3  ja_prose   2 1236
run_one en_bullets_bs2_r3 en_bullets 2 1236
run_one en_bullets_bs3_r3 en_bullets 3 1236
run_one ja_prose_bs3_r3  ja_prose   3 1236
run_one ja_prose_bs2_drift ja_prose  2 1234   # 先頭と同一条件

say "=== 40 N2b ALL DONE $(date) ==="
