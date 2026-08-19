#!/bin/bash
# 37-M9-B-PROPOSAL §3 の N2 — decode の壁時計を GPU busy と待ちに割る。
#
# 既に在る計器 2 本を立てて回すだけで、`Sources/` は 1 行も触らない:
#   TF_PREFILL_HOST_PROFILE=1  … decode 側の busy/idle 記録を有効にする
#                                (RealForwardRunner.swift:2696 の profilingDecodeQueue)
#   TF_PREFILL_GPU_PROFILE=1   … その内訳を attn/shared/moe/head に割る
#                                (同 :2709、PrefillGPUProfile.swift)
# どちらも decode 32 歩ごとに stderr へ 1 対の行を吐く:
#   [decode gpuq busy=..s(..%) idle=..s(..%) span=..s buffers=N tok=32]
#   [decode gpu attn=..s(..%) shared=..s(..%) moe=..s(..%) head=..s(..%) total=..s]
#
# 軸は 2 本だけ: register (2^H の代理。38 §3) × スロット数。
#   ja_prose   = bench/mtp_goal_prompt.json        (2^H 22.8、有利)
#   en_bullets = bench/mtp_register_en_bullets.json (2^H 41.8、苦手)
# 運用点で回す (温度・top-k・top-p は CLI 既定のまま **指定しない**。40 §4-4)。
# 20 秒クールダウン、3 反復 (seed 1234/1235/1236)、スロット切り替えは 1 回だけ
# (48 → 32。48 の前には暖機を 1 本入れる。40 §4-3)。
# 計器そのものの摂動を測るため、48 スロットの 2 セルは env 無しでも 1 本ずつ回す。
# 先頭と同じ条件を末尾にもう 1 本置いて熱ドリフトを見る (38 の作法)。
#
# 使い方:  ./bench/mtp40_n2_driver.sh            # 全 17 run (約 12 分)
#          DRY=1 ./bench/mtp40_n2_driver.sh      # 組み立てたコマンドを印字するだけ
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUT="$ROOT/bench/logs/40_n2"
mkdir -p "$OUT"
LOG="$OUT/driver.log"
COOL_S="${COOL_S:-20}"
MODEL="${MODEL:-scratch/gemma4-qat.gturbo}"
IMGDIR="${TF_SAMPLE_IMGS:-$HOME/Pictures/sample_imgs}"
IMG="${TF_SWEEP_IMAGE:-$IMGDIR/NO_FUSION_0401_001.png}"   # 39 (N1) の画像 A
MAXNEW="${MAXNEW:-192}"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

prompt_for() {
  case "$1" in
    ja_prose)   echo "bench/mtp_goal_prompt.json" ;;
    en_bullets) echo "bench/mtp_register_en_bullets.json" ;;
    *) echo "unknown register: $1" >&2; exit 2 ;;
  esac
}

guard() {
  # 自分以外の推論プロセスが居たら測らない (mtp_cd_sweep.py と同じ作法)。
  local found
  found=$(pgrep -fl 'TurboFieldfare|llama-server|llama-bench|mlx' \
          | grep -v -e pgrep -e '/bin/bash' -e '/bin/sh' -e '/bin/zsh' -e 'mtp40_n2_driver')
  if [ -n "$found" ]; then
    say "ABORT: 他の推論プロセスが居る: $found"
    exit 3
  fi
}

# $1=名前 $2=register $3=slots $4=seed $5=profile(on|off)
run_one() {
  local name="$1" reg="$2" slots="$3" seed="$4" prof="$5"
  local p; p="$(prompt_for "$reg")"
  local -a env_prefix=()
  if [ "$prof" = "on" ]; then
    env_prefix=(env TF_PREFILL_HOST_PROFILE=1 TF_PREFILL_GPU_PROFILE=1)
  else
    env_prefix=(env)
  fi
  local -a cmd=(
    "${env_prefix[@]}"
    "$ROOT/.build/release/TurboFieldfareCLI"
    --model "$MODEL"
    --messages-file "$p"
    --image "$IMG" --image-tokens 280
    --seed "$seed" --max-new "$MAXNEW" --max-context 4096
    --expert-cache-slots "$slots"
    --thinking off --draft-block-size 0
    --verification trusted-install --prefill on
  )
  if [ "${DRY:-0}" = "1" ]; then
    echo "${cmd[*]}  > $OUT/$name.out 2> $OUT/$name.err"
    return 0
  fi
  guard
  say "start $name (reg=$reg slots=$slots seed=$seed prof=$prof)"
  "${cmd[@]}" > "$OUT/$name.out" 2> "$OUT/$name.err"
  local rc=$?
  say "done  $name rc=$rc  $(grep -o 'tok/s=[0-9.]*' "$OUT/$name.err" | tail -1)"
  sleep "$COOL_S"
}

say "=== 40 N2 start $(date) ==="
say "image = $IMG"
md5 "$IMG" | tee -a "$LOG"
say "sysctl $(sysctl -n iogpu.wired_limit_mb) MB wired limit"
say "commit $(git rev-parse --short HEAD)"

# --- 48 スロット (N1 の運用点) ---------------------------------------------
run_one warm_s48            ja_prose   48 9999 on   # 暖機 (6.78GB のページイン)
run_one ja_prose_s48_r1     ja_prose   48 1234 on
run_one en_bullets_s48_r1   en_bullets 48 1234 on
run_one en_bullets_s48_r2   en_bullets 48 1235 on
run_one ja_prose_s48_r2     ja_prose   48 1235 on
run_one ja_prose_s48_r3     ja_prose   48 1236 on
run_one en_bullets_s48_r3   en_bullets 48 1236 on
run_one ja_prose_s48_noprof ja_prose   48 1234 off  # 計器の摂動
run_one en_bullets_s48_noprof en_bullets 48 1234 off
run_one ja_prose_s48_drift  ja_prose   48 1234 on   # 先頭と同一条件 (熱ドリフト)

# --- 32 スロット (37 §1a の 9 run と同じスロット数) ------------------------
run_one warm_s32            ja_prose   32 9999 on
run_one ja_prose_s32_r1     ja_prose   32 1234 on
run_one en_bullets_s32_r1   en_bullets 32 1234 on
run_one en_bullets_s32_r2   en_bullets 32 1235 on
run_one ja_prose_s32_r2     ja_prose   32 1235 on
run_one ja_prose_s32_r3     ja_prose   32 1236 on
run_one en_bullets_s32_r3   en_bullets 32 1236 on

say "=== 40 N2 ALL DONE $(date) ==="
