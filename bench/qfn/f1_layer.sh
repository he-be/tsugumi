#!/bin/zsh
# F1 (docs/qfn-prefill/12 §2): one layer's GEMMs at T = 2048 through the tensor ops, resident and cold, on this host.
#   bench/qfn/f1_layer.sh <label> [iterations] [cold runs] [layers...]    # env: QFN_MODEL, QFN_ROUTES
# resident: one process per layer, arms A C D M, `iterations` rounds after a warm-up (TsugumiKernelCheck
#           --qfn-layer-bench ... resident). cold: per layer, arm and run, a new process drops the layer's ranges from
#           the page cache (msync MS_INVALIDATE, checked with mincore in its log) and times one round.
# Each run's vm_stat (Pageins / Swapouts) before and after is kept next to its log.
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; it=${2:-5}; cruns=${3:-3}; shift 3 2>/dev/null || shift $#
layers=(${@:-0 3 24 47})
model=${QFN_MODEL:-$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2}
routes=${QFN_ROUTES:-$repo/scratch/qfn/routes}
host=$(sysctl -n hw.model | tr ',' '-')
out=$repo/bench/qfn/results/$(date +%F)-$host-$label
mkdir -p $out
bin=$repo/.build/release/TsugumiKernelCheck
{ sh $repo/bench/hostinfo.sh $bin; echo "iogpu.wired_limit_mb $(sysctl -n iogpu.wired_limit_mb)"; } > $out/hostinfo.txt
env | grep -E '^(QFN|Q38|TF)_' > $out/env.txt || true
: > $out/command.txt
gguf=$model/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf
side=$model/down/Qwen3.8-Flash-Next-Q2KDown640.gguf
vmsnap() { vm_stat | grep -E 'Pageins|Swapouts'; memory_pressure -Q | tail -1; }
for L in $layers; do
  cmd=($bin --qfn-layer-bench $gguf $side $routes/c2048-l$L.bin resident $it ACDM)
  echo "${cmd[*]}" >> $out/command.txt
  vmsnap > $out/layer-$L.resident.pre.txt
  $repo/Scripts/qwen38/guarded.sh $out/layer-$L.resident.log "${cmd[@]}" || true
  vmsnap > $out/layer-$L.resident.post.txt
  cat $out/layer-$L.resident.log
  sleep ${COOL_S:-4}
done
for L in $layers; do
  for arm in C D M; do
    for i in $(seq 1 $cruns); do
      cmd=($bin --qfn-layer-bench $gguf $side $routes/c2048-l$L.bin cold 1 $arm)
      echo "${cmd[*]}" >> $out/command.txt
      vmsnap > $out/layer-$L.cold-$arm-$i.pre.txt
      $repo/Scripts/qwen38/guarded.sh $out/layer-$L.cold-$arm-$i.log "${cmd[@]}" || true
      vmsnap > $out/layer-$L.cold-$arm-$i.post.txt
      grep -E 'not in page cache before|cold read|arm .* GPU' $out/layer-$L.cold-$arm-$i.log | sed "s/^/L$L $arm #$i /"
    done
  done
done
echo "evidence: ${out#$repo/}"
