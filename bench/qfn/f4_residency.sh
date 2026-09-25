#!/bin/zsh
# F4 (docs/qfn-prefill/12 §2): how much of an M6 share stays in the page cache when the same route streams 4 times.
#   bench/qfn/f4_residency.sh <label> [runs] [GB...]    # env: QFN_MODEL, QFN_ROUTES, QFN_SCRATCH_GB (1)
# Per shape (layers = candidate A, experts = candidate B), share size and run: a new process drops the share from the
# page cache and makes 4 passes (TsugumiKernelCheck --qfn-residency). iogpu.wired_limit_mb is left as it is and recorded.
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; runs=${2:-3}; shift 2 2>/dev/null || shift $#
sizes=(${@:-8 10 12 14})
model=${QFN_MODEL:-$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2}
routes=${QFN_ROUTES:-$repo/scratch/qfn/routes}
host=$(sysctl -n hw.model | tr ',' '-')
out=$repo/bench/qfn/results/$(date +%F)-$host-$label
mkdir -p $out
bin=$repo/.build/release/TsugumiKernelCheck
{ sh $repo/bench/hostinfo.sh $bin; echo "iogpu.wired_limit_mb $(sysctl -n iogpu.wired_limit_mb)"; } > $out/hostinfo.txt
env | grep -E '^(QFN|Q38|TF)_' > $out/env.txt || true
: > $out/command.txt
vmsnap() { vm_stat | grep -E 'Pageins|Swapouts'; memory_pressure -Q | tail -1; }
for shape in layers experts; do
  for gb in $sizes; do
    for i in $(seq 1 $runs); do
      cmd=($bin --qfn-residency $model/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf
           $model/down/Qwen3.8-Flash-Next-Q2KDown640.gguf $routes/c2048 $shape $gb 4 ${QFN_SCRATCH_GB:-1})
      echo "${cmd[*]}" >> $out/command.txt
      vmsnap > $out/$shape-$gb-$i.pre.txt
      $repo/Scripts/qwen38/guarded.sh $out/$shape-$gb-$i.log "${cmd[@]}" || true
      vmsnap > $out/$shape-$gb-$i.post.txt
      sed "s/^/$shape $gb #$i /" $out/$shape-$gb-$i.log
      sleep ${COOL_S:-4}
    done
  done
done
echo "evidence: ${out#$repo/}"
