#!/bin/zsh
# F4 again (docs/qfn-prefill/17): the M6 share in the process's own GPU-resident buffers, the rest streamed with F_NOCACHE.
#   bench/qfn/f4_owned.sh <label> <runs> <shape:share GB:resident GB>...   # env: QFN_MODEL, QFN_ROUTES, QFN_SCRATCH_GB (1)
# One process per config and run (TsugumiKernelCheck --qfn-residency-owned ... 4 passes). iogpu.wired_limit_mb is left
# as it is and recorded.
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; runs=${2:?runs}; shift 2
model=${QFN_MODEL:-$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2}
routes=${QFN_ROUTES:-$repo/scratch/qfn/routes}
host=$(sysctl -n hw.model | tr ',' '-')
out=$repo/bench/qfn/results/$(date +%F)-$host-$label
mkdir -p $out
bin=$repo/.build/release/TsugumiKernelCheck
{ sh $repo/bench/hostinfo.sh $bin; echo "iogpu.wired_limit_mb $(sysctl -n iogpu.wired_limit_mb)"; } > $out/hostinfo.txt
env | grep -E '^(QFN|Q38|TF)_' > $out/env.txt || true
touch $out/command.txt
vmsnap() { vm_stat | grep -E 'Pageins|Swapouts'; memory_pressure -Q | tail -1; }
for cfg in $@; do
  parts=(${(s/:/)cfg}); shape=$parts[1]; gb=$parts[2]; res=$parts[3]
  for i in $(seq 1 $runs); do
    name=$shape-$gb-r$res-$i
    cmd=($bin --qfn-residency-owned $model/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf
         $model/down/Qwen3.8-Flash-Next-Q2KDown640.gguf $routes/c2048 $shape $gb $res 4 ${QFN_SCRATCH_GB:-1})
    echo "${cmd[*]}" >> $out/command.txt
    vmsnap > $out/$name.pre.txt
    $repo/Scripts/qwen38/guarded.sh $out/$name.log "${cmd[@]}" || true
    vmsnap > $out/$name.post.txt
    sed "s/^/$name /" $out/$name.log
    sleep ${COOL_S:-4}
  done
done
echo "evidence: ${out#$repo/}"
