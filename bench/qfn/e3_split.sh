#!/bin/zsh
# E3 (docs/qfn-prefill/07): candidate B's routed step over the TB4 bridge, 48 layers x <chunks>, at share sigma.
#   bench/qfn/e3_split.sh <label> <sigma> [chunks]
# M6 serves (`--qfn-split-probe serve`, bound to its bridge0 address), the MBP drives. Both sides print a line per layer.
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; sigma=${2:?sigma}; chunks=${3:-4}
mgguf=$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf
rgguf='$HOME/dev/tsugumi/scratch/Qwen3.8-Flash-Next-DS4-IQ2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf'
out=$repo/bench/qfn/results/$(date +%F)-pair-$label
mkdir -p $out
bin=$repo/.build/release/TsugumiKernelCheck
for h in local m6; do
  if [[ $h == local ]]; then pgrep -x TsugumiKernelCheck && { echo "KernelCheck already running here"; exit 1; } || true
  else ssh m6 'pgrep -x TsugumiKernelCheck' && { echo "KernelCheck already running on M6"; exit 1; } || true; fi
done
m6_ip=$(ssh m6 "ifconfig bridge0 | awk '/inet / {print \$2}'")
mbp_ip=$(ifconfig bridge0 | awk '/inet / {print $2}')
sh $repo/bench/hostinfo.sh $bin > $out/hostinfo.mbp.txt
ssh m6 'cd dev/tsugumi && sh bench/hostinfo.sh .build/release/TsugumiKernelCheck' > $out/hostinfo.m6.txt
{ echo "bridge0 MBP $mbp_ip -> M6 $m6_ip / B sigma = $sigma"; ping -c 5 -q $m6_ip | tail -1; } > $out/link.txt
scmd="cd dev/tsugumi && .build/release/TsugumiKernelCheck --qfn-split-probe serve $m6_ip 5301 $rgguf scratch/qfn/routes/c2048 $sigma"
ccmd=($bin --qfn-split-probe run $m6_ip 5301 $mgguf $repo/scratch/qfn/routes/c2048 $sigma $chunks)
{ echo "M6: $scmd"; echo "MBP: ${ccmd[*]}"; } > $out/command.txt
for h in pre post; do :; done
{ vm_stat | grep -E 'Swapouts|Pageins'; } > $out/run-1.pre.txt; ssh m6 'vm_stat | grep -E "Swapouts|Pageins"' > $out/run-1.m6.pre.txt
if [[ $sigma != 0 ]]; then
  ssh m6 "$scmd" > $out/run-1.m6.log 2>&1 &
  until grep -q listening $out/run-1.m6.log 2>/dev/null; do sleep 1; done
fi
$repo/Scripts/qwen38/guarded.sh $out/run-1.log "${ccmd[@]}" || true
wait
{ vm_stat | grep -E 'Swapouts|Pageins'; } > $out/run-1.post.txt; ssh m6 'vm_stat | grep -E "Swapouts|Pageins"' > $out/run-1.m6.post.txt
grep -E 'chunk [0-9]|GUARD' $out/run-1.log
echo "evidence: ${out#$repo/}"
