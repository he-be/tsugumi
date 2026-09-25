#!/bin/zsh
# F2 / F3 (docs/qfn-prefill/12): candidates A and B simulated on both machines, 48 layers x <chunks> (T = 2048).
#   bench/qfn/f23_pair.sh <label> <runs> <shape:param:resident GB[:feed[:slab]]>...    # e.g. A:32:11.1 B:0.55:11.1 A:20:0 A:0:0 A:40:11.1:0.59 A:48:11.1:0.59:2
# param is k (A) or sigma (B); resident GB 0 = the M6 streams its share every time (cold); A:0 / B:0 = the MBP alone.
# feed (A only, docs/qfn-prefill/19 §2 Q4-b) = QFN_FEED on both sides: that fraction of the M6's streamed bytes comes from the MBP.
# slab (A only, docs/qfn-prefill/04 Q5) = QFN_SLAB on the M6: staging buffers, the next layers read behind this one's GPU (1).
# Runs go round-robin over the configs (run 1 of every config, then run 2, ...). One process per side and run: the M6
# serves (`--qfn-pair-sim serve`, bound to its bridge0 address) under its own guarded.sh, the MBP drives. env: CHUNKS (2).
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; runs=${2:?runs}; shift 2
chunks=${CHUNKS:-2}
mm=$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2
rm_='$HOME/dev/tsugumi/scratch/Qwen3.8-Flash-Next-DS4-IQ2'
gg=Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf; sc=down/Qwen3.8-Flash-Next-Q2KDown640.gguf
out=$repo/bench/qfn/results/$(date +%F)-pair-$label
mkdir -p $out
bin=$repo/.build/release/TsugumiKernelCheck
pgrep -x TsugumiKernelCheck && { echo "KernelCheck already running here"; exit 1; } || true
ssh m6 'pgrep -x TsugumiKernelCheck' && { echo "KernelCheck already running on M6"; exit 1; } || true
m6_ip=$(ssh m6 "ifconfig bridge0 | awk '/inet / {print \$2}'")
mbp_ip=$(ifconfig bridge0 | awk '/inet / {print $2}')
sh $repo/bench/hostinfo.sh $bin > $out/hostinfo.mbp.txt
ssh m6 'cd dev/tsugumi && sh bench/hostinfo.sh .build/release/TsugumiKernelCheck; echo "iogpu.wired_limit_mb $(sysctl -n iogpu.wired_limit_mb)"' > $out/hostinfo.m6.txt
{ echo "bridge0 MBP $mbp_ip -> M6 $m6_ip"; ping -c 5 -q $m6_ip | tail -1; } > $out/link.txt
env | grep -E '^(QFN|Q38|TF)_' > $out/env.txt || true
: > $out/command.txt
vmsnap() { vm_stat | grep -E 'Pageins|Swapouts'; memory_pressure -Q | tail -1; }
for i in $(seq 1 $runs); do
  for cfg in $@; do
    parts=(${(s/:/)cfg}); shape=$parts[1]; p=$parts[2]; res=$parts[3]; feed=${parts[4]:-0}; slab=${parts[5]:-1}
    name=$shape-$p-r$res
    [[ $feed != 0 ]] && name=$name-f$feed
    [[ $slab != 1 ]] && name=$name-s$slab
    name=$name-$i
    solo=0; [[ $p == 0 ]] && solo=1
    scmd="cd dev/tsugumi && QFN_FEED=$feed QFN_SLAB=$slab Scripts/qwen38/guarded.sh scratch/qfn-pair.m6.log .build/release/TsugumiKernelCheck --qfn-pair-sim serve $m6_ip 5302 $rm_/$gg $rm_/$sc scratch/qfn/routes/c2048 $shape $p $res"
    ccmd=(env QFN_FEED=$feed $bin --qfn-pair-sim run $m6_ip 5302 $mm/$gg $mm/$sc $repo/scratch/qfn/routes/c2048 $shape $p $chunks)
    { echo "$name M6: $scmd"; echo "$name MBP: ${ccmd[*]}"; } >> $out/command.txt
    vmsnap > $out/$name.pre.txt; ssh m6 'vm_stat | grep -E "Pageins|Swapouts"; memory_pressure -Q | tail -1' > $out/$name.m6.pre.txt
    if (( ! solo )); then
      ssh m6 'rm -f dev/tsugumi/scratch/qfn-pair.m6.log'
      ssh m6 "$scmd" > /dev/null 2>&1 &
      until ssh m6 'grep -q listening dev/tsugumi/scratch/qfn-pair.m6.log 2>/dev/null'; do sleep 1; done
    fi
    $repo/Scripts/qwen38/guarded.sh $out/$name.log "${ccmd[@]}" || true
    wait
    (( solo )) || ssh m6 'cat dev/tsugumi/scratch/qfn-pair.m6.log' > $out/$name.m6.log
    vmsnap > $out/$name.post.txt; ssh m6 'vm_stat | grep -E "Pageins|Swapouts"; memory_pressure -Q | tail -1' > $out/$name.m6.post.txt
    echo "$name: $(grep -E '^total' $out/$name.log) | $(grep -h GUARD $out/$name.log $out/$name.m6.log 2>/dev/null | tr '\n' ' ')"
    sleep ${COOL_S:-4}
  done
done
echo "evidence: ${out#$repo/}"
