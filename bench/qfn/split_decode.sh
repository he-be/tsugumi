#!/bin/zsh
# docs/qfn-prefill/23: the MTP spec decode (chain 1) with the trunk split by layers over the two machines.
#   bench/qfn/split_decode.sh <label> <runs> <arm>...        # arms S0 S16 S24 S32 S48 (S<j>: the M6 runs layers 0 .. j-1)
# First the g run (S0, the sampler, instruct SEED (1)): if it stops at EOS before NEW tokens, the seed goes up by one
# (up to 5 tries). Then every arm `runs` times round-robin with --q38-force <g's out>, one process per side and run,
# both under guarded.sh with GUARD_LOG (wired max). A run with Swapouts on either side, or a non-zero exit, is invalid
# and the arm gets one more run at the end (at most 2 more per arm, 23 §0).
# env: NEW (400), TOKENS (scratch/qwen38/prompt-code.tokens, the same relative path on the M6), SEED (1), COOL_S (20).
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; runs=${2:?runs}; shift 2
arms=($@)
new=${NEW:-400}; tokens=${TOKENS:-scratch/qwen38/prompt-code.tokens}; seed=${SEED:-1}
mm=$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2
rm_='$HOME/dev/tsugumi/scratch/Qwen3.8-Flash-Next-DS4-IQ2'
gg=Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf; pl=ple-bf16/Qwen3.8-Flash-Next-PLE-BF16.gguf
out=$repo/bench/qfn/results/$(date +%F)-pair-$label
mkdir -p $out
bin=$repo/.build/release/TsugumiKernelCheck
pgrep -x TsugumiKernelCheck && { echo "KernelCheck already running here"; exit 1; } || true
ssh m6 'pgrep -x TsugumiKernelCheck' && { echo "KernelCheck already running on M6"; exit 1; } || true
m6_ip=$(ssh m6 "ifconfig bridge0 | awk '/inet / {print \$2}'")
mbp_ip=$(ifconfig bridge0 | awk '/inet / {print $2}')
{ sh $repo/bench/hostinfo.sh $bin; echo "iogpu.wired_limit_mb $(sysctl -n iogpu.wired_limit_mb)"; } > $out/hostinfo.mbp.txt
ssh m6 'cd dev/tsugumi && sh bench/hostinfo.sh .build/release/TsugumiKernelCheck; echo "iogpu.wired_limit_mb $(sysctl -n iogpu.wired_limit_mb)"; mount | grep SSD256' > $out/hostinfo.m6.txt
{ echo "bridge0 MBP $mbp_ip -> M6 $m6_ip"; ping -c 5 -q $m6_ip | tail -1; } > $out/link.txt
env | grep -E '^(QFN|Q38|TF)_' > $out/env.txt || true
echo "NEW=$new TOKENS=$tokens" >> $out/env.txt
: > $out/command.txt
vmsnap() { vm_stat | grep -E 'Pageins|Swapouts'; memory_pressure -Q | tail -1; }
m6snap() { ssh m6 'vm_stat | grep -E "Pageins|Swapouts"; memory_pressure -Q | tail -1'; }
mbpcmd() { echo $bin --qwen38-split-decode run $m6_ip 5303 $1 $repo/$tokens --q38-gguf $mm/$gg --q38-ple $mm/$pl --q38-new $new; }

# The g run.
g=$out/g.out
for try in 1 2 3 4 5; do
  cmd=($(mbpcmd 0) --q38-seed $seed --q38-out $g)
  echo "g seed $seed: ${cmd[*]}" >> $out/command.txt
  vmsnap > $out/g-seed$seed.pre.txt
  GUARD_LOG=$out/g-seed$seed.guard.txt $repo/Scripts/qwen38/guarded.sh $out/g-seed$seed.log "${cmd[@]}" || true
  vmsnap > $out/g-seed$seed.post.txt
  got=$(head -1 $g | tr ',' '\n' | wc -l | tr -d ' ')
  echo "g seed $seed: $got tokens | $(grep GUARD $out/g-seed$seed.log)"
  (( got >= new )) && break
  seed=$(( seed + 1 ))
  sleep ${COOL_S:-20}
done
(( got >= new )) || { echo "no g run reached $new tokens"; exit 1; }
echo "g: seed $seed" >> $out/env.txt
sleep ${COOL_S:-20}

typeset -A valid extra
for a in $arms; do valid[$a]=0; extra[$a]=0; done
one() {  # <arm> <name>
  local a=$1 name=$2 j=${1#S}
  local scmd="cd dev/tsugumi && GUARD_LOG=scratch/split-decode.m6.guard.txt Scripts/qwen38/guarded.sh scratch/split-decode.m6.log .build/release/TsugumiKernelCheck --qwen38-split-decode serve $m6_ip 5303 $j \$HOME/dev/tsugumi/$tokens --q38-gguf $rm_/$gg --q38-ple $rm_/$pl --q38-new $new"
  local ccmd=($(mbpcmd $j) --q38-force $g --q38-out $out/$name.out)
  { (( j > 0 )) && echo "$name M6: $scmd"; echo "$name MBP: ${ccmd[*]}"; } >> $out/command.txt
  vmsnap > $out/$name.pre.txt; m6snap > $out/$name.m6.pre.txt
  if (( j > 0 )); then
    ssh m6 'rm -f dev/tsugumi/scratch/split-decode.m6.log dev/tsugumi/scratch/split-decode.m6.guard.txt'
    ssh m6 "$scmd" > /dev/null 2>&1 &
    until ssh m6 'grep -q listening dev/tsugumi/scratch/split-decode.m6.log 2>/dev/null'; do sleep 1; done
  fi
  GUARD_LOG=$out/$name.guard.txt $repo/Scripts/qwen38/guarded.sh $out/$name.log "${ccmd[@]}" || true
  wait
  if (( j > 0 )); then
    ssh m6 'cat dev/tsugumi/scratch/split-decode.m6.log' > $out/$name.m6.log
    ssh m6 'cat dev/tsugumi/scratch/split-decode.m6.guard.txt' > $out/$name.m6.guard.txt
  fi
  vmsnap > $out/$name.post.txt; m6snap > $out/$name.m6.post.txt
  local ok=1
  grep -q 'GUARD: exit 0, Swapouts delta peak 0 pages' $out/$name.log || ok=0
  (( j > 0 )) && { grep -q 'GUARD: exit 0, Swapouts delta peak 0 pages' $out/$name.m6.log || ok=0; }
  (( ok )) && valid[$a]=$(( valid[$a] + 1 ))
  echo "$name: $(grep -E 'window tokens' $out/$name.log) | $(grep -E 'accepted' $out/$name.log) | valid $ok | $(grep -h GUARD $out/$name.log $out/$name.m6.log 2>/dev/null | tr '\n' ' ')"
  sleep ${COOL_S:-20}
}
for i in $(seq 1 $runs); do
  for a in $arms; do one $a $a-$i; done
done
for a in $arms; do
  while (( valid[$a] < runs && extra[$a] < 2 )); do
    extra[$a]=$(( extra[$a] + 1 ))
    one $a $a-x${extra[$a]}
  done
done
echo "evidence: ${out#$repo/}"
