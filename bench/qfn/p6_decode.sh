#!/bin/zsh
# P6 (docs/qfn-prefill/04 Q6, 22): this host alone decodes with today's code (all 48 layers, page cache), to see whether
# the M6 running every layer in decode (A k 48 kept for decode) could be as fast as the MBP alone.
#   bench/qfn/p6_decode.sh <label> [runs] [new tokens]    # env: QFN_MODEL, QFN_TOKENS (prompt-code.tokens)
# One process per run: --qwen38-generate, MTP off, instruct seed 1, PLE BF16. vm_stat before / after each run.
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; runs=${2:-3}; new=${3:-200}
model=${QFN_MODEL:-$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2}
tokens=${QFN_TOKENS:-$repo/scratch/qwen38/prompt-code.tokens}
host=$(sysctl -n hw.model | tr ',' '-')
out=$repo/bench/qfn/results/$(date +%F)-$host-$label
mkdir -p $out
bin=$repo/.build/release/TsugumiKernelCheck
{ sh $repo/bench/hostinfo.sh $bin; echo "iogpu.wired_limit_mb $(sysctl -n iogpu.wired_limit_mb)"; } > $out/hostinfo.txt
env | grep -E '^(QFN|Q38|TF)_' > $out/env.txt || true
cmd=($bin --qwen38-generate $tokens --q38-new $new --q38-mtp off --q38-out $out/generate.out
     --q38-gguf $model/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf --q38-ple $model/ple-bf16/Qwen3.8-Flash-Next-PLE-BF16.gguf)
echo "${cmd[*]}" > $out/command.txt
vmsnap() { vm_stat | grep -E 'Pageins|Swapouts'; memory_pressure -Q | tail -1; }
for i in $(seq 1 $runs); do
  vmsnap > $out/run-$i.pre.txt
  $repo/Scripts/qwen38/guarded.sh $out/run-$i.log "${cmd[@]}" || true
  vmsnap > $out/run-$i.post.txt
  echo "run $i: $(grep -iE 'steady|tok/s' $out/run-$i.log | tail -2 | tr '\n' ' ') | $(grep GUARD $out/run-$i.log)"
  sleep ${COOL_S:-20}
done
echo "evidence: ${out#$repo/}"
