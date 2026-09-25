#!/bin/zsh
# E1 (docs/qfn-prefill/07): today's QFN prefill bench on this host, stage by stage. Same command on both machines.
#   bench/qfn/e1_stages.sh <label> [n]        # env: QFN_MODEL (model dir), QFN_TOKENS (token file), COOL_S (20)
# Each run is a fresh process: 4096 tokens of the token file in chunks of 2048, Q38_SPLIT_PRE=1, Q38_MEM_LOG=1,
# the PLE table is the BF16 one (the manifest's). Output: bench/qfn/results/<date>-<host>-<label>/
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; n=${2:-3}
model=${QFN_MODEL:-$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2}
tokens=${QFN_TOKENS:-$repo/scratch/qwen38/prompt-code.tokens}
host=$(sysctl -n hw.model | tr ',' '-')
out=$repo/bench/qfn/results/$(date +%F)-$host-$label
mkdir -p $out
bin=$repo/.build/release/TsugumiKernelCheck
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiKernelCheck|swiftpm-testing-helper|mlx_lm|llama-' ; then
  echo "another model process is running; refusing"; exit 1
fi
sh $repo/bench/hostinfo.sh $bin > $out/hostinfo.txt
cmd=(env Q38_SPLIT_PRE=1 Q38_MEM_LOG=1 $bin --qwen38-prefill-bench $tokens --q38-tokens 4096 --q38-chunk 2048
     --q38-gguf $model/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf --q38-ple $model/ple-bf16/Qwen3.8-Flash-Next-PLE-BF16.gguf)
echo "$repo/Scripts/qwen38/guarded.sh run-N.log ${cmd[*]}" > $out/command.txt
env | grep -E '^(Q38_|TF_)' > $out/env.txt || true
snap() { { vm_stat; memory_pressure -Q | tail -1; iostat -d -I; } > $1; }
for i in $(seq 1 $n); do
  snap $out/run-$i.pre.txt
  GUARD_LOG=$out/run-$i.guard.txt caffeinate -i $repo/Scripts/qwen38/guarded.sh $out/run-$i.log "${cmd[@]}" || true
  snap $out/run-$i.post.txt
  grep -E '^\s+\[|prefill [0-9]+ tokens|GUARD' $out/run-$i.log
  [[ $i -lt $n ]] && sleep ${COOL_S:-20}
done
echo "evidence: ${out#$repo/}"
