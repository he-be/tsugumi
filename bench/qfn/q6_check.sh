#!/bin/zsh
# Q6 correctness (docs/qfn-prefill/04 Q6, 22 §4 の 3): TsugumiKernelCheck --qwen38-pair-check on one 本文 1 prompt, the
# MBP alone against the pair, for each k. Evidence is filed automatically.
#   bench/qfn/q6_check.sh <label> [k ...]      # default k 48
# QFN_CHECK_PROMPT (bench/m6/prompts/ja-twomachine.json), QFN_CHECK_NEW (64). The M6 serve runs with its defaults.
# Output: bench/qfn/results/<date>-pair-<label>/ (hostinfo.{mbp,m6}.txt link.txt env.txt command.txt
#   check-k<k>.{log,guard.txt,pre.txt,post.txt,pre.m6.txt,post.m6.txt} serve.m6.log serve.m6.guard.txt)
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; shift
ks=(${@:-48})
model=${QFN_MODEL:-$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2}
prompt=${QFN_CHECK_PROMPT:-$repo/bench/m6/prompts/ja-twomachine.json}
new=${QFN_CHECK_NEW:-64}
bin=$repo/.build/release/TsugumiKernelCheck
[[ -x $bin ]] || { echo "no $bin"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiToolLoopCheck|TsugumiKernelCheck|mlx_lm|llama-'; then
    echo "another model process is running here"; exit 1
fi
out=$repo/bench/qfn/results/$(date +%F)-pair-$label
mkdir -p $out
sh $repo/bench/hostinfo.sh $bin > $out/hostinfo.mbp.txt
{ env | grep -E '^(Q38_|TF_|QFN_)' || true; } > $out/env.txt
: > $out/command.txt
trap "$repo/bench/qfn/pair_serve.sh stop $out" EXIT
$repo/bench/qfn/pair_serve.sh start $out
m6=$(awk '/-> M6/ {print $NF}' $out/link.txt)
snap() { { date '+%H:%M:%S'; memory_pressure -Q | tail -1; vm_stat; } > $1; }
for k in $ks; do
    check=($bin --qwen38-pair-check $prompt --q38-pair $m6 --q38-pair-k $k --q38-new $new
           --q38-gguf $model/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf --q38-ple $model/ple-bf16/Qwen3.8-Flash-Next-PLE-BF16.gguf)
    echo "MBP: GUARD_LOG=check-k$k.guard.txt caffeinate -i Scripts/qwen38/guarded.sh check-k$k.log ${check[*]}" >> $out/command.txt
    snap $out/check-k$k.pre.txt; $repo/bench/qfn/pair_serve.sh snap $out/check-k$k.pre.m6.txt
    rc=0
    GUARD_LOG=$out/check-k$k.guard.txt caffeinate -i $repo/Scripts/qwen38/guarded.sh $out/check-k$k.log "${check[@]}" || rc=$?
    snap $out/check-k$k.post.txt; $repo/bench/qfn/pair_serve.sh snap $out/check-k$k.post.m6.txt
    echo "k $k: exit $rc"; grep -E '^(rows|greedy|solo|forced|free)' $out/check-k$k.log || true
    $repo/bench/qfn/pair_serve.sh alive || { echo "the M6 serve ended"; exit 1; }
    sleep ${COOL_S:-20}
done
echo "evidence: ${out#$repo/}"
