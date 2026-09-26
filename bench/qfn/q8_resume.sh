#!/bin/zsh
# Q8 C1 correctness (docs/qfn-prefill/29 §3): TsugumiKernelCheck --qwen38-resume (checkpoints against a whole recompute,
# docs/qwen38/18 §2) with the chunks cut at the checkpoints (the default) and with Q38_CUT_INCHUNK=1, MBP alone.
#   bench/qfn/q8_resume.sh <label> [mtp:chunk ...]      # default off:2048 spec:2048 off:12
# A chunk under Q38_GDN_MIN_T (16) runs the serial GDN step, so chunk 12 checks that path's segments.
# QFN_PAIR_K=<k>: the pair (the M6 serve with its defaults, bench/qfn/pair_serve.sh), both arms with Q38_PAIR / Q38_PAIR_K.
# QFN_RESUME_A / QFN_RESUME_B (scratch/qwen38/resume/tool.tokens, tool-u5.tokens), QFN_RESUME_NEW (64).
# Output: bench/qfn/results/<date>-Mac15-6-<label>/ (hostinfo.mbp.txt env.txt command.txt
#   <arm>-<mtp>-c<chunk>.{log,guard.txt,pre.txt,post.txt}); with QFN_PAIR_K <date>-pair-<label>/ and the pair_serve.sh files.
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; shift
cases=($@)
(( $# )) || cases=(off:2048 spec:2048 off:12)
model=${QFN_MODEL:-$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2}
a=${QFN_RESUME_A:-$repo/scratch/qwen38/resume/tool.tokens}
b=${QFN_RESUME_B:-$repo/scratch/qwen38/resume/tool-u5.tokens}
new=${QFN_RESUME_NEW:-64}
bin=$repo/.build/release/TsugumiKernelCheck
[[ -x $bin ]] || { echo "no $bin"; exit 1; }
if pgrep -fl 'TsugumiServer|TsugumiMac|TsugumiDecodeService|TsugumiCLI|TsugumiToolLoopCheck|TsugumiKernelCheck|mlx_lm|llama-'; then
    echo "another model process is running here"; exit 1
fi
if env | grep -qE '^(Q38_|TF_)'; then echo "unset every Q38_/TF_ variable (the arms set their own)"; exit 1; fi
pairk=${QFN_PAIR_K:-0}
out=$repo/bench/qfn/results/$(date +%F)-Mac15-6-$label
(( pairk )) && out=$repo/bench/qfn/results/$(date +%F)-pair-$label
mkdir -p $out
sh $repo/bench/hostinfo.sh $bin > $out/hostinfo.mbp.txt
{ env | grep -E '^(QFN_)' || true; } > $out/env.txt
touch $out/command.txt
pairenv=()
if (( pairk )); then
    trap "$repo/bench/qfn/pair_serve.sh stop $out" EXIT
    $repo/bench/qfn/pair_serve.sh start $out
    pairenv=(Q38_PAIR=$(awk '/-> M6/ {print $NF}' $out/link.txt) Q38_PAIR_K=$pairk)
fi
snap() { { date '+%H:%M:%S'; memory_pressure -Q | tail -1; vm_stat; } > $1; }
for c in $cases; do
    mtp=${c%%:*} chunk=${c##*:}
    for arm in split inchunk; do
        armenv=($pairenv)
        [[ $arm == inchunk ]] && armenv+=(Q38_CUT_INCHUNK=1)
        tag=$arm-$mtp-c$chunk
        check=($bin --qwen38-resume $a $b --q38-mtp $mtp --q38-chunk $chunk --q38-new $new
               --q38-gguf $model/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf
               --q38-ple $model/ple-bf16/Qwen3.8-Flash-Next-PLE-BF16.gguf)
        echo "$tag: env ${armenv[*]} GUARD_LOG=$tag.guard.txt caffeinate -i Scripts/qwen38/guarded.sh $tag.log ${check[*]}" >> $out/command.txt
        snap $out/$tag.pre.txt
        (( pairk )) && $repo/bench/qfn/pair_serve.sh snap $out/$tag.pre.m6.txt
        rc=0
        env $armenv GUARD_LOG=$out/$tag.guard.txt caffeinate -i $repo/Scripts/qwen38/guarded.sh $out/$tag.log "${check[@]}" || rc=$?
        snap $out/$tag.post.txt
        (( pairk )) && { $repo/bench/qfn/pair_serve.sh snap $out/$tag.post.m6.txt; $repo/bench/qfn/pair_serve.sh alive || { echo "the M6 serve ended"; exit 1; } }
        echo "$tag: exit $rc"; grep -E 'FAIL|same|DIFFERENT|checkpoints at|GUARD' $out/$tag.log || true
        sleep ${COOL_S:-20}
    done
done
