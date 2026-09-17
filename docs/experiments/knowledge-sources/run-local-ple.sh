#!/bin/zsh
# PLE table A/B on this Mac (docs/qwen38/31): the Serper-free stages (bare = stage 1, offline = stage 2) with the local
# Qwen3.8 runner, Q4_1 table vs BF16 table, arms alternated in ABBA order per repeat.
#
#   docs/experiments/knowledge-sources/run-local-ple.sh <model-dir-q41> <model-dir-bf16> [repeats]
#
# Each model dir holds only manifest.json (absolute paths; only "ple" differs) and a tokenizer symlink, so the real
# install's manifest is not touched. Every run is its own process under Scripts/qwen38/guarded.sh.
# Results: scratch/knowledge-sources/qwen-local/<arm>/<condition>/rep<N>/ (turns.jsonl, rounds.jsonl, run.log, guard.log).
set -u
cd "$(dirname "$0")/../../.."
q41=${1:?q41 model dir}; bf16=${2:?bf16 model dir}; repeats=${3:-3}
root=scratch/knowledge-sources/qwen-local
kdir=docs/experiments/knowledge-sources
cool=${COOL_S:-20}
bin=.build/release/TsugumiToolLoopCheck

one() {  # arm condition rep
    local arm=$1 cond=$2 rep=$3 dir flags questions out
    [[ $arm == q41 ]] && dir=$q41 || dir=$bf16
    case $cond in
        bare) flags=(--network model); questions=$kdir/questions-v2-stage1.json ;;
        offline) flags=(--network offline --max-rounds 6); questions=$kdir/questions-v2-stage2.json ;;
    esac
    out=$root/$arm/$cond/rep$rep
    if [[ -s $out/turns.jsonl ]]; then echo "skip $out"; return; fi
    mkdir -p $out
    while pgrep -x TsugumiToolLoopCheck > /dev/null; do echo "waiting for a previous TsugumiToolLoopCheck"; sleep 5; done
    echo "[$(date '+%F %T')] start $arm/$cond/rep$rep  swapouts $(vm_stat | awk '/Swapouts/ {print $2}')"
    GUARD_LOG=$out/guard.log caffeinate -i Scripts/qwen38/guarded.sh $out/run.log \
        $bin --out $out --model $dir --conversations $questions "${flags[@]}" \
        --thinking off --context 32768
    echo "[$(date '+%F %T')] end   $arm/$cond/rep$rep  exit $?  $(grep -c . $out/turns.jsonl 2>/dev/null) turns  $(grep -h GUARD $out/run.log)"
    sleep $cool
}

for rep in $(seq 1 $repeats); do
    if (( rep % 2 )); then order=(q41 bf16 bf16 q41); else order=(bf16 q41 q41 bf16); fi
    one $order[1] bare $rep; one $order[2] bare $rep
    one $order[3] offline $rep; one $order[4] offline $rep
done
echo "[$(date '+%F %T')] all done"
