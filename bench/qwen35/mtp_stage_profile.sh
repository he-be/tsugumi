#!/bin/zsh
# Stage-by-stage attribution of the 1.21x that `36-MTP-DECODE.md` §4-4 left
# unconfirmed: the same cut on plain decode, on the width-1 T-row control, and
# on the width-2 verify pass. One GPU process at a time, cooldown between runs.
set -e
CLI=.build/release/TurboFieldfareCLI
MODEL=scratch/ornith-oq4e-g64.gturbo
OUT=${OUT:-scratch/qwen35/mtp38}
TASK=${TASK:-t2-code}
MAXNEW=${MAXNEW:-64}
COOL=${COOL:-10}
SUFFIX=${SUFFIX:-}
mkdir -p $OUT
# name, "ENV=1 ENV=2", "--flags"
run() {
  local name=$1 envs=$2 flags=$3
  echo "### $name"
  env TF_QWEN_STAGE_PROFILE=1 ${=envs} \
    $CLI --model $MODEL --messages-file bench/qwen35/$TASK.json \
         --temperature 0 --repetition-penalty 1 --thinking off \
         --max-new $MAXNEW ${=flags} \
    > $OUT/$TASK.$name$SUFFIX.log 2>&1
  grep -E "^\[" $OUT/$TASK.$name$SUFFIX.log > $OUT/$TASK.$name$SUFFIX.footer || true
  sleep $COOL
}
# `qblock` arms put the prompt's query-blocked attention back in the verify
# pass — the shape `36-MTP-DECODE.md` measured, kept as the control for the
# split-KV rows kernel that is now the default.
ARMS=${ARMS:-"base width1 width2"}
for arm in ${=ARMS}; do
  case $arm in
    base)         run base         "" "" ;;
    width1)       run width1       "TF_QWEN_MTP_NO_DRAFT=1" "--qwen-mtp" ;;
    width2)       run width2       "" "--qwen-mtp" ;;
    width1qblock) run width1qblock "TF_QWEN_MTP_NO_DRAFT=1 TF_QWEN_MTP_ROWS_ATTN=0" "--qwen-mtp" ;;
    width2qblock) run width2qblock "TF_QWEN_MTP_ROWS_ATTN=0" "--qwen-mtp" ;;
    # `39-VERIFY-PREFETCH.md`: the residency-set commit off the calling thread,
    # and the arm that drops it altogether (the control that says whether the
    # background commit is buying anything, `27-PHASE6-THROUGHPUT.md` §9-2 ②).
    baseasync)    run baseasync    "TF_EXPERT_MMAP_RESIDENCY_ASYNC=1" "" ;;
    width2async)  run width2async  "TF_EXPERT_MMAP_RESIDENCY_ASYNC=1" "--qwen-mtp" ;;
    basenores)    run basenores    "TF_EXPERT_MMAP_RESIDENCY=0" "" ;;
    width2nores)  run width2nores  "TF_EXPERT_MMAP_RESIDENCY=0" "--qwen-mtp" ;;
    width2pf)     run width2pf     "TF_QWEN_MTP_PREFETCH=4" "--qwen-mtp" ;;
  esac
done
