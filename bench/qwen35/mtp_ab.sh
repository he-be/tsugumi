#!/bin/zsh
# MTP の腕と素の decode を**交互に**、回ごとに順も入れ替えて回す
# (`docs/qwen35moe/32-NVMAI-ADOPT.md` §5-1 の作法)。採点なので temp 0、
# クールダウン 10 秒。
set -e
CLI=.build/release/TsugumiCLI
MODEL=scratch/ornith-oq4e-g64.moepack
OUT=${OUT:-scratch/qwen35/mtp-ab}
MAXNEW=${MAXNEW:-192}
REPS=${REPS:-2}
TASKS=${TASKS:-"t2-code a1-agent-edit a2-agent-tool t3-en-prose t4-summarize"}
# `mtpqb` is the MTP arm with the verify pass back on the prompt's query-blocked
# attention — the shape `36-MTP-DECODE.md` measured. Kept as an arm rather than
# a remembered number so the comparison is inside one thermal session
# (`38-MTP-VERIFY-PATH.md`).
ARMS=${ARMS:-"base mtp"}
mkdir -p $OUT
for task in ${=TASKS}; do
  for rep in $(seq 1 $REPS); do
    # Rotate the order every repetition so a monotone drift cannot favour one
    # arm (`32-NVMAI-ADOPT.md` §5-1).
    arms=$(python3 -c "
import sys
a=sys.argv[1].split(); r=int(sys.argv[2])
print(' '.join(a[r % len(a):] + a[:r % len(a)]))" "$ARMS" "$rep")
    for arm in ${=arms}; do
      envs=""
      flag=""
      # 既定は採点用の greedy。`*sample` の腕だけが公式推奨サンプラで走る
      # (`docs/qwen35moe/42-SAMPLING.md` S1)。同じ種を渡すので、腕の中では
      # 再現する。
      samp="--temperature 0"
      case $arm in
        mtp)   flag="--qwen-mtp" ;;
        # 42 §4 段取り 6: サーバーが実際に走らせる腕。
        mtpsample)  flag="--qwen-mtp"
                    samp="--temperature 0.6 --top-k 20 --top-p 0.95 --seed 42" ;;
        basesample) samp="--temperature 0.6 --top-k 20 --top-p 0.95 --seed 42" ;;
        mtpqb) flag="--qwen-mtp"; envs="TF_QWEN_MTP_ROWS_ATTN=0" ;;
        # `39-VERIFY-PREFETCH.md`: the residency-set commit off the calling
        # thread (both loops), and the verify pass's cross-layer read-ahead.
        mtpasync)  flag="--qwen-mtp"; envs="TF_EXPERT_MMAP_RESIDENCY_ASYNC=1" ;;
        baseasync) envs="TF_EXPERT_MMAP_RESIDENCY_ASYNC=1" ;;
        # `27-PHASE6-THROUGHPUT.md` §9-2 ②: no residency set at all. The control
        # that says whether committing in the background buys anything over not
        # committing — `useResource` keeps both arms correct.
        basenores) envs="TF_EXPERT_MMAP_RESIDENCY=0" ;;
        mtpnores)  flag="--qwen-mtp"; envs="TF_EXPERT_MMAP_RESIDENCY=0" ;;
        mtppf)     flag="--qwen-mtp"; envs="TF_QWEN_MTP_PREFETCH=4" ;;
        mtppfasync) flag="--qwen-mtp"; envs="TF_EXPERT_MMAP_RESIDENCY_ASYNC=1 TF_QWEN_MTP_PREFETCH=4" ;;
      esac
      echo "### $task rep$rep $arm"
      env ${=envs} $CLI --model $MODEL --messages-file bench/qwen35/$task.json \
           ${=samp} --repetition-penalty 1 --thinking off \
           --max-new $MAXNEW ${=flag} \
           --dump-tokens $OUT/$task.$arm.$rep.json 2>&1 \
        | grep -E "^\[" | tee $OUT/$task.$arm.$rep.footer
      sleep ${COOLDOWN:-10}
    done
  done
done
