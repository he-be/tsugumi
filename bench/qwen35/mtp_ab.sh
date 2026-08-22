#!/bin/zsh
# MTP の腕と素の decode を**交互に**、回ごとに順も入れ替えて回す
# (`docs/qwen35moe/32-NVMAI-ADOPT.md` §5-1 の作法)。採点なので temp 0、
# クールダウン 10 秒。
set -e
CLI=.build/release/TurboFieldfareCLI
MODEL=scratch/ornith-oq4e-g64.gturbo
OUT=${OUT:-scratch/qwen35/mtp-ab}
MAXNEW=${MAXNEW:-192}
REPS=${REPS:-2}
TASKS=${TASKS:-"t2-code a1-agent-edit a2-agent-tool t3-en-prose t4-summarize"}
mkdir -p $OUT
for task in ${=TASKS}; do
  for rep in $(seq 1 $REPS); do
    if [ $((rep % 2)) -eq 1 ]; then arms="base mtp"; else arms="mtp base"; fi
    for arm in ${=arms}; do
      flag=""
      [ "$arm" = "mtp" ] && flag="--qwen-mtp"
      echo "### $task rep$rep $arm"
      $CLI --model $MODEL --messages-file bench/qwen35/$task.json \
           --temperature 0 --repetition-penalty 1 --thinking off \
           --max-new $MAXNEW $flag \
           --dump-tokens $OUT/$task.$arm.$rep.json 2>&1 \
        | grep -E "^\[" | tee $OUT/$task.$arm.$rep.footer
      sleep 10
    done
  done
done
