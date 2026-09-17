#!/bin/zsh
# One model through the conditions of the knowledge-source experiment (README.md), on llama-swap.
#
#   docs/experiments/knowledge-sources/run.sh gemma model offline online offline-fixed online-fixed
#   ONLY=B1-kinosaki,H3-tozai-line docs/experiments/knowledge-sources/run.sh qwen online
#
# Conditions:
#   bare (model)   no tools
#   offline        local Wikipedia, up to 6 tool rounds (the app's default)
#   online         web + Wikipedia, up to 6 tool rounds (search and a page read forced first, as in the app)
#   offline-fixed  local Wikipedia, 1 tool round
#   online-fixed   web + Wikipedia, 2 tool rounds: the forced search and the forced page read, then the answer
#
# Results go to scratch/knowledge-sources/<model>/<condition>/ (turns.jsonl, rounds.jsonl, run.log). A condition whose
# turns.jsonl exists is skipped unless FORCE=1 (Serper is paid per query). With ONLY set, the run writes to
# <condition>-only-<timestamp>/ instead and never skips. REPEATS=N runs each question N times (reproducibility checks). The web store is per model and condition, so a rerun replays
# the same search results and pages instead of querying again.
set -u
cd "$(dirname "$0")/../../.."

model_key=${1:?usage: run.sh gemma|g12|qwen condition...}
shift
endpoint=${ENDPOINT:-http://192.168.1.9:8080}
case $model_key in
    gemma) remote=${REMOTE_MODEL:-gemma4-26b-a4b-mtp-1}; kind_dir=scratch/gemma4-qat-sym.gturbo ;;
    g12) remote=${REMOTE_MODEL:-gemma4-12b}; kind_dir=scratch/gemma4-qat-sym.gturbo ;;
    qwen) remote=${REMOTE_MODEL:-qwen3.8-flash-next-iq3-instruct}; kind_dir=$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2 ;;
    *) echo "unknown model $model_key" >&2; exit 2 ;;
esac
questions=${QUESTIONS:-docs/experiments/knowledge-sources/questions-v2.json}
# The app's context on every condition, whatever the server's slot holds (the Qwen slot is 98K).
context=${CONTEXT:-32768}
root=scratch/knowledge-sources

[[ -x .build/release/TsugumiToolLoopCheck ]] || swift build -c release --product TsugumiToolLoopCheck || exit 2

for condition in "$@"; do
    case $condition in
        bare|model) flags=(--network model) ;;
        offline) flags=(--network offline --max-rounds 6) ;;
        online) flags=(--network online --max-rounds 6) ;;
        offline-fixed) flags=(--network offline --max-rounds 1) ;;
        online-fixed) flags=(--network online --max-rounds 2) ;;
        *) echo "unknown condition $condition" >&2; exit 2 ;;
    esac
    out=$root/$model_key/$condition
    only=()
    if [[ -n ${ONLY:-} ]]; then
        out=$out-only-$(date +%Y%m%d-%H%M%S)
        only=(--only "$ONLY")
    elif [[ -s $out/turns.jsonl && ${FORCE:-0} != 1 ]]; then
        echo "skip $model_key/$condition: $out/turns.jsonl exists (FORCE=1 to rerun)"
        continue
    fi
    mkdir -p "$out"
    echo "run $model_key/$condition -> $out ($remote at $endpoint)"
    .build/release/TsugumiToolLoopCheck --out "$out" --model "$kind_dir" --conversations "$questions" \
        "${flags[@]}" "${only[@]}" --repeats "${REPEATS:-1}" --thinking off --context "$context" \
        --endpoint "$endpoint" --remote-model "$remote" \
        --web-store "$root/web-store/$model_key-$condition" \
        2>&1 | tee "$out/run.log" | grep --line-buffered -E "turn [0-9]+: |FAIL|passed|not usable|load failed"
done
