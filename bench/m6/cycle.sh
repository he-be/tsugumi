#!/bin/sh
# One MBP -> M6 cycle (docs/m6-prefill/01-ROLES.md §2): push HEAD, build on M6,
# run a bench/m6 script there, pull the evidence directory back.
#   bench/m6/cycle.sh prefill_bench.sh g0-baseline 6
#   bench/m6/cycle.sh run_probe.sh gpu_family_probe
#   TF_PREFILL_GPU_PROFILE=1 bench/m6/cycle.sh prefill_bench.sh g0-profile 3
# Refuses to run on a dirty tree: the commit stamp in hostinfo.txt must be honest.
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
script=${1:?bench/m6 script name}; shift
remote=${M6_HOST:-m6}
rdir=${M6_REPO:-dev/tsugumi}
if ! git -C "$repo" diff --quiet HEAD; then
    echo "working tree is dirty; commit first so the evidence carries a clean commit id"; exit 1
fi
git -C "$repo" push "$remote" HEAD
ssh "$remote" "cd $rdir && swift build -c release --product TsugumiCLI 2>&1 | tail -2"
envs=$(env | grep '^TF_\|^PROMPT_TOKENS=\|^MODEL=' | tr '\n' ' ' || true)
ssh "$remote" "cd $rdir && env $envs sh bench/m6/$script $*"
rsync -a "$remote:$rdir/bench/m6/results/" "$repo/bench/m6/results/"
echo "pulled into bench/m6/results/"
