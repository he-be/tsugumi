#!/bin/zsh
# E2 (docs/qfn-prefill/07): cold read of one layer's routed experts (gate/up from the GGUF, down from the sidecar,
# 710 MB) on this host, 4 threads x 16 MB per group as GGUFFile.preadRanges. One layer per run, never one read before.
#   bench/qfn/e2_ssd.sh <label> <share> <layers...>     # share 0 = internal only, 1 = mirror only, 0.32 = both at once
#   env: QFN_MODEL, QFN_MIRROR (/Volumes/SSD256/qfn/mirror)
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; share=${2:?share}; shift 2
model=${QFN_MODEL:-$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2}
mirror=${QFN_MIRROR:-/Volumes/SSD256/qfn/mirror}
host=$(sysctl -n hw.model | tr ',' '-')
out=$repo/bench/qfn/results/$(date +%F)-$host-$label
mkdir -p $out
cc -O2 -o $out/../.range_read.$host $repo/bench/qfn/range_read.c
sh $repo/bench/hostinfo.sh > $out/hostinfo.txt
for L in $@; do
  E2_MAIN_RANGES=$repo/bench/qfn/expert-ranges.txt E2_DOWN_RANGES=$repo/bench/qfn/down-ranges.txt \
    python3 $repo/bench/qfn/e2_lists.py $model $mirror $L $share > $out/layer-$L.list
  echo "range_read layer-$L.list 4 16   # share $share" >> $out/command.txt
  { iostat -d -I; } > $out/layer-$L.pre.txt
  $out/../.range_read.$host $out/layer-$L.list 4 16 | tee $out/layer-$L.log
  { iostat -d -I; } > $out/layer-$L.post.txt
  sleep 2
done
rm -f $out/../.range_read.$host
echo "evidence: ${out#$repo/}"
