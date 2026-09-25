#!/bin/zsh
# E4-(a) (docs/qfn-prefill/07): one layer of routed experts at T = 2048 through today's GEMM arms (per-pair kernels,
# dequant + MPS sgemm), on this host. Route files from `Q38_DUMP_ROUTE` (scratch/qfn/routes/c2048-l<L>.bin).
#   bench/qfn/e4_gemm.sh <label> [iterations] [layers...]     # env: QFN_MODEL, QFN_ROUTES
set -eu
repo=${0:A:h:h:h}
label=${1:?label}; it=${2:-5}; shift 2 2>/dev/null || shift $#
layers=(${@:-0 24 47})
model=${QFN_MODEL:-$HOME/LLM/Qwen3.8-Flash-Next-DS4-IQ2}
routes=${QFN_ROUTES:-$repo/scratch/qfn/routes}
host=$(sysctl -n hw.model | tr ',' '-')
out=$repo/bench/qfn/results/$(date +%F)-$host-$label
mkdir -p $out
bin=$repo/.build/release/TsugumiKernelCheck
sh $repo/bench/hostinfo.sh $bin > $out/hostinfo.txt
: > $out/command.txt
for L in $layers; do
  cmd=($bin --q2-gemm-bench $model/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf $routes/c2048-l$L.bin 4 $it 2048)
  echo "${cmd[*]}" >> $out/command.txt
  $repo/Scripts/qwen38/guarded.sh $out/layer-$L.log "${cmd[@]}" || true
  cat $out/layer-$L.log
  sleep ${COOL_S:-4}
done
echo "evidence: ${out#$repo/}"
