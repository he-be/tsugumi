#!/bin/zsh
# The M6 side of the Qwen3.8 pair (docs/qfn-prefill/22 §4): TsugumiKernelCheck --qwen38-pair-serve on the M6's bridge0,
# under Scripts/qwen38/guarded.sh with caffeinate -i and a GUARD_LOG. Used by q6_check.sh and pair_bench.sh --pair.
#   bench/qfn/pair_serve.sh start <out dir> [VAR=value ...]   # waits for "listening"; <out>/hostinfo.m6.txt, link.txt
#   bench/qfn/pair_serve.sh stop <out dir>                    # stops the serve it started, copies its log and guard log
#   bench/qfn/pair_serve.sh alive                             # exit 0 while the serve runs
#   bench/qfn/pair_serve.sh snap <file>                       # the M6's vm_stat, memory_pressure and iostat (disk0 / disk4)
# QFN_M6_MODEL (default ~/dev/tsugumi/scratch/Qwen3.8-Flash-Next-DS4-IQ2 on the M6). The serve's defaults are the Q6
# shape (P1, 10 resident layers, 2 slabs, feed 0.59); VAR=value pairs change them (they go to env.m6.txt).
set -eu
repo=${0:A:h:h:h}
cmd=${1:?start|stop|alive|snap}; shift
m6model=${QFN_M6_MODEL:-'$HOME/dev/tsugumi/scratch/Qwen3.8-Flash-Next-DS4-IQ2'}
gg=Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf; pl=ple-bf16/Qwen3.8-Flash-Next-PLE-BF16.gguf
m6ip() { ssh m6 "ifconfig bridge0 | awk '/inet / {print \$2}'"; }
case $cmd in
alive) ssh m6 'pgrep -f "TsugumiKernelCheck --qwen38-pair-serve" > /dev/null' ;;
snap) ssh m6 'date "+%H:%M:%S"; memory_pressure -Q | tail -1; vm_stat; iostat -d -I disk0 disk4' > ${1:?file} ;;
start)
    out=${1:?out dir}; shift
    if ssh m6 'pgrep -fl "TsugumiKernelCheck|TsugumiServer|mlx_lm|llama-"'; then echo "a model process runs on the M6"; exit 1; fi
    ip=$(m6ip)
    [[ -n $ip ]] || { echo "the M6 has no bridge0 address"; exit 1; }
    ssh m6 'cd dev/tsugumi && sh bench/hostinfo.sh .build/release/TsugumiKernelCheck; echo "iogpu.wired_limit_mb $(sysctl -n iogpu.wired_limit_mb)"; mount | grep SSD256' > $out/hostinfo.m6.txt
    grep -q SSD256 $out/hostinfo.m6.txt || { echo "SSD256 is not mounted on the M6 (the PLE BF16 is there)"; exit 1; }
    print -l -- "$@" > $out/env.m6.txt
    serve="TsugumiKernelCheck --qwen38-pair-serve $ip 5310 --q38-gguf $m6model/$gg --q38-ple $m6model/$pl"
    echo "M6: cd dev/tsugumi && env $* GUARD_LOG=scratch/pair-serve.guard.txt caffeinate -i Scripts/qwen38/guarded.sh scratch/pair-serve.log .build/release/$serve" >> $out/command.txt
    ssh m6 "cd dev/tsugumi && rm -f scratch/pair-serve.log scratch/pair-serve.guard.txt && nohup env $* GUARD_LOG=scratch/pair-serve.guard.txt caffeinate -i Scripts/qwen38/guarded.sh scratch/pair-serve.log .build/release/$serve > /dev/null 2>&1 &"
    for i in {1..90}; do
        ssh m6 'grep -qE "^listening|^GUARD|rror" dev/tsugumi/scratch/pair-serve.log 2>/dev/null' && break
        sleep 2
    done
    ssh m6 'cat dev/tsugumi/scratch/pair-serve.log'
    ssh m6 'grep -q "^listening" dev/tsugumi/scratch/pair-serve.log' || { echo "the serve did not start"; exit 1; }
    { echo "bridge0 MBP $(ifconfig bridge0 | awk '/inet / {print $2}') -> M6 $ip"; ping -c 5 -q $ip | tail -1; } > $out/link.txt ;;
stop)
    out=${1:?out dir}
    ssh m6 'pkill -f "TsugumiKernelCheck --qwen38-pair-serve"' || true
    for i in {1..30}; do $0 alive || break; sleep 1; done
    sleep 2
    ssh m6 'cat dev/tsugumi/scratch/pair-serve.log' > $out/serve.m6.log || true
    ssh m6 'cat dev/tsugumi/scratch/pair-serve.guard.txt' > $out/serve.m6.guard.txt || true ;;
*) echo "start|stop|alive|snap"; exit 1 ;;
esac
