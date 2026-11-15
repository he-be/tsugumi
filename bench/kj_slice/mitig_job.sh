#!/bin/bash
# 英名 (item) の崩れを減らす手を、1 つのサーバーに要求ごとのサンプリング値を足して比べる。
#   mitig_job.sh <dir> <gguf> <payloads> <tag>     (環境変数は serve.sh と同じ)
dir=$1; gguf=$2; payloads=$3; tag=$4
cd $dir
./serve.sh 9310 $gguf > serve-mitig-$tag.log 2>&1 &
sp=$!
for i in $(seq 150); do curl -s -m 2 http://127.0.0.1:9310/health | grep -q ok && break; sleep 2; done
run() { python3 run_arm.py http://127.0.0.1:9310 $payloads item_${tag}_$1.jsonl --only item --rounds 5 --extra "$2"; }
run base '{}'
run rp105 '{"repeat_penalty":1.05}'
run rp110 '{"repeat_penalty":1.1}'
run dry '{"dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":2}'
run temp07 '{"temperature":0.7}'
run topp09 '{"top_p":0.9}'
run pres05 '{"presence_penalty":0.5}'
kill $sp; wait $sp
echo MITIG_JOB_DONE
