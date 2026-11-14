#!/bin/bash
# 英名 (item) だけを全 96 枚 × 5 回、通常サンプリングで引く。崩れ (上限 120 まで書いて JSON が閉じない) の率を見る
cd /root/diet
until grep -q EVAL_JOB_DONE iq3_job.log; do sleep 10; done
Q4=/root/models/gemma-4-26b-a4b-qat-2/Huihui-gemma-4-26B-A4B-it-qat-abliterated-lattice-Q4_0.gguf
for spec in full=$Q4 k112=/root/diet/lattice-keep_112.gguf k96=/root/diet/lattice-keep_96.gguf k80=/root/diet/lattice-keep_80.gguf k64=/root/diet/lattice-keep_64.gguf iq3xxs=/root/diet/gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf; do
  arm=${spec%%=*}; gguf=${spec#*=}
  ./serve.sh 9302 $gguf > serve-$arm.log 2>&1 &
  sp=$!
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:9302/health | grep -q ok && break; sleep 2; done
  /root/xolotl/.venv/bin/python run_arm.py http://127.0.0.1:9302 kj_payloads96.jsonl item_$arm.jsonl --only item --rounds 5
  kill $sp; wait $sp
done
echo ITEM_JOB_DONE
