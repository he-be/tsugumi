#!/bin/bash
cd /root/diet
until grep -q EVAL_JOB_DONE slice_job.log; do sleep 10; done
./eval_job.sh iq3xxs=/root/diet/gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf
