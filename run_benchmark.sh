#!/usr/bin/env bash
python -u ./run_benchmark.py \
--isilon_host 172.28.10.151 \
--noflush \
--model vgg16 \
-np 48 \
-npernode 16 \
-H dgx2-1 \
-H dgx2-2 \
-H dgx2-3 \
|& tee -a /imagenet-scratch/logs/run_benchmark.log
