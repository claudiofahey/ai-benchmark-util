#!/usr/bin/env bash
python -u ./run_benchmark.py \
--isilon_host 127.0.0.1 \
--noflush \
--model resnet50 \
-np 48 \
-npernode 16 \
-H dgx2-1 \
-H dgx2-2 \
-H dgx2-3 \
|& tee -a /imagenet-scratch/logs/run_benchmark.log
