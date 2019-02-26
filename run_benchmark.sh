#!/usr/bin/env bash
python -u ./run_benchmark.py \
--isilon_host 127.0.0.1 \
--noflush \
--model resnet50 \
-np 1 \
-npernode 1 \
-H localhost \
|& tee -a /imagenet-scratch/logs/run_benchmark.log
