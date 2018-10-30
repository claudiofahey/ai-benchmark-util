#!/usr/bin/env bash
python -u ./run_benchmark.py \
--model resnet50 \
-np 72 \
-npernode 8 \
-H DGX1-1 \
-H DGX1-2 \
-H DGX1-3 \
-H DGX1-4 \
-H DGX1-5 \
-H DGX1-6 \
-H DGX1-7 \
-H DGX1-8 \
-H DGX1-9 \
|& tee -a /imagenet-scratch/logs/run_benchmark.log
