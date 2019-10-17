#!/usr/bin/env bash
set -ex
python -u ./run_benchmark.py $* \
|& tee -a /imagenet-scratch/logs/run_benchmark.log
