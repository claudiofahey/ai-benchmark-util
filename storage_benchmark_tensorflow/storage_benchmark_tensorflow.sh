#!/bin/bash
set -ex
python -u \
./storage_benchmark_tensorflow.py $* \
|& tee -a /imagenet-scratch/logs/storage_benchmark_tensorflow.log
