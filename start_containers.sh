#!/usr/bin/env bash
# This script starts the TensorFlow Docker container on multiple hosts.
# You may add multiple -H parameters, one for each hostname.

./start_containers.py \
-H localhost \
--scripts_dir /mnt/isilon/data/tf-bench-util \
--benchmarks_dir /mnt/isilon/data/tensorflow-benchmarks \
--imagenet_data_dir /mnt/isilon/data/imagenet-data \
--imagenet_scratch_dir /mnt/isilon/data/imagenet-scratch
