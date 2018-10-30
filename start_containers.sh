#!/usr/bin/env bash
./start_containers.py \
-H DGX1-1 \
-H DGX1-2 \
-H DGX1-3 \
-H DGX1-4 \
-H DGX1-5 \
-H DGX1-6 \
-H DGX1-7 \
-H DGX1-8 \
-H DGX1-9 \
--scripts_dir /mnt/isilon/data/tf-bench-util \
--benchmarks_dir /mnt/isilon/data/tensorflow-benchmarks \
--imagenet_data_dir /mnt/isilon/data/imagenet-data \
--imagenet_scratch_dir /mnt/isilon/data/imagenet-scratch
