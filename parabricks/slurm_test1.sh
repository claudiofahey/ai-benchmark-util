#!/usr/bin/env bash
echo slurm_test1.sh: $(hostname): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}: BEGIN
sleep 10s
echo slurm_test1.sh: $(hostname): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}: END
