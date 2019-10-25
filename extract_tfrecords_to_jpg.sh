#!/bin/bash

input_files="/imagenet-scratch/tfrecords1729/train-*"
output_dir="/imagenet-scratch/train1729"
mkdir -p "${output_dir}"

export CUDA_VISIBLE_DEVICES=""

time mpirun \
--allow-run-as-root \
-np 48 \
-H dgx2-1:96 \
-H dgx2-2:96 \
-H dgx2-3:96 \
-bind-to none \
--map-by node \
-mca plm_rsh_agent ssh \
-mca plm_rsh_args "-p 2222" \
-x LD_LIBRARY_PATH \
-x PATH \
-x CUDA_VISIBLE_DEVICES \
extract_tfrecords_to_jpg_mpi.py \
-i "${input_files}" \
-o "${output_dir}"
