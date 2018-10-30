#!/bin/bash

output_dir="/imagenet-scratch/tfrecords1729"
mkdir -p "${output_dir}"

export CUDA_VISIBLE_DEVICES=""

time mpirun \
--allow-run-as-root \
-np 512 \
-H dgx1-1:80 \
-H dgx1-2:80 \
-H dgx1-3:80 \
-H dgx1-4:80 \
-H dgx1-5:80 \
-H dgx1-6:80 \
-H dgx1-7:80 \
-H dgx1-8:80 \
-H dgx1-9:80 \
-bind-to none \
--map-by node \
-mca plm_rsh_agent ssh \
-mca plm_rsh_args "-p 2222" \
-x LD_LIBRARY_PATH \
-x PATH \
-x CUDA_VISIBLE_DEVICES \
resize_tfrecords_mpi.py \
-i "/imagenet-scratch/tfrecords/train-*" \
-o ${output_dir}
