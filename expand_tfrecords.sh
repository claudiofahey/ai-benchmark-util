#!/bin/bash

num_copies=150
out_dir="/imagenet-scratch/tfrecords-${num_copies}x"

mkdir -p "${out_dir}"

time mpirun \
--allow-run-as-root \
-np 72 \
-H DGX1-1:8 \
-H DGX1-2:8 \
-H DGX1-3:8 \
-H DGX1-4:8 \
-H DGX1-5:8 \
-H DGX1-6:8 \
-H DGX1-7:8 \
-H DGX1-8:8 \
-H DGX1-9:8 \
-bind-to none \
--map-by node \
-mca plm_rsh_agent ssh \
-mca plm_rsh_args "-p 2222" \
expand_tfrecords_mpi.py \
-i /imagenet-scratch/tfrecords \
-o ${out_dir} \
--num_copies ${num_copies}
