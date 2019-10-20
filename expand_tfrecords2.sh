#!/bin/bash

num_copies=13
out_dir="/imagenet-scratch/tfrecords1729-${num_copies}x"

mkdir -p "${out_dir}"

time mpirun \
--allow-run-as-root \
-np 48 \
-H dgx2-1:16 \
-H dgx2-2:16 \
-H dgx2-3:16 \
-bind-to none \
--map-by node \
-mca plm_rsh_agent ssh \
-mca plm_rsh_args "-p 2222" \
expand_tfrecords_mpi.py \
-i /imagenet-scratch/tfrecords1729 \
-o ${out_dir} \
--num_copies ${num_copies}
