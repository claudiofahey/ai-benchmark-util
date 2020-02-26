#!/bin/bash

num_copies=7
out_dir="/mnt/filestore1/data/imagenet-scratch/tfrecords-${num_copies}x"

mkdir -p "${out_dir}"

time mpirun \
--allow-run-as-root \
-np 8 \
-H dl-worker9:16 \
-H dl-worker10:16 \
-bind-to none \
--map-by node \
-mca plm_rsh_agent ssh \
-mca plm_rsh_args "-p 2222" \
expand_tfrecords_mpi.py \
-i /mnt/filestore1/data/imagenet-scratch/tfrecords \
-o ${out_dir} \
--num_copies ${num_copies}
