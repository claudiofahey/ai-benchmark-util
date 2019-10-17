#!/bin/bash
set -ex

HOSTS="\
-H dgx2-1:80 \
-H dgx2-2:80 \
-H dgx2-3:80 \
"

subdir="tfrecords/train-*"

mpirun \
--allow-run-as-root \
-np 48 \
${HOSTS} \
--map-by node \
--bind-to socket \
--report-bindings \
-mca plm_rsh_agent ssh \
-mca plm_rsh_args "-p 2222" \
-mca pml ob1 \
-mca btl ^openib \
-mca btl_tcp_if_exclude lo,docker0 \
-x LD_LIBRARY_PATH \
-x PATH \
-x CUDA_VISIBLE_DEVICES="" \
python -u \
./storage_benchmark_tensorflow.py \
 -i "/mnt/isilon1/data/imagenet-scratch/${subdir}" \
|& tee -a /imagenet-scratch/logs/storage_benchmark_tensorflow.log
