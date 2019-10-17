#!/bin/bash
set -ex

HOSTS="\
-H dgx2-1:80 \
-H dgx2-2:80 \
-H dgx2-3:80 \
"

# Flush Isilon
ssh -p 22 root@172.28.10.151 isi_for_array isi_flush

# Flush Linux
mpirun \
--allow-run-as-root \
${HOSTS} \
-npernode 1 \
-mca plm_rsh_agent ssh \
-mca plm_rsh_args "-p 2222" \
./drop_caches.sh

subdir="tfrecords-150x/train-*"

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
 -i "/mnt/isilon2/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon4/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon5/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon6/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon7/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon8/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon9/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon10/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon11/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon13/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon14/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon15/data/imagenet-scratch/${subdir}" \
 -i "/mnt/isilon16/data/imagenet-scratch/${subdir}" \
|& tee -a /imagenet-scratch/logs/storage_benchmark_tensorflow.log
