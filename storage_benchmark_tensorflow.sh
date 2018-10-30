#!/bin/bash
set -ex

HOSTS="\
-H dgx1-1:80 \
-H dgx1-2:80 \
-H dgx1-3:80 \
-H dgx1-4:80 \
-H dgx1-5:80 \
-H dgx1-6:80 \
-H dgx1-7:80 \
-H dgx1-8:80 \
-H dgx1-9:80 \
"

# Flush Isilon
ssh -p 22 root@10.55.67.211 isi_for_array isi_flush

# Flush Linux
mpirun \
--allow-run-as-root \
${HOSTS} \
-npernode 1 \
-mca plm_rsh_agent ssh \
-mca plm_rsh_args "-p 2222" \
./drop_caches.sh

subdir="tfrecords150/train-*"

mpirun \
--allow-run-as-root \
-np 126 \
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
 -i "/imagenet-scratch1/${subdir}" \
 -i "/imagenet-scratch2/${subdir}" \
 -i "/imagenet-scratch3/${subdir}" \
 -i "/imagenet-scratch4/${subdir}" \
 -i "/imagenet-scratch5/${subdir}" \
 -i "/imagenet-scratch6/${subdir}" \
 -i "/imagenet-scratch7/${subdir}" \
 -i "/imagenet-scratch8/${subdir}" \
 -i "/imagenet-scratch9/${subdir}" \
 -i "/imagenet-scratch10/${subdir}" \
 -i "/imagenet-scratch11/${subdir}" \
 -i "/imagenet-scratch12/${subdir}" \
 -i "/imagenet-scratch13/${subdir}" \
 -i "/imagenet-scratch14/${subdir}" \
 -i "/imagenet-scratch15/${subdir}" \
 -i "/imagenet-scratch16/${subdir}" \
|& tee -a /imagenet-scratch/logs/storage_benchmark_tensorflow.log
