#!/bin/bash
set -ex

FLUSH=false
INSTALL=true

HOSTS="\
-H dgx2-1:80 \
-H dgx2-2:80 \
-H dgx2-3:80 \
"

if [[ "$FLUSH" = true ]] ; then
    # Flush Isilon
    ssh -p 22 root@172.28.10.151 isi_for_array isi_flush
    # Flush Linux
    mpirun \
    --allow-run-as-root \
    ${HOSTS} \
    -npernode 1 \
    -mca plm_rsh_agent ssh \
    -mca plm_rsh_args "-p 2222" \
    ../drop_caches.sh
fi

# Install Python libraries
if [[ "$INSTALL" = true ]] ; then
    mpirun \
    --allow-run-as-root \
    ${HOSTS} \
    -npernode 1 \
    -mca plm_rsh_agent ssh \
    -mca plm_rsh_args "-p 2222" \
    pip install --user --requirement requirements.txt
fi

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
./storage_benchmark_tensorflow.py $* \
|& tee -a /imagenet-scratch/logs/storage_benchmark_tensorflow.log
