#!/usr/bin/env bash
# Reads sample IDs from sample_ids.txt and submits a Slurm job for each sample.
set -ex

NUM_GPUS=${NUM_GPUS:-8}
LOG_DIR="/mnt/isilon/data/genomics/logs"

scancel -u dgxuser

# Flush caches
ssh root@172.28.10.151 isi_for_array isi_flush &
for h in dgx2-1 dgx2-2 dgx2-3; do
    ssh root@${h} /mnt/isilon/data/tf-bench-util/drop_caches.sh &
done
wait

cat $* | \
xargs -n 1 -P 0 -i \
sbatch \
--gres gpu:${NUM_GPUS} \
--job-name "{}" \
--output "${LOG_DIR}/{}.log" \
--verbose \
parabricks_germline_pipeline_slurm.py \
--sample_id "{}"

sleep 1s
squeue
sleep 5s

tail -f ${LOG_DIR}/*.log
