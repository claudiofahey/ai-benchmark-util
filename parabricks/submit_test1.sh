#!/usr/bin/env bash
set -ex
scancel -u dgxuser
for i in {1..1}; do
    sample_id="sample${i}"#
    sample_id="LP6005441-DNA_A10"
    sbatch \
    --gres gpu:4 \
    --job-name "${sample_id}-job" \
    --output "${sample_id}.out" \
    parabricks_germline_pipeline_slurm.py \
    --sample_id "${sample_id}"
done
sleep 1s
squeue
tail -f *.out
