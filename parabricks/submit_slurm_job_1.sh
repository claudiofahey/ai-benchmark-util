#!/usr/bin/env bash
set -ex

NUM_GPUS=${NUM_GPUS:-8}

SAMPLE_ID="SS6004472"

srun \
--accel-bind=gv \
--cpus-per-task 2 \
--cpu-bind verbose \
--gres gpu:${NUM_GPUS} \
--job-name "${SAMPLE_ID}" \
--ntasks 1 \
--unbuffered \
--verbose \
parabricks_germline_pipeline_slurm.py \
--germline false \
--deepvariant false \
--fq2bam true \
--sample_id "${SAMPLE_ID}"
