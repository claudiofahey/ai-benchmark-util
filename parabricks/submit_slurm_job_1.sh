#!/usr/bin/env bash
# This script can be used to submit a single Parabricks job to Slurm.
set -ex

NUM_GPUS=${NUM_GPUS:-8}

SAMPLE_ID="SS6004472"

srun \
--accel-bind=gv \
--cpus-per-task 48 \
--cpu-bind verbose \
--gres gpu:${NUM_GPUS} \
--job-name "${SAMPLE_ID}" \
--ntasks 1 \
--verbose \
parabricks_germline_pipeline_slurm.py \
--germline false \
--deepvariant false \
--fq2bam true \
--sample_id "${SAMPLE_ID}"
