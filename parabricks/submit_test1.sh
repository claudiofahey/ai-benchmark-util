#!/usr/bin/env bash
set -ex
cmd="sbatch --gres gpu:4 ./parabricks_germline_pipeline_slurm.py --sample_id"
scancel -u dgxuser
for i in {1..13}; do
    ${cmd} sample${i}
done
sleep 5s
squeue
sleep 5s
tail -f *.out
