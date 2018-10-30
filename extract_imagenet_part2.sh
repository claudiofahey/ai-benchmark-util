#!/bin/bash
set -ex
mpirun --allow-run-as-root -np 16 extract_imagenet_part2_mpi.py -i /imagenet-scratch/train -o /imagenet-scratch/train
rm -rf /imagenet-scratch/train/*.tar
