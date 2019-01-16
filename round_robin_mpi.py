#!/usr/bin/env python3
#
# Copyright (c) 2018 Dell Inc., or its subsidiaries. All Rights Reserved.
# DELL CONFIDENTIAL AND PROPRIETARY.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
This script can be called first by mpirun to change the data_dir parameter passed to tf_cnn_benchmarks.py.
This is useful to use different mount points.
"""

import argparse
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(description='')
    args, unknown_args = parser.parse_known_args()

    cmd = unknown_args

    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

    i = 1 + (rank % 4)
    cmd += ['--data_dir=/mnt/isilon%d/data/imagenet-scratch/tfrecords' % i]

    print('round_robin_mpi.py: ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
