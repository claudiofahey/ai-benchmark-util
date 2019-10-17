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
    parser.add_argument('--data_dir', action='append')
    args, unknown_args = parser.parse_known_args()

    cmd = unknown_args

    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

    i = (rank % len(args.data_dir))
    cmd += ['--data_dir=%s' % args.data_dir[i]]

    print('round_robin_mpi.py: ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
