#!/usr/bin/env python3
#
# Copyright (c) 2018 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
This script starts the TensorFlow Docker container on multiple hosts.
"""


import argparse
import subprocess
from multiprocessing import Pool
import functools

def start_container(args, host):
    cmd = [
        'ssh',
        '-p', '22',
        host,
        'docker', 'stop', args.container_name,
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=False)

    cmd = [
        'ssh',
        '-p', '22',
        host,
        'nvidia-docker', 'run', '--rm', '--detach', '--privileged',
        '-v', '%s:/scripts' % args.scripts_dir,
        '-v', '%s:/tensorflow-benchmarks' % args.benchmarks_dir,
        '-v', '%s:/imagenet-data:ro' % args.imagenet_data_dir,
        '-v', '%s:/imagenet-scratch' % args.imagenet_scratch_dir,
        ]
    for i in range(1, 16+1):
        cmd += ['-v', '/mnt/isilon%d/data/imagenet-scratch:/imagenet-scratch%d' % (i, i)]
    cmd += [
        '--network=host',
        '--shm-size=1g', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
        '--name', args.container_name,
        'claudiofahey/tensorflow:18.09-py3-custom',
        'bash', '-c', '"/usr/sbin/sshd ; sleep infinity"',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def start_containers(args):
    with Pool(16) as p:
        p.map(functools.partial(start_container, args), args.host)


def main():
    parser = argparse.ArgumentParser(description='Start Docker containers for TensorFlow benchmarking')
    parser.add_argument('-H','--host', action='append', required=True,
                        help='List of hosts on which to invoke processes.')
    parser.add_argument('--scripts_dir', action='store', required=True,
                        help='Fully qualified path to the scripts directory.')
    parser.add_argument('--benchmarks_dir', action='store', required=True,
                        help='Fully qualified path to the TensorFlow Benchmarks directory.')
    parser.add_argument('--imagenet_data_dir', action='store', required=True,
                        help='Fully qualified path to the directory containing the original ImageNet data.')
    parser.add_argument('--imagenet_scratch_dir', action='store', required=True,
                        help='Fully qualified path to the ImageNet scratch directory.')
    parser.add_argument('--container_name', action='store', default='tf',
                        help='Name to assign to the containers.')
    args = parser.parse_args()
    start_containers(args)


if __name__ == '__main__':
    main()
