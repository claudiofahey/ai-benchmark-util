#!/usr/bin/env python3
#
# Copyright (c) 2018 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
This is a wrapper for tf_cnn_benchmarks.py.
It flushes caches and builds the command line for tf_cnn_benchmarks.py.
It supports training and inference modes.
"""

import argparse
import subprocess
import datetime
import os
import time

def run_tf_cnn_benchmarks(args, unknown_args):
    print('run_tf_cnn_benchmarks: BEGIN')
    print(datetime.datetime.utcnow())
    print('args=%s' % str(args))

    mpi_hosts = ','.join(['%s:%d' % (h, args.npernode) for h in args.host])
    num_hosts = len(args.host)

    if args.eval:
        run_id = args.run_id
        train_dir = os.path.join(args.train_dir, run_id)
        eval_cmd = [
            '--eval',
        ]
    elif args.forward_only:
        run_id = '%s-%s-forward_only' % (time.strftime("%Y-%m-%d-%H-%M-%S"), args.model)
        train_dir = os.path.join(args.train_dir, run_id)
        eval_cmd = [
            '--forward_only',
            '--summary_verbosity=1',
            '--save_summaries_steps=100',
        ]
    else:
        run_id = '%s-%s' % (time.strftime("%Y-%m-%d-%H-%M-%S"), args.model)
        train_dir = os.path.join(args.train_dir, run_id)
        eval_cmd = [
            '--summary_verbosity=1',
            '--save_summaries_steps=100',
            '--save_model_secs=600',
        ]

    # Flush Isilon cache.
    if args.flush:
        cmd = [
            'ssh',
            '-p', '22',
            '%s@%s' % (args.isilon_user, args.isilon_host),
            'isi_for_array', 'isi_flush',
        ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

    # Drop caches on all slave nodes.
    if args.flush:
        cmd = [
            'mpirun',
            '--n', str(num_hosts),
            '-npernode', '1',
            '-allow-run-as-root',
            '--host', mpi_hosts,
            '-mca', 'plm_rsh_agent', 'ssh',
            '-mca', 'plm_rsh_args', '"-p 2222"',
            '/scripts/drop_caches.sh',
        ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

    # Execute benchmark.

    mpirun_cmd = []
    horovod_parameters = []
    if args.mpi:
        mpirun_cmd = [
            'mpirun',
            '--n', str(args.n),
            '-allow-run-as-root',
            '--host', mpi_hosts,
            '--report-bindings',
            '-bind-to', 'none',
            '-map-by', 'slot',
            '-x', 'LD_LIBRARY_PATH',
            '-x', 'PATH',
            '-mca', 'plm_rsh_agent', 'ssh',
            '-mca', 'plm_rsh_args', '"-p 2222"',
            '-mca', 'pml', 'ob1',
            '-mca', 'btl', '^openib',
            '-mca', 'btl_tcp_if_include', 'enp53s0',         # force to use this TCP interface for MPI BTL
            '-x', 'NCCL_DEBUG=INFO',                         # Enable debug logging
            '-x', 'NCCL_IB_HCA=mlx5',                        # Assign the RoCE interface for NCCL - this allows all 4 NICs to be used
            '-x', 'NCCL_IB_SL=4',                            # InfiniBand Service Level
            '-x', 'NCCL_IB_GID_INDEX=3',                     # RoCE priority
            '-x', 'NCCL_NET_GDR_READ=1',                     # RoCE receive to memory directly
            '-x', 'NCCL_SOCKET_IFNAME=^docker0,lo',          # Do not let NCCL use docker0 interface. See https://github.com/uber/horovod/blob/master/docs/running.md#hangs-due-to-non-routed-network-interfaces.
            './round_robin_mpi.py',
        ]
        horovod_parameters = [
            '--variable_update=horovod',
            '--horovod_device=gpu',
        ]

    cmd = mpirun_cmd + [
        'python',
        '-u',
        '/tensorflow-benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py',
        '--model=%s' % args.model,
        '--batch_size=192',
        '--batch_group_size=10',
        # '--num_epochs=4',
        '--num_batches=5000',
        '--nodistortions',
        '--num_gpus=1',
        '--device=gpu',
        '--force_gpu_compatible=True',
        '--data_format=NCHW',
        '--use_fp16=True',  '--use_tf_layers=False',  # For Tensor Cores, fp16
        # '--use_fp16=False', '--use_tf_layers=True',   # For CUDA Cores, fp32
        '--data_name=imagenet',
        '--use_datasets=True',
        '--data_dir=/imagenet-scratch1/tfrecords',  # Note that this is overridden by round_robin_mpi.py
        '--num_intra_threads=1',
        '--num_inter_threads=40',
        '--datasets_prefetch_buffer_size=20',
        '--datasets_num_private_threads=2',
        '--train_dir=%s' % train_dir,
        '--sync_on_finish=True',
        ] + eval_cmd + horovod_parameters

    cmd += unknown_args

    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    print(datetime.datetime.utcnow())
    print('run_tf_cnn_benchmarks: END')


def main():
    parser = argparse.ArgumentParser(description='Execute TensorFlow CNN benchmarks')
    parser.add_argument('-np', '--n', action='store', type=int, default=1,
                        help='Run this many copies of the program on the given nodes.')
    parser.add_argument('-npernode', '--npernode', action='store', type=int, default=1,
                        help='On each node, launch this many processes.')
    parser.add_argument('-H', '--host', action='append', required=True,
                        help='List of hosts on which to invoke processes.')
    parser.add_argument('--noflush', action='store_false', dest='flush',
                        help='Do not flush caches.')
    parser.add_argument('--train_dir', action='store', default='/imagenet-scratch/train_dir')
    parser.add_argument('--nompi', action='store_false', dest='mpi',
                        help='Do not use MPI.')
    parser.add_argument('--eval', action='store_true',
                        help='Perform inference instead of training.')
    parser.add_argument('--forward_only', action='store_true',
                        help='Perform inference instead of training.')
    parser.add_argument('--model', action='store', default='resnet50')
    parser.add_argument('--run_id', action='store')
    parser.add_argument('--isilon_host', action='store',
                        help='IP address or hostname of an Isilon node. You must enable password-less SSH.')
    parser.add_argument('--isilon_user', action='store', default='root',
                        help='SSH user used to connect to Isilon.')
    args, unknown_args = parser.parse_known_args()
    run_tf_cnn_benchmarks(args, unknown_args)


if __name__ == '__main__':
    main()
