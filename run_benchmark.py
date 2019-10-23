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

import configargparse
import subprocess
import datetime
import os
import time


def parse_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def run_tf_cnn_benchmarks(args, unknown_args):
    print('run_tf_cnn_benchmarks: BEGIN')
    print(datetime.datetime.utcnow())
    print('args=%s' % str(args))

    data_dir = args.data_dir + [args.data_dir_template % (i+1) for i in range(args.data_dir_template_count)]
    print('data_dir=%s' % str(data_dir))

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
            '--n', str(args.np),
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

    if args.noop:
        mpirun_cmd += ['/bin/echo']

    cmd = mpirun_cmd + [
        'python',
        '-u',
        '/tensorflow-benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py',
        '--model=%s' % args.model,
        '--batch_size=%d' % args.batch_size,
        '--batch_group_size=%d' % args.batch_group_size,
        '--num_batches=%d' % args.num_batches,
        '--nodistortions',
        '--num_gpus=1',
        '--device=gpu',
        '--force_gpu_compatible=True',
        '--data_format=NCHW',
        '--use_fp16=%s' % str(args.fp16),
        '--use_tf_layers=%s' % str(args.fp16),
        '--data_name=imagenet',
        '--use_datasets=True',
        '--num_intra_threads=%d' % args.num_intra_threads,
        '--num_inter_threads=%d' % args.num_inter_threads,
        '--datasets_prefetch_buffer_size=%d' % args.datasets_prefetch_buffer_size,
        '--datasets_num_private_threads=%d' % args.datasets_num_private_threads,
        '--train_dir=%s' % train_dir,
        '--sync_on_finish=True',
        ] + eval_cmd + horovod_parameters

    cmd += ['--data_dir=%s' % d for d in data_dir]
    cmd += unknown_args

    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    print('args=%s' % str(args))
    print(datetime.datetime.utcnow())
    print('run_tf_cnn_benchmarks: END')


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser = configargparse.ArgParser(
        description='Execute TensorFlow CNN benchmarks',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[
            os.path.join(script_dir, 'run_benchmark_defaults.yaml'),
            os.path.join(script_dir, 'run_benchmark_environment.yaml'),
            './suite_defaults.yaml',
        ],
    )
    parser.add('--batch_group_size', type=int, default=10)
    parser.add('--batch_size', type=int, default=256, help='Number of records per batch')
    parser.add('--config', '-c', required=False, is_config_file=True, help='config file path')
    parser.add('--data_dir', action='append', default=[])
    parser.add('--data_dir_template')
    parser.add('--data_dir_template_count', type=int, default=0)
    parser.add('--datasets_prefetch_buffer_size', type=int, default=20)
    parser.add('--datasets_num_private_threads', type=int, default=2)
    parser.add('--eval', type=parse_bool, default=False,
               help='Perform inference instead of training.')
    parser.add('--flush', type=parse_bool, default=False, help='Flush caches')
    parser.add('--fp16', type=parse_bool, default=True,  help='Use FP16, otherwise use FP32')
    parser.add('--forward_only', type=parse_bool, default=False,
               help='Perform inference instead of training.')
    parser.add('--host', '-H', action='append', required=True, help='List of hosts on which to invoke processes.')
    parser.add('--isilon_host',
               help='IP address or hostname of an Isilon node. You must enable password-less SSH.')
    parser.add('--isilon_user', default='root',
               help='SSH user used to connect to Isilon.')
    parser.add('--model', default='resnet50')
    parser.add('--mpi', type=parse_bool, default=True, help='Use MPI.')
    parser.add('--noop', type=parse_bool, default=False)
    parser.add('--np', type=int, default=1, help='Run this many copies of the program on the given nodes.')
    parser.add('--npernode', type=int, default=80, help='On each node, launch this many processes.')
    parser.add('--num_batches', type=int, default=500)
    parser.add('--num_hosts', type=int, default=0, help="If >0, use exactly this many hosts")
    parser.add('--num_intra_threads', type=int, default=1)
    parser.add('--num_inter_threads', type=int, default=40)
    parser.add('--repeat', type=int, default=1)
    parser.add('--run_id')
    parser.add('--train_dir', default='/imagenet-scratch/train_dir')
    args, unknown_args = parser.parse_known_args()

    os.chdir(script_dir)

    if args.num_hosts > 0:
        if args.num_hosts < len(args.host):
            args.host = args.host[0:args.num_hosts]
        elif args.num_hosts > len(args.host):
            raise Exception('Not enough hosts')

    for i in range(args.repeat):
        run_tf_cnn_benchmarks(args, unknown_args)


if __name__ == '__main__':
    main()
