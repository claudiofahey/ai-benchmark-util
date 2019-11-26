#!/usr/bin/env python3
#
# Copyright (c) Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""Start Dask containers, flush caches, run benchmark in container"""

import configargparse
import logging
import os
import subprocess
import sys


def parse_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def flush_caches(args):
    # Flush Isilon cache.
    if args.flush_isilon:
        cmd = [
            'ssh',
            '%s@%s' % (args.isilon_user, args.isilon_host),
            'isi_for_array', 'isi_flush',
        ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

    # Drop caches on all compute nodes.
    if args.flush_compute:
        drop_caches_file_name = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'drop_caches.sh')
        assert os.path.isfile(drop_caches_file_name)
        for host in args.host:
            cmd = [
                'ssh',
                '%s@%s' % (args.user, host),
                drop_caches_file_name,
            ]
            print(' '.join(cmd))
            subprocess.run(cmd, check=True)


def run_benchmark_driver(args, unknown_args):
    logging.info('run_benchmark_driver: BEGIN')

    start_dask_cmd = [
        './start_dask.py',
        '--container_name', args.container_name,
        '--device_memory_limit_gib', str(args.device_memory_limit_gib),
        '--docker_image', args.docker_image,
        '--memory_limit_gib', str(args.memory_limit_gib),
        '--start_notebook', str(False),
    ]
    start_dask_cmd += [a for h in args.host for a in ['--host', h]]
    print(' '.join(start_dask_cmd))
    subprocess.run(start_dask_cmd, check=True)

    flush_caches(args)

    host = args.benchmark_host
    container_name = args.container_name + '-driver'
    cmd = [
        'ssh',
        '-p', '22',
        '%s@%s' % (args.user, host),
        'nvidia-docker',
        'run',
        '--rm',
        '-v', '/mnt:/mnt',
        # '--network=host',
        '--name', container_name,
        args.docker_image,
        '/mnt/isilon/data/tf-bench-util/rapids/rapids_benchmark.py',
        '--num_workers', str(args.num_workers),
    ]
    cmd += unknown_args
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

    stop_dask_cmd = start_dask_cmd + ['--start', str(False)]
    print(' '.join(stop_dask_cmd))
    subprocess.run(stop_dask_cmd, check=True)

    logging.info('run_benchmark_driver: END')


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser = configargparse.ArgParser(
        description='Flush caches, start the Dask cluster, and run the RAPIDS benchmark',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument('--benchmark_host', default='')
    parser.add_argument('--config', '-c', required=False, is_config_file=True, help='config file path')
    parser.add_argument('--container_name', action='store', default='rapids-benchmark',
                        help='Name to assign to the containers.')
    parser.add_argument('--device_memory_limit_gib', type=float, default=26.0)
    parser.add_argument('--docker_image', action='store',
                        default='nvcr.io/nvidia/rapidsai/rapidsai:0.10-cuda10.0-runtime-ubuntu18.04',
                        help='Docker image tag.')
    parser.add_argument('--flush_compute', type=parse_bool, default=False, help='Flush compute caches')
    parser.add_argument('--flush_isilon', type=parse_bool, default=False, help='Flush Isilon caches')
    parser.add_argument('--host', '-H', action='append', required=True, help='List of hosts on which to invoke processes.')
    parser.add_argument('--isilon_host',
               help='IP address or hostname of an Isilon node. You must enable password-less SSH.')
    parser.add_argument('--isilon_user', default='root',
               help='SSH user used to connect to Isilon.')
    parser.add_argument('--memory_limit_gib', type=float, default=64.0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_level', type=int, default=logging.INFO, help='10=DEBUG,20=INFO')
    parser.add_argument('--user', action='store',
                        default='root', help='SSH user')
    args, unknown_args = parser.parse_known_args()

    os.chdir(script_dir)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=args.log_level, stream=sys.stdout)

    if args.benchmark_host == '':
        args.benchmark_host = args.host[0]

    logging.info('args=%s' % str(args))
    logging.info('unknown_args=%s' % str(unknown_args))

    run_benchmark_driver(args, unknown_args)


if __name__ == '__main__':
    main()
