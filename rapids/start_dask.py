#!/usr/bin/env python3
#
# Copyright (c) 2019 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
Start Dask cluster on multiple hosts using SSH and Docker.
"""


import configargparse
import subprocess
from multiprocessing import Pool
import functools


def stop_containers_on_host(args, host):
    """Stop any container that starts with container name."""
    cmd = [
        'ssh',
        '-p', '22',
        '%s@%s' % (args.user, host),
        'bash', '-c',
        '"docker ps --format {{.Names}} | grep ^%s- | xargs -i docker stop {}"' % args.container_name
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=False)


def stop_containers(args):
    with Pool(16) as p:
        p.map(functools.partial(stop_containers_on_host, args), args.host)


def init_host(args, host):
    cmd = [
        'ssh',
        '-p', '22',
        '%s@%s' % (args.user, host),
        'nvidia-docker',
        'pull',
        args.docker_image,
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def init_hosts(args):
    with Pool(16) as p:
        p.map(functools.partial(init_host, args), args.host)


def start_scheduler(args):
    host = args.scheduler_host
    container_name = args.container_name + '-scheduler'
    cmd = [
        'ssh',
        '-p', '22',
        '%s@%s' % (args.user, host),
        'nvidia-docker',
        'run',
        '--rm',
        '--detach',
        '-v', '/mnt:/mnt',
        '--network=host',
        '--name', container_name,
        '--entrypoint', '/usr/bin/tini',     # do not run Jupyter Notebook
        args.docker_image,
        '--',
        'bash', '-c',
        '"source activate rapids && dask-scheduler"',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def start_notebook(args):
    host = args.scheduler_host
    container_name = args.container_name + '-notebook'
    cmd = [
        'ssh',
        '-p', '22',
        '%s@%s' % (args.user, host),
        'nvidia-docker',
        'run',
        '--rm',
        '--detach',
        '-v', '/mnt:/rapids/notebooks/mnt',
        '-v', '/mnt:/mnt',
        '--network=host',
        '--name', container_name,
        '--entrypoint', '/usr/bin/tini',     # do not run Jupyter Notebook
        args.docker_image,
        '--',
        'bash', '-c',
        '"source activate rapids && jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=\'\'"',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def start_worker(args, host):
    container_name = args.container_name + '-worker'
    dask_worker_cmd = [
        'dask-cuda-worker',
        '--nthreads', '5',
        '--memory-limit', '%d' % int(args.memory_limit_gib * 1024**3),
        '--device-memory-limit', '%d' % int(args.device_memory_limit_gib * 1024**3),
        '--local-directory', '/dask-local-directory',
        '%s:8786' % args.scheduler_host,
    ]
    cmd = [
        'ssh',
        '-p', '22',
        '%s@%s' % (args.user, host),
        'nvidia-docker',
        'run',
        '--rm',
        '--detach',
        '-v', '/mnt:/mnt',
        '-v', '/raid/tmp:/dask-local-directory',
        '--network=host',
        '--name', container_name,
        '--entrypoint', '/usr/bin/tini',     # do not run Jupyter Notebook
        args.docker_image,
        '--',
        'bash', '-c',
        '"source activate rapids && %s"' % ' '.join(dask_worker_cmd),
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def start_containers(args):
    start_notebook(args)
    start_scheduler(args)
    with Pool(16) as p:
        p.map(functools.partial(start_worker, args), args.host)


def main():
    parser = configargparse.ArgParser(
        description='Start Dask cluster on multiple hosts using SSH and Docker',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=['start_dask.yaml'],
    )
    parser.add_argument('--container_name', action='store', default='dask',
                        help='Name to assign to the containers.')
    parser.add_argument('--device_memory_limit_gib', type=float, default=26)
    parser.add_argument('--docker_image', action='store',
                        default='nvcr.io/nvidia/rapidsai/rapidsai:0.10-cuda10.0-runtime-ubuntu18.04',
                        help='Docker image tag.')
    parser.add_argument('--host', '-H', action='append', required=True, help='List of hosts on which to invoke processes.')
    parser.add_argument('--memory_limit_gib', type=float, default=64)
    parser.add_argument('--nostart', dest='start', action='store_false',
                        default=True, help='Do not start containers')
    parser.add_argument('--scheduler_host', default='')
    parser.add_argument('--user', action='store',
                        default='root', help='SSH user')
    args = parser.parse_args()

    if args.scheduler_host == '':
        args.scheduler_host = args.host[0]

    stop_containers(args)
    if args.start:
        init_hosts(args)
        start_containers(args)


if __name__ == '__main__':
    main()
