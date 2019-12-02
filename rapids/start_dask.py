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


def parse_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


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
        ] + [a for v in args.volume for a in ['-v', v]] + [
        '--network=host',
        '--name', container_name,
        args.docker_image,
        'dask-scheduler',
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
        ] + [a for v in args.volume for a in ['-v', v]] + [
        '--network=host',
        '--name', container_name,
        args.docker_image,
        'jupyter-lab',
        '--allow-root',
        '--ip=0.0.0.0',
        '--no-browser',
        '--NotebookApp.token=""',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def start_worker(args, host):
    container_name = args.container_name + '-worker'
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
        ] + [a for v in args.volume for a in ['-v', v]] + [
        '--network=host',
        '--name', container_name,
        args.docker_image,
        'dask-cuda-worker',
        '--nthreads', '5',
        '--memory-limit', '%d' % int(args.memory_limit_gib * 1024 ** 3),
        '--device-memory-limit', '%d' % int(args.device_memory_limit_gib * 1024 ** 3),
        '--local-directory', '/dask-local-directory',
        '%s:8786' % args.scheduler_host,
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def wait_for_cluster(args):
    container_name = args.container_name + '-client'
    cmd = [
        'nvidia-docker',
        'run',
        '--rm',
        '--name', container_name,
        args.docker_image,
        'python',
        '-c',
        """
from dask.distributed import Client
client=Client(address='%s:8786')
client.wait_for_workers(48)
        """ % args.scheduler_host,
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def start_containers(args):
    if args.start_notebook:
        start_notebook(args)
    start_scheduler(args)
    worker_hosts = args.host[0:args.num_worker_hosts]
    with Pool(16) as p:
        p.map(functools.partial(start_worker, args), worker_hosts)
    if args.wait:
        wait_for_cluster(args)


def main():
    parser = configargparse.ArgParser(
        description='Start Dask cluster on multiple hosts using SSH and Docker',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument('--config', '-c', required=False, is_config_file=True, help='config file path')
    parser.add_argument('--container_name', action='store', default='dask',
                        help='Name to assign to the containers.')
    parser.add_argument('--device_memory_limit_gib', type=float, default=26.0)
    parser.add_argument('--docker_image', action='store',
                        default='nvcr.io/nvidia/rapidsai/rapidsai:0.10-cuda10.0-runtime-ubuntu18.04',
                        help='Docker image tag.')
    parser.add_argument('--host', '-H', action='append', required=True, help='List of hosts on which to run Dask services.')
    parser.add_argument('--memory_limit_gib', type=float, default=64.0)
    parser.add_argument('--num_worker_hosts', type=int, default=0, help='Number of hosts to start Dask workers on (0=all)')
    parser.add_argument('--scheduler_host', default='')
    parser.add_argument('--start', type=parse_bool, default=True)
    parser.add_argument('--start_notebook', type=parse_bool, default=True)
    parser.add_argument('--user', action='store',
                        default='root', help='SSH user')
    parser.add_argument('--volume', '-v', action='append', default=[])
    parser.add_argument('--wait', type=parse_bool, default=False)
    args = parser.parse_args()

    if args.scheduler_host == '':
        args.scheduler_host = args.host[0]

    if args.num_worker_hosts <= 0:
        args.num_worker_hosts = len(args.host)

    stop_containers(args)
    if args.start:
        init_hosts(args)
        start_containers(args)


if __name__ == '__main__':
    main()
