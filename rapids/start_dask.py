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
from joblib import Parallel, delayed
import subprocess


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
        '%s@%s' % (args.user, host),
        'bash', '-c',
        '"docker ps --format {{.Names}} | grep ^%s- | xargs -i -P 0 docker stop {}"' % args.container_name
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=False)


def stop_containers(args):
    jobs = (delayed(stop_containers_on_host)(args, h) for h in args.host)
    Parallel(n_jobs=16)(jobs)


def init_host(args, host):
    cmd = [
        'ssh',
        '%s@%s' % (args.user, host),
        'nvidia-docker',
        'pull',
        args.docker_image,
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def init_hosts(args):
    jobs = (delayed(init_host)(args, h) for h in args.host)
    Parallel(n_jobs=16)(jobs)


def start_scheduler(args):
    host = args.scheduler_host
    container_name = args.container_name + '-scheduler'
    calculated_volumes = [v % 1 for v in args.volume_template]
    cmd = [
        'ssh',
        '%s@%s' % (args.user, host),
        'nvidia-docker',
        'run',
        '--rm',
        '--detach',
        '-v', '/mnt:/mnt',
        ] + [a for v in args.volume for a in ['-v', v]] + [
        ] + [a for v in calculated_volumes for a in ['-v', v]] + [
        '--network=host',
        '--name', container_name,
        args.docker_image,
        'dask-scheduler',
        '--port', '%d' % args.scheduler_port,
        '--dashboard-address', ':%d' % (args.scheduler_port + 1),
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def start_worker(args, host, container_number):
    gpu_ids = [str(i) for i in range(args.num_gpus_per_host)]
    container_name = '%s-worker-%d' % (args.container_name, container_number)
    calculated_volumes = [v % (container_number + 1) for v in args.volume_template]
    cuda_visible_devices = ','.join(gpu_ids[container_number::args.num_containers_per_host])
    cmd = [
        'ssh',
        '%s@%s' % (args.user, host),
        'nvidia-docker',
        'run',
        '--rm',
        '--detach',
        '-e', 'CUDA_VISIBLE_DEVICES=%s' % cuda_visible_devices,
        '-v', '/mnt:/mnt',
        '-v', '/raid/tmp:/dask-local-directory',
        ] + [a for v in args.volume for a in ['-v', v]] + [
        ] + [a for v in calculated_volumes for a in ['-v', v]] + [
        '--network=host',
        '--name', container_name,
        args.docker_image,
        'dask-cuda-worker',
        '--nthreads', '%d' % args.nthreads,
        '--memory-limit', '%d' % int(args.memory_limit_gib * 1024 ** 3),
        '--device-memory-limit', '%d' % int(args.device_memory_limit_gib * 1024 ** 3),
        '--local-directory', '/dask-local-directory',
        '%s:%d' % (args.scheduler_host, args.scheduler_port),
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def start_notebook(args):
    host = args.scheduler_host
    container_name = args.container_name + '-notebook'
    cmd = [
        'ssh',
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


def wait_for_cluster(args):
    container_name = args.container_name + '-client'
    num_workers = args.num_worker_hosts * args.num_gpus_per_host
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
client=Client(address='%s:%d')
client.wait_for_workers(%d)
        """ % (args.scheduler_host, args.scheduler_port, num_workers),
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def start_containers(args):
    if args.start_notebook:
        start_notebook(args)
    start_scheduler(args)
    worker_hosts = args.host[0:args.num_worker_hosts]
    jobs = (delayed(start_worker)(args, h, i)
            for i in range(args.num_containers_per_host)
            for h in worker_hosts)
    Parallel(n_jobs=10*args.num_worker_hosts)(jobs)
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
    parser.add_argument('--nthreads', type=int, default=5)
    parser.add_argument('--num_containers_per_host', type=int, default=1)
    parser.add_argument('--num_gpus_per_host', type=int, default=16)
    parser.add_argument('--num_worker_hosts', type=int, default=0, help='Number of hosts to start Dask workers on (0=all)')
    parser.add_argument('--scheduler_host', default='')
    parser.add_argument('--scheduler_port', type=int, default=8786)
    parser.add_argument('--start', type=parse_bool, default=True)
    parser.add_argument('--start_notebook', type=parse_bool, default=True)
    parser.add_argument('--user', action='store',
                        default='root', help='SSH user')
    parser.add_argument('--volume', '-v', action='append', default=[])
    parser.add_argument('--volume_template', action='append', default=[])
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
