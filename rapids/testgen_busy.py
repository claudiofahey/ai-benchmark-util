#!/usr/bin/env python3

"""
This script generates test definitions for P3 Test Driver.
Usage: ./testgen.py | p3_test_driver -t - -c p3_test_driver.config.yaml
"""

import json
import os
import sys


def get_data_path(
    base_dir,
    data_file_prefix,
    size_multiplier,
    partitions,
    stripe_size_MiB,
    compression,
    file_format,
    batch,
    ):

    basename = '%s-%0.2fx-%dp-%dMiB-%s.%s.%d' % (data_file_prefix, size_multiplier, partitions, stripe_size_MiB, compression, file_format, batch)
    return os.path.join(base_dir, basename)


def add_test():
    flush_compute = not cached
    flush_isilon = not cached
    data_file_prefix = 'perf-from-spark'
    input_dir = [
        get_data_path(base_dir, data_file_prefix, size_multiplier, partitions, stripe_size_MiB, compression, file_format, batch)
        for batch in range(num_batches)]
    input_file = [os.path.join(d, '\\*.%s' % file_format) for d in input_dir]
    num_workers = num_worker_hosts * num_gpus_per_host
    t = dict(
        test='simple',
        record_as_test='rapids',
        max_test_attempts=2,
        base_dir=base_dir,
        cached=cached,
        command_template=[
            './rapids_benchmark_driver.py',
            '--cudf_engine', cudf_engine,
            '--docker_image', docker_image,
            '--flush_compute', '%d' % flush_compute,
            '--flush_isilon', '%d' % flush_isilon,
            '--isilon_host', '%(isilon_host)s',
            '--keep_dask_running', '%d' % keep_dask_running,
            '--num_gpus_per_host', '%d' % num_gpus_per_host,
            '--num_worker_hosts', '%d' % num_worker_hosts,
            '--single_batch', '%d' % single_batch,
            ] + [a for h in host for a in ['--host', h]] + [
            ] + [a for i in input_file for a in ['--input_file', i]] + [
            ] + [a for v in volume for a in ['-v', v]] + [
        ],
        compression=compression,
        data_file_prefix=data_file_prefix,
        docker_image=docker_image,
        file_format=file_format,
        flush_compute=flush_compute,
        flush_isilon=flush_isilon,
        host=host,
        input_file=input_file,
        isilon_access_pattern=isilon_access_pattern,
        json_regex=['FINAL RESULTS JSON: (.*)$'],
        num_batches=num_batches,
        num_gpus_per_host=num_gpus_per_host,
        num_workers=num_workers,
        num_worker_hosts=num_worker_hosts,
        partitions=partitions,
        single_batch=single_batch,
        size_multiplier=size_multiplier,
        storage_type=storage_type,
        stripe_size_MiB=stripe_size_MiB,
        warmup=warmup,
    )
    test_list.append(t)


test_list = []

host = [
    '10.200.11.12',
    '10.200.11.13',
    '10.200.11.11',
]
base_dir_map = dict(
    local='/raid/mortgage',
    isilon='/mnt/isilon1/data/mortgage',
)
volume = [
    '/raid:/raid',
]
keep_dask_running = False
single_batch = False
isilon_access_pattern = 'streaming'
#docker_image = 'claudiofahey/rapidsai:0.10-cuda10.0-runtime-ubuntu18.04-custom'
docker_image = 'claudiofahey/rapidsai:46ee5e319153ba1b29021aba56db9a47ab81f1b978ae7c03e73c402cbc9dcf4b'

for repeat in range(1):
    for cached in [False]:
        for storage_type in ['isilon']:
            base_dir = base_dir_map[storage_type]
            for size_multiplier in [3.0]:
                for partitions in [48]:
                    for stripe_size_MiB in [2048]:
                        for compression in ['snappy']:
                            for file_format in ['orc']:
                                    for num_worker_hosts in [3]:
                                        for num_gpus_per_host in [16]:
                                            for num_batches in [100]:
                                                for cudf_engine in ['pyarrow']:
                                                    for warmup in [False]:
                                                        add_test()

print(json.dumps(test_list, sort_keys=True, indent=4, ensure_ascii=False))
print('Number of tests generated: %d' % len(test_list), file=sys.stderr)
