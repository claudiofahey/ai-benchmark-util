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
    file_format):
    basename = '%s-%0.2fx-%dp-%dMiB-%s.%s' % (data_file_prefix, size_multiplier, partitions, stripe_size_MiB, compression, file_format)
    return os.path.join(base_dir, basename)


def add_test():
    flush_compute = not cached
    flush_isilon = not cached
    data_file_prefix = 'perf-from-spark'
    input_dir = get_data_path(base_dir, data_file_prefix, size_multiplier, partitions, stripe_size_MiB, compression, file_format)
    if not os.path.isdir(input_dir):
        print('Skipping missing input_dir %s' % input_dir, file=sys.stderr)
        return
    input_file = [os.path.join(input_dir, '\\*.%s' % file_format)]
    num_workers = num_worker_hosts * num_gpus_per_host
    t = dict(
        test='simple',
        record_as_test='rapids',
        max_test_attempts=1,
        base_dir=base_dir,
        cached=cached,
        command_template=[
            './rapids_benchmark_driver.py',
            '--docker_image', docker_image,
            '--flush_compute', '%d' % flush_compute,
            '--flush_isilon', '%d' % flush_isilon,
            '--isilon_host', '%(isilon_host)s',
            '--keep_dask_running', '%d' % keep_dask_running,
            '--num_gpus_per_host', '%d' % num_gpus_per_host,
            '--num_worker_hosts', '%d' % num_worker_hosts,
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
        json_regex=['FINAL RESULTS JSON: (.*)$'],
        num_gpus_per_host=num_gpus_per_host,
        num_workers=num_workers,
        num_worker_hosts=num_worker_hosts,
        partitions=partitions,
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

for repeat in range(3):
    for cached in [False,True]:
        for storage_type in ['local','isilon']: # 'isilon','local'
            base_dir = base_dir_map[storage_type]
            for size_multiplier in [3.0,2.0,1.0]:
                for partitions in [48]:
                    for stripe_size_MiB in [2048,1024,512,256,128,64]:
                        for compression in ['snappy']:
                            for file_format in ['orc']:
                                for docker_image in ['claudiofahey/rapidsai:a359097c3c18a534b91557d5abe772c73ef57d11de3dfb632e1516b0a01745f1']:
                                    for num_worker_hosts in [3]:
                                        for num_gpus_per_host in [16]:
                                            for warmup in [True, False] if cached else [False]:
                                                    add_test()

# Full suite
# for repeat in range(0):
#     for cached in [False, True]:
#         for storage_type in ['local','isilon']:
#             base_dir = base_dir_map[storage_type]
#             for compression in ['snappy', 'None']:
#                 for partitions in [96]:
#                     for docker_image in ['claudiofahey/rapidsai:a359097c3c18a534b91557d5abe772c73ef57d11de3dfb632e1516b0a01745f1']:
#                         for num_workers in [48]:
#                             for input_file in [[base_dir + '/perf-%s.orc/\\*.orc' % (compression,)]]:
#                                 for warmup in [True, False] if cached else [False]:
#                                                     add_test()


print(json.dumps(test_list, sort_keys=True, indent=4, ensure_ascii=False))
print('Number of tests generated: %d' % len(test_list), file=sys.stderr)
