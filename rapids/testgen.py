#!/usr/bin/env python3

"""
This script generates test definitions for P3 Test Driver.
Usage: ./testgen.py | p3_test_driver -t - -c p3_test_driver.config.yaml
"""

import json
import sys


def add_test():
    flush_compute = not cached
    flush_isilon = not cached
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
            '--num_workers', '%d' % num_workers,
            ] + [a for h in host for a in ['--host', h]] + [
            ] + [a for i in input_file for a in ['--input_file', i]] + [
            ] + [a for v in volume for a in ['-v', v]] + [
        ],
        compression=compression,
        docker_image=docker_image,
        flush_compute=flush_compute,
        flush_isilon=flush_isilon,
        host=host,
        input_file=input_file,
        json_regex=['FINAL RESULTS JSON: (.*)$'],
        num_workers=num_workers,
        partitions=partitions,
        storage_type=storage_type,
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

for repeat in range(0):
    for cached in [True]:
        for storage_type in ['isilon']:
            base_dir = base_dir_map[storage_type]
            for compression in ['snappy']:
                for partitions in [96]:
                    for docker_image in ['claudiofahey/rapidsai:a359097c3c18a534b91557d5abe772c73ef57d11de3dfb632e1516b0a01745f1']:
                        for num_workers in [48]:
                            for input_file in [[base_dir + '/from-spark3.orc/\\*.orc']]:
                                for warmup in [False]:
                                                    add_test()

# Full suite
for repeat in range(0):
    for cached in [False, True]:
        for storage_type in ['local','isilon']:
            base_dir = base_dir_map[storage_type]
            for compression in ['snappy', 'None']:
                for partitions in [96]:
                    for docker_image in ['claudiofahey/rapidsai:a359097c3c18a534b91557d5abe772c73ef57d11de3dfb632e1516b0a01745f1']:
                        for num_workers in [48]:
                            for input_file in [[base_dir + '/perf-%s.orc/\\*.orc' % (compression,)]]:
                                for warmup in [True, False] if cached else [False]:
                                                    add_test()


print(json.dumps(test_list, sort_keys=True, indent=4, ensure_ascii=False))
print('Number of tests generated: %d' % len(test_list), file=sys.stderr)
