#!/usr/bin/env python3

"""
This script generates test definitions for P3 Test Driver.
Usage: ./testgen.py | p3_test_driver -t - -c p3_test_driver.config.yaml
"""

import json
import sys


def add_test():
    num_copies = 1 if cached else num_copies_uncached

    # if num_copies == 1 and image_resize_factor == 1.0:
    #     data_dir_suffix = ''
    # elif num_copies == 150 and image_resize_factor == 1.0:
    #     data_dir_suffix = '-150x'
    # elif num_copies == 1 and image_resize_factor == 3.0:
    #     data_dir_suffix = '1729'
    # else:
    #     raise Exception()
    #
    data_dir_suffix = '-png'
    fuse_decode_and_crop = False

    data_dir_template = '/mnt/isilon%%d/data/imagenet-scratch/tfrecords' + data_dir_suffix
    flush = not cached

    t = dict(
        test='simple',
        record_as_test='tensorflow_cnn_benchmark',
        max_test_attempts=2,
        pre_commands=[
            dict(key='tensorflow_benchmark_git_hash',
                 command_template='cd ../tensorflow-benchmarks && git rev-parse --short HEAD'),
            dict(key='NVIDIA_TENSORFLOW_VERSION',
                 command_template='docker exec tf /bin/bash -c "echo \\$NVIDIA_TENSORFLOW_VERSION"'),
            dict(key='TENSORFLOW_VERSION',
                 command_template='docker exec tf /bin/bash -c "echo \\$TENSORFLOW_VERSION"'),
        ],
        command_template=[
            'docker',
            'exec',
            'tf',
            './run_benchmark.py',
            '--batch_group_size', '%d' % batch_group_size,
            '--batch_size', '%d' % batch_size,
            '--data_dir_template', data_dir_template,
            '--data_dir_template_count', '%d' % data_dir_template_count,
            '--datasets_prefetch_buffer_size', '%d' % datasets_prefetch_buffer_size,
            '--datasets_num_private_threads', '%d' % datasets_num_private_threads,
            '--flush', '%d' % flush,
            '--fp16', '%d' % fp16,
            '--fuse_decode_and_crop', '%d' % fuse_decode_and_crop,
            '--isilon_host', '%(isilon_host)s',
            '--model', model,
            '--noop', '%d' % noop,
            '--np', '%d' % np,
            '--npernode', '%d' % npernode,
            '--num_batches', '%d' % num_batches,
            '--num_hosts', '%d' % num_hosts,
            '--num_intra_threads', '%d' % num_intra_threads,
            '--num_inter_threads', '%d' % num_inter_threads,
        ],
        batch_group_size=batch_group_size,
        batch_size=batch_size,
        cached=cached,
        data_dir_suffix=data_dir_suffix,
        data_dir_template=data_dir_template,
        data_dir_template_count=data_dir_template_count,
        datasets_prefetch_buffer_size=datasets_prefetch_buffer_size,
        datasets_num_private_threads=datasets_num_private_threads,
        flush=flush,
        fp16=fp16,
        fuse_decode_and_crop=fuse_decode_and_crop,
        image_format='PNG',
        image_resize_factor=image_resize_factor,
        isilon_flush=flush,
        model=model,
        np=np,
        npernode=npernode,
        num_batches=num_batches,
        num_copies=num_copies,
        num_hosts=num_hosts,
        num_intra_threads=num_intra_threads,
        num_inter_threads=num_inter_threads,
        storage_type=storage_type,
    )
    test_list.append(t)


test_list = []

num_copies_uncached = 150
image_resize_factor = 1.0
fp16 = True
noop = False
storage_type = 'isilon'

for repeat in range(1):
    for cached in [True]:
        for model in ['trivial']:
            for batch_group_size in [10]:
                if model == 'vgg16':
                    batch_sizes = [192]
                else:
                    batch_sizes = [256]
                for batch_size in batch_sizes:
                    for data_dir_template_count in [1 if cached else 16]:
                        for datasets_prefetch_buffer_size in [20,5]:
                            for datasets_num_private_threads in [11,10,8,6,4]:
                                for num_batches in [250]:
                                    for num_hosts in [1]:
                                        for npernode in [16]:
                                            np = num_hosts * npernode
                                            for num_inter_threads in [40]:
                                                for num_intra_threads in [1]:
                                                    add_test()

# Full test suite
for repeat in range(0):
    for cached in [False, True]:
        for model in ['resnet50', 'vgg16', 'resnet152', 'inception3', 'inception4']:
            for batch_group_size in [10]:
                if model == 'vgg16':
                    batch_sizes = [192]
                else:
                    batch_sizes = [256]
                for batch_size in batch_sizes:
                    for data_dir_template_count in [1 if cached else 16]:
                        for datasets_prefetch_buffer_size in [40]:
                            for datasets_num_private_threads in [4]:
                                for num_batches in [1000]:
                                    for num_hosts in [3, 2, 1]:
                                        for npernode in [16]:
                                            np = num_hosts * npernode
                                            for num_intra_threads in [1]:
                                                for num_inter_threads in [40]:
                                                    add_test()


print(json.dumps(test_list, sort_keys=True, indent=4, ensure_ascii=False))
print('Number of tests generated: %d' % len(test_list), file=sys.stderr)
