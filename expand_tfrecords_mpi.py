#!/usr/bin/env python3
#
# Copyright (c) 2018 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
This script will make multiple copies of the original TFRecord training files.
"""

import os
import argparse
from os.path import join, basename, splitext
from shutil import copyfile


def worker(rank, size, input_dir, output_dir, num_copies):
    num_files = 1024

    # use round-robin scheduling
    i = rank
    while (i < num_files):
        in_file_name = join(input_dir, 'train-%05d-of-%05d' % (i, num_files))
        for copy in range(num_copies):
            out_file_name = join(output_dir, 'train-%05d-of-%05d-copy-%05d' % (i, num_files, copy))
            print('%s => %s' % (in_file_name, out_file_name))
            copyfile(in_file_name, out_file_name)
        i += size
    return


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input_dir', help='Input directory', required=True)
    parser.add_argument('-o','--output_dir', help='Output directory', required=True)
    parser.add_argument('-n','--num_copies', type=int, help='Number of copies', required=True)
    args = vars(parser.parse_args())

    input_dir = args['input_dir']
    output_dir = args['output_dir']
    num_copies = args['num_copies']

    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

    worker(rank, size, input_dir, output_dir, num_copies)


if __name__ == '__main__':
    main()
