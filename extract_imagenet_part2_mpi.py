#!/usr/bin/env python3
#
# Copyright (c) 2018 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
This script will extract all .tar files in the input directory to subdirectories in the output directory.
This is used to extract the training files nXXXXXXXX.tar to nXXXXXXXX/*.JPG.
"""

import os
import argparse
import subprocess
from os.path import join, basename, splitext
from glob import glob

def worker(rank, size, data_in_dir, data_out_dir):
    in_files = sorted(glob(join(data_in_dir, '*.tar')))
    num_files = len(in_files)

    # use round-robin scheduling
    i = rank    
    while (i < num_files):
        in_file_name = in_files[i]
        label = splitext(basename(in_file_name))[0]
        out_dir_name = join(data_out_dir, label)
        print('%s %s %d' % (in_file_name, label, rank))
        os.makedirs(out_dir_name, exist_ok=True)
        subprocess.run(['tar', '-xf', in_file_name, '-C', out_dir_name], check=True)
        i += size
    return

def main():
    parser = argparse.ArgumentParser(description='Data augmentation with random transformations')
    parser.add_argument('-i','--input_data_dir', help='Input data directory', required=True)
    parser.add_argument('-o','--output_data_dir', help='Output data directory', required=True)
    args = vars(parser.parse_args())

    data_input_dir = args['input_data_dir']
    data_output_dir = args['output_data_dir']
    
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

    worker(rank, size, data_input_dir, data_output_dir)

if __name__ == '__main__':
    main()
