#!/usr/bin/env python3
#
# Copyright (c) 2018 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#


"""
This reads TFRecord files and outputs individual JPEG files.
This is the inverse of build_imagenet_data.py.
"""

import argparse
import os
from glob import glob

import tensorflow as tf


def process_tf_record_file(input_tf_record_filename, output_dir):
    """Read single TFRecord file, write JPEG files."""
    tf_record_iterator = tf.python_io.tf_record_iterator(path=input_tf_record_filename)
    for record_string in tf_record_iterator:

        # Parse input record.
        example = tf.train.Example()
        example.ParseFromString(record_string)
        filename = example.features.feature['image/filename'].bytes_list.value[0]
        synset = example.features.feature['image/class/synset'].bytes_list.value[0]
        original_encoded = example.features.feature['image/encoded'].bytes_list.value[0]

        # Create directory.
        synset_dir = os.path.join(output_dir, synset.decode())
        os.makedirs(synset_dir, exist_ok=True)
        output_jpeg_file = os.path.join(synset_dir, filename.decode())
        print(filename, synset, output_jpeg_file)

        # Write JPEG file.
        with open(output_jpeg_file, "wb") as output_jpeg_file:
            output_jpeg_file.write(original_encoded)


def worker(rank, size, input_files, output_dir):
    with tf.Session():
        input_tf_record_filenames = sorted(glob(input_files))
        num_files = len(input_tf_record_filenames)
        i = rank
        while i < num_files:
            input_tf_record_filename = input_tf_record_filenames[i]
            print(rank, input_tf_record_filename, output_dir)
            process_tf_record_file(input_tf_record_filename, output_dir)
            i += size


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input_files', help='Input files', required=True)
    parser.add_argument('-o', '--output_dir', help='Output directory', required=True)
    args = parser.parse_args()
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    worker(rank, size, args.input_files, args.output_dir)


if __name__ == '__main__':
    main()
