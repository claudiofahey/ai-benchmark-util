#!/usr/bin/python3
#
# Copyright (c) 2019 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
"""

import glob
import os
import sys
import configargparse


def filter_sample_ids(args):
    sample_records = []
    for sample_id_file_name in args.sample_id_file:
        with open(sample_id_file_name) as f:
            sample_records += [line.rstrip('\n').split(',') for line in f]
    for sample_record in sample_records:
        sample_id = sample_record[0]
        input_files = glob.glob(os.path.join(args.input_dir, sample_id, '*_*.fq.gz'))
        num_input_files = len(input_files)
        if num_input_files == 110:
            print(','.join(sample_record))
        else:
            print('%s has %d files' % (sample_id, num_input_files), file=sys.stderr)


def main():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add('--input_dir', help='Input directory', default='/mnt/isilon1/data/genomics/from_nas/fq')
    parser.add('--sample_id_file', action='append', default=[])
    args = parser.parse_args()
    filter_sample_ids(args)


if __name__ == '__main__':
    main()
