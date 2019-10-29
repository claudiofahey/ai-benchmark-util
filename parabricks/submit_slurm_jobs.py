#!/usr/bin/python3 -u
#
# Copyright (c) 2019 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
"""

import configargparse
import datetime
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from p3_test_driver.system_command import system_command, ssh, time_duration_to_seconds
from p3_test_driver.json_util import append_to_json_file
from p3_test_driver.p3_util import mkdir_for_file


def parse_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def flush_caches(args):
    # Flush Isilon cache.
    if args.flush:
        cmd = [
            'ssh',
            '-p', '22',
            '%s@%s' % (args.isilon_user, args.isilon_host),
            'isi_for_array', 'isi_flush',
        ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

    # Drop caches on all slave nodes.
    if args.flush:
        for host in args.host:
            ssh('root', host, '/mnt/isilon/data/tf-bench-util/drop_caches.sh')
        # cmd = [
        #     'srun',
        #     '--nodelist', ','.join(args.host),
        #     '../drop_caches.sh',
        # ]
        # print(' '.join(cmd))
        # subprocess.run(cmd, check=True)


def submit_slurm_jobs(args):
    if args.batch_uuid is None:
        args.batch_uuid = str(uuid.uuid4())
    logging.info('batch_uuid=%s' % args.batch_uuid)

    log_dir = os.path.join(args.log_dir, args.batch_uuid)
    logging.info('log_dir=%s' % log_dir)

    sample_records = [line.split(',') for line in args.sample_id]
    for sample_id_file_name in args.sample_id_file:
        with open(sample_id_file_name) as f:
            sample_records += [line.rstrip('\n').split(',') for line in f]

    logging.info('sample_records=%s', str(sample_records))

    if args.cancel_jobs:
        cmd = ['scancel', '-u', os.environ['USER']]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

    os.makedirs(log_dir, exist_ok=True)

    flush_caches(args)

    if True:
        for sample_rec in sample_records:
            sample_id = sample_rec[0]
            cmd = [
                # 'echo',
                'sbatch',
                '--gres', 'gpu:%d' % args.num_gpus,
                '--job-name', sample_id,
                '--output', os.path.join(log_dir, '%s.log' % sample_id),
                '--nodelist', ','.join(args.host),
                '--verbose',
                'parabricks_germline_pipeline_slurm.py',
                '--sample_id', sample_id,
                '--batch_uuid', args.batch_uuid,
            ]
            return_code, output, errors = system_command(
                cmd,
                print_command=True,
                print_output=True,
                raise_on_error=True,
                shell=False,
            )

    logging.info('Jobs started. Logging to: %s' % log_dir)


def main():
    parser = configargparse.ArgParser(
        description='Submit Parabricks jobs to Slurm.',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add('--batch_uuid', required=False)
    parser.add('--cancel_jobs', type=parse_bool, default=False)
    parser.add('--config', '-c', default='submit_slurm_jobs.yaml',
               required=False, is_config_file=True, help='config file path')
    parser.add('--flush', type=parse_bool, default=False, help='Flush caches')
    parser.add('--host', '-H', action='append', required=False, help='List of hosts on which to invoke processes.')
    parser.add('--isilon_host',
               help='IP address or hostname of an Isilon node. You must enable password-less SSH.')
    parser.add('--isilon_user', default='root',
               help='SSH user used to connect to Isilon.')
    parser.add('--log_level', type=int, default=logging.INFO, help='10=DEBUG,20=INFO')
    parser.add('--log_dir', help='Log directory', default='/tmp', required=True)
    parser.add('--num_gpus', type=int, default=4)
    parser.add('--sample_id', action='append', default=[], required=False)
    parser.add('--sample_id_file', action='append', default=[], required=False)
    args = parser.parse_args()

    # Initialize logging
    root_logger = logging.getLogger()
    root_logger.setLevel(args.log_level)
    console_handler = logging.StreamHandler(sys.stdout)
    logging.Formatter.converter = time.gmtime
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    root_logger.addHandler(console_handler)

    logging.debug('args=%s' % str(args))
    submit_slurm_jobs(args)


if __name__ == '__main__':
    main()
