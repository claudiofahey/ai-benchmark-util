#!/usr/bin/env python3
#
# Copyright (c) Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
Execute RAPIDS benchmark.
This should run within a RAPIDS container and connect to an existing Dask cluster.
"""

import configargparse
import dask_cudf
from dask.distributed import Client, wait
import glob
import json
import logging
import os
import sys
import time


def parse_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def gpu_load_performance_data(performance_path, **kwargs):
    ddf = dask_cudf.read_orc(performance_path, **kwargs)
    return ddf


def run_benchmark(args):
    logging.info('run_benchmark: BEGIN')

    client = Client(address=args.scheduler_address)
    logging.info('client=%s' % str(client))

    logging.info('Waiting for %d Dask workers' % args.num_workers)
    client.wait_for_workers(args.num_workers)

    if args.single_batch:
        input_files = [f for p in args.input_file for f in sorted(glob.glob(p))]
        logging.info('len(input_files)=%d' % len(input_files))
        logging.debug('input_files=%s' % str(input_files))

        input_file_sizes = [os.path.getsize(f) for f in input_files]

        perf_ddf = gpu_load_performance_data(input_files, engine=args.cudf_engine)
        logging.debug('perf_ddf=%s' % str(perf_ddf.head()))

        t0 = time.time()
        perf_ddf = perf_ddf.persist()
        wait(perf_ddf)
        t1 = time.time()
        persist_sec = t1 - t0
        logging.info('persist_sec=%f' % persist_sec)

        logging.info('perf_ddf=%s' % str(perf_ddf))

        compute_sec_list = []

        for i in range(3):
            t0 = time.time()
            computed = perf_ddf.groupby(['servicer'])['interest_rate'].max().compute()
            t1 = time.time()
            compute_sec = t1 - t0
            compute_sec_list += [compute_sec]
            logging.info('compute_sec=%f' % compute_sec)
            logging.info('len(computed)=%s' % len(computed))
            logging.debug('computed=%s' % str(computed))

        logging.info('compute_sec_list=%s' % str(compute_sec_list))
        logging.info('len(perf_ddf)=%s' % len(perf_ddf))

        checksum = int(perf_ddf['loan_id'].sum().compute())

        results = dict(
            checksum=checksum,
            compute_sec_list=compute_sec_list,
            num_input_files=len(input_files),
            input_file_sizes=input_file_sizes,
            persist_sec=persist_sec,
            dask_cudf_version=dask_cudf.__version__,
        )
    else:
        logging.info('Getting input file list')
        glob_t0 = time.time()
        input_files = [sorted(glob.glob(p)) for p in args.input_file]
        glob_sec = time.time() - glob_t0

        logging.info('Getting file sizes')
        getsize_t0 = time.time()
        input_file_sizes = [[os.path.getsize(f) for f in batch_files] for batch_files in input_files]
        getsize_sec = time.time() - getsize_t0

        logging.info('Creating distributed data frames')
        create_ddf_t0 = time.time()
        perf_ddfs = [gpu_load_performance_data(batch_files, engine=args.cudf_engine) for batch_files in input_files]
        create_ddf_sec = time.time() - create_ddf_t0

        compute_sec_list = []
        for batch, perf_ddf in enumerate(perf_ddfs):
            logging.info('Computing batch %d' % batch)
            compute_t0 = time.time()
            computed = perf_ddf.groupby(['servicer'])['interest_rate'].max().compute()
            logging.info('len(computed)=%s' % len(computed))
            logging.debug('computed=%s' % str(computed))
            del perf_ddf
            compute_sec = time.time() - compute_t0
            compute_sec_list += [compute_sec]
            logging.info('compute_sec=%f' % compute_sec)

        results = dict(
            create_ddf_sec=create_ddf_sec,
            compute_sec_list=compute_sec_list,
            dask_cudf_version=dask_cudf.__version__,
            getsize_sec=getsize_sec,
            glob_sec=glob_sec,
            input_files=input_files,
            input_file_sizes=input_file_sizes,
        )

    logging.info('FINAL RESULTS JSON: ' + json.dumps(results))
    logging.info('run_benchmark: END')


def main():
    parser = configargparse.ArgParser(
        description='Execute RAPIDS benchmarks',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument('--config', '-c', required=False, is_config_file=True, help='config file path')
    parser.add_argument('--cudf_engine', default='cudf', help='cudf or pyarrow')
    parser.add_argument('--input_file', action='append', help='Input file', required=True)
    parser.add_argument('--log_level', type=int, default=logging.INFO, help='10=DEBUG,20=INFO')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--scheduler_address', default='127.0.0.1:8786', help='Dask scheduler address')
    parser.add_argument('--single_batch', type=parse_bool, default=False)
    args = parser.parse_args()

    root_logger = logging.getLogger()
    root_logger.handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=args.log_level, stream=sys.stdout)

    logging.info('args=%s' % str(args))

    run_benchmark(args)


if __name__ == '__main__':
    main()
