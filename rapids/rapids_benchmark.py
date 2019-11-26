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
import sys
import time


def gpu_load_performance_data(performance_path):
    """ Loads performance data

    Returns
    -------
    GPU DataFrame
    """

    cols = [
        "loan_id", "monthly_reporting_period", "servicer", "interest_rate", "current_actual_upb",
        "loan_age", "remaining_months_to_legal_maturity", "adj_remaining_months_to_maturity",
        "maturity_date", "msa", "current_loan_delinquency_status", "mod_flag", "zero_balance_code",
        "zero_balance_effective_date", "last_paid_installment_date", "foreclosed_after",
        "disposition_date", "foreclosure_costs", "prop_preservation_and_repair_costs",
        "asset_recovery_costs", "misc_holding_expenses", "holding_taxes", "net_sale_proceeds",
        "credit_enhancement_proceeds", "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
        "non_interest_bearing_upb", "principal_forgiveness_upb", "repurchase_make_whole_proceeds_flag",
        "foreclosure_principal_write_off_amount", "servicing_activity_indicator"
    ]

    ddf = dask_cudf.read_orc(performance_path)
    # Fix column names from ORC file
    ddf = ddf.rename(columns=dict(zip(ddf.columns, cols)))
    return ddf


def run_benchmark(args):
    logging.info('run_benchmark: BEGIN')
    logging.info('args=%s' % str(args))

    client = Client(address=args.scheduler_address)
    logging.info('client=%s' % str(client))

    logging.info('Waiting for %d Dask workers' % args.num_workers)
    client.wait_for_workers(args.num_workers)

    input_files = [f for p in args.input_file for f in sorted(glob.glob(p))]
    logging.info('input_files=%s' % str(input_files))

    perf_ddf = gpu_load_performance_data(input_files)
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
        persist_sec=persist_sec,
    )
    logging.info('FINAL RESULTS JSON: ' + json.dumps(results))

    logging.info('run_benchmark: END')


def main():
    parser = configargparse.ArgParser(
        description='Execute RAPIDS benchmarks',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument('--config', '-c', required=False, is_config_file=True, help='config file path')
    parser.add_argument('--input_file', action='append', help='Input file', required=True)
    parser.add_argument('--log_level', type=int, default=logging.INFO, help='10=DEBUG,20=INFO')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--scheduler_address', default='127.0.0.1:8786', help='Dask scheduler address')
    args = parser.parse_args()

    root_logger = logging.getLogger()
    root_logger.handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=args.log_level, stream=sys.stdout)

    logging.info('args=%s' % str(args))

    run_benchmark(args)


if __name__ == '__main__':
    main()
