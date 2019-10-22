#!/usr/bin/env python3
#
# Copyright (c) 2019 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
This script will record the byte counters on Mellanox HCAs at millisecond resolution.
It outputs a CSV file that can be analyzed with analyze_mlx_counters.ipynb.
"""

import argparse
import time
import sys


class Counter:
    device: int
    port: int
    counter_name: str
    filename: str
    counter_value_prev: int
    t_prev: float
    max_rate_gbps: float
    file: object


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--sleep_sec', type=float, default=0.0)
    parser.add_argument('--run_sec', type=float, default=60.0)
    parser.add_argument('--device', '-d', type=int, action='append', default=[0, 1, 8, 9])
    parser.add_argument('--port', '-p', type=int, action='append', default=[1])
    parser.add_argument('--counter', '-c', action='append', default=['port_rcv_data', 'port_xmit_data'])
    args, filenames = parser.parse_known_args()

    counters = []
    for device in args.device:
        for port in args.port:
            for counter_name in args.counter:
                counter = Counter()
                counter.device = device
                counter.port = port
                counter.counter_name = counter_name
                counter.filename = '/sys/class/infiniband/mlx5_%d/ports/%d/counters/%s' % (counter.device, counter.port, counter.counter_name)
                counter.counter_value_prev = -1
                counter.t_prev = 0.0
                counter.max_rate_gbps = 0.0
                counters += [counter]

    for counter in counters:
        counter.file = open(counter.filename, 'r')
    t = 0.0
    stop_time = time.time() + args.run_sec
    iterations = 0
    while True:
        for counter in counters:
            counter.file.seek(0)
            t = time.time()
            counter_value = int(counter.file.readlines()[0].rstrip())
            time_delta = t - counter.t_prev
            if counter.counter_value_prev != -1:
                counter_delta = counter_value - counter.counter_value_prev
                rate = counter_delta / time_delta
                rate_gbps = rate * 4 * 8e-9
                if counter.max_rate_gbps < rate_gbps:
                    counter.max_rate_gbps = rate_gbps
                print('%0.9f,%17d,%7.3f,%d,%d,%s' % (t, counter_value, rate_gbps, counter.device, counter.port, counter.counter_name))
            counter.counter_value_prev = counter_value
            counter.t_prev = t
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)
        iterations += 1
        if t > stop_time:
            break

    for counter in counters:
        print('max rate: %7.3f,%d,%d,%s' % (counter.max_rate_gbps, counter.device, counter.port, counter.counter_name), file=sys.stderr)
    print('iteration frequency: %f per sec' % (iterations / args.run_sec), file=sys.stderr)

if __name__ == '__main__':
    main()
