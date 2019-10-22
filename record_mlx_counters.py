#!/usr/bin/env python3
#
# Copyright (c) 2019 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

import argparse
import time
import glob


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--sleep_sec', type=float, default=0.0)
    args, filenames = parser.parse_known_args()
    if not filenames:
        filenames = [
            '/sys/class/infiniband/mlx5_0/ports/*/counters/port_xmit_data',
            '/sys/class/infiniband/mlx5_0/ports/*/counters/port_rcv_data',
        ]
    filenames = [f for filename in filenames for f in sorted(glob.glob(filename))]
    files = [open(f, 'r') for f in filenames]
    counter_value_prev = [-1 for f in filenames]
    t_prev = 0
    while True:
        t = time.time()
        time_delta = t - t_prev
        for i in range(len(filenames)):
            files[i].seek(0)
            counter_value = int(files[i].readlines()[0].rstrip())
            if counter_value_prev[i] != -1:
                counter_delta = counter_value - counter_value_prev[i]
                rate = counter_delta / time_delta
                print('%0.9f,%15d,%6.0f,%s' % (t, counter_value, rate*1e-6, filenames[i]))
            counter_value_prev[i] = counter_value
        t_prev = t
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)


if __name__ == '__main__':
    main()
