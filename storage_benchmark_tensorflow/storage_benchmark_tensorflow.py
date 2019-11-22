#!/usr/bin/env python3
#
# Copyright (c) 2018 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
This uses TensorFlow to read a set of TFRecord files and show the throughput.
It can read files synchronously (all processes stay at the same step) or
asynchronously (all processes are independent).

To install:
pip install --user --requirement requirements.txt
"""

import configargparse
import time
import socket
import horovod.tensorflow as hvd
import tensorflow as tf
import numpy as np
import datetime
import glob
import json
import os
import subprocess
import sys
import uuid
from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import threadpool
from token_bucket import TokenBucket


def parse_bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def process_record(example_serialized):
    # example_serialized = tf.Print(example_serialized, [example_serialized], 'example_serialized: ', first_n=10, summarize=10)
    # print('example_serialized=%s' % str(example_serialized))
    example_bytes = tf.io.decode_raw(example_serialized, tf.uint8, name='example_bytes')
    # example_bytes = tf.Print(example_bytes, [example_bytes], 'example_bytes: ', first_n=10, summarize=10)
    num_bytes = tf.size(example_bytes, out_type=tf.int64)
    # print('num_bytes=%s' % str(num_bytes))
    return num_bytes


def create_iterator(input_file_spec, input_filenames, args):
    if input_filenames:
        ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(input_filenames))
    elif input_file_spec:
        ds = tf.data.TFRecordDataset.list_files(input_file_spec)
    else:
        raise ValueError('You must specify input_file_spec or input_filenames')

    if args.parallel_interleave_cycle_length:
        ds = ds.apply(
            interleave_ops.parallel_interleave(
                lambda f: tf.data.TFRecordDataset(f, buffer_size=args.dataset_buffer_size),
                cycle_length=args.parallel_interleave_cycle_length))
    else:
        ds = ds.apply(tf.data.TFRecordDataset)

    ds = ds.prefetch(buffer_size=args.prefetch_records)
    ds = ds.repeat()
    num_splits = 1
    ds = ds.apply(
        batching.map_and_batch(
            map_func=process_record,
            batch_size=args.batch_size,
            num_parallel_batches=num_splits))
    ds = ds.prefetch(buffer_size=num_splits)

    if args.num_threads:
        ds = threadpool.override_threadpool(
            ds,
            threadpool.PrivateThreadPool(
                args.num_threads, display_name='input_pipeline_thread_pool'))
        ds_iterator = ds.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                             ds_iterator.initializer)
    else:
        ds_iterator = ds.make_one_shot_iterator()
    return ds_iterator


def worker(rank, size, args):
    if rank == 0:
        print('storage_benchmark_tensorflow: BEGIN')
        print(datetime.datetime.utcnow())

    metrics_file_name = os.path.join(args.metrics_directory, 'storage_benchmark_tensorflow_metrics-%s-%d.log' % (args.test_uuid, rank))
    with open(metrics_file_name, 'a') as metrics_file:

        hostname = socket.gethostname()

        # Set random seed to have deterministic behavior.
        tf.set_random_seed(rank + 1)

        # Round robin the input file spec. This allows multiple mount points to be used.
        input_file_spec = args.input_file_specs[hvd.local_rank() % len(args.input_file_specs)]
        print('rank=%3d: %s: input_file_spec=%s' % (rank, hostname, input_file_spec))

        if args.round_robin_files:
            # Distribute sets of file names evenly over all processes and without overlap.
            all_input_filenames = sorted(glob.glob(input_file_spec))
            num_files = len(all_input_filenames)
            i = rank
            input_filenames = []
            while i < num_files:
                input_filenames.append(all_input_filenames[i])
                i += size
            print('rank=%3d: Found %d total files. %d files assigned to this process.' % (rank, len(all_input_filenames), len(input_filenames)))
            if len(input_filenames) == 0:
                raise ValueError('Not enough matching files.')
            input_file_spec = None
        else:
            # This will use tf.data.TFRecordDataset.list_files to randomly distribute files.
            input_filenames = None

        #
        # Build execution graph.
        #

        ds_iterator = create_iterator(input_file_spec, input_filenames, args)

        # num_bytes_tensor is an int64 tensor of shape (batch_size).
        num_bytes_tensor = ds_iterator.get_next()

        # When num_bytes_for_step_tensor is evaluated, it reads the TFRecord files.
        num_bytes_for_step_tensor = tf.reduce_sum(num_bytes_tensor)

        # The following operations are used to synchronize the processes when running in sync mode.
        if args.sync:
            stop_flag_placeholder = tf.placeholder(tf.bool, shape=())
            stop_flag_broadcast_tensor = hvd.broadcast(stop_flag_placeholder, 0, 'stop_flag_broadcast')
            num_bytes_for_step_placeholder = tf.placeholder(tf.int64, shape=())
            total_bytes_for_step_tensor = hvd.allreduce(num_bytes_for_step_placeholder, average=False)

        #
        # Start the TensorFlow session and execute the graph.
        #

        config = tf.ConfigProto()
        config.device_count['GPU'] = 0
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1
        print('rank=%3d: Creating session' % rank)
        with tf.Session(config=config) as session:
            print('rank=%3d: Session created' % rank)
            session.run([tf.initializers.global_variables(), tf.tables_initializer()])
            print('rank=%3d: Initialized variables' % rank)

            # Run first step. This can take 30 seconds for 100,000 files.
            print('rank=%3d: Running first step' % rank)
            _ = session.run(num_bytes_for_step_tensor)
            print('rank=%3d: First step complete' % rank)

            # Wait for barrier so we know when all processes have finished the first step.
            print('rank=%3d: Waiting for barrier' % rank)
            session.run(hvd.allreduce(tf.constant(0)))
            if rank == 0:
                print('rank=%3d: Completed waiting for barrier' % rank)

            # To ensure that all processes finish warmup and stop at exactly the same time,
            # the rank 0 node broadcasts its time to all other ranks.
            # This also serves as a synchronization barrier.
            local_t0 = time.time()
            t0_tensor = tf.constant(local_t0, tf.float64)
            t0_tensor = hvd.broadcast(t0_tensor, 0, 't0')
            t0 = session.run(t0_tensor)

            start_time = t0 + args.warmup_sec
            stop_time = start_time + args.run_sec
            step = 0
            warmed_up = False
            num_records = 0
            num_bytes = 0
            total_bytes = 0
            next_report_time = time.time() + args.report_period_sec

            if args.throttle_total_rate_bytes_per_sec:
                throttle_rate_bytes_per_sec = args.throttle_total_rate_bytes_per_sec / size
                burst_sec = 1.0
                throttle = TokenBucket(tokens=args.throttle_rate_bytes_per_sec*burst_sec, fill_rate=throttle_rate_bytes_per_sec)
            else:
                throttle = None

            while True:
                # Reset all counters when warmup completes.
                t = time.time()
                if not warmed_up and t >= start_time:
                    print('rank=%3d: warmup complete at step %d' % (rank, step))
                    warmed_up = True
                    t0 = start_time
                    step = 0
                    num_records = 0
                    num_bytes = 0
                    total_bytes = 0

                # Run a single step of batch_size records per process.
                run_options = tf.RunOptions()
                # run_options.timeout_in_ms = 10000
                num_bytes_for_step = np.int64(0)
                try:
                    num_bytes_for_step = session.run(num_bytes_for_step_tensor, options=run_options)
                except Exception as e:
                    print('rank=%3d: %s: ERROR: %s' % (rank, hostname, e))

                step_dt = time.time() - t

                if (warmed_up or step >= 1) and step_dt > args.warn_latency_sec:
                    print('rank=%3d: %s: WARNING: step %d took %0.3f seconds' %
                          (rank, hostname, step, step_dt))
                    next_report_time = 0.0

                # Calculate local stop flag. In sync mode, this is broadcast from rank 0.
                stop_flag = time.time() >= stop_time

                # Use Horovod to aggregate the byte counter across all processes.
                # This also acts as a synchronization barrier, much like gradient descent when
                # it shares gradients.
                # Also coordinate the stop flag so all processes stop at the same step.
                sync_dt = 0.0
                if args.sync:
                    t = time.time()
                    total_bytes_for_step, stop_flag = session.run(
                        [total_bytes_for_step_tensor, stop_flag_broadcast_tensor],
                        feed_dict={
                            num_bytes_for_step_placeholder: num_bytes_for_step,
                            stop_flag_placeholder: stop_flag,
                        },
                    )

                    total_bytes += total_bytes_for_step

                    sync_dt = time.time() - t
                    if warmed_up and sync_dt > 30.0:
                        print('rank=%3d: %s: WARNING: sync after step %d took %0.3f seconds' %
                              (rank, hostname, step, sync_dt))
                        next_report_time = 0.0

                num_records += args.batch_size
                num_bytes += num_bytes_for_step
                t = time.time()

                metrics = {
                    '@timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
                    'batch_size': args.batch_size,
                    'rank': rank,
                    'hostname': hostname,
                    'step': step,
                    'num_bytes': int(num_bytes_for_step),
                    'latency_sec': step_dt,
                    'sync_latency_sec': sync_dt,
                }
                json.dump(metrics, metrics_file)
                metrics_file.write("\n")
                metrics_file.flush()

                if t >= next_report_time:
                    dt = t - t0
                    if not args.sync:
                        records_per_sec = num_records / dt
                        bytes_per_sec = num_bytes / dt
                        MB_per_sec = bytes_per_sec / 1e6
                        print('rank=%3d: warmed_up=%d, step=%6d, records/sec=%8.0f, MB/sec=%11.3f, records=%10d, bytes=%15d, dt=%9.3f' %
                              (rank, warmed_up, step, records_per_sec, MB_per_sec, num_records, num_bytes, dt))
                    if args.sync:
                        if rank == 0:
                            total_records = num_records * size
                            records_per_sec = total_records / dt
                            bytes_per_sec = total_bytes / dt
                            MB_per_sec = bytes_per_sec / 1e6
                            print('TOTAL:    warmed up=%d, step=%6d, records/sec=%8.0f, MB/sec=%11.3f, records=%10d, bytes=%15d, dt=%9.3f' %
                                (warmed_up, step, records_per_sec, MB_per_sec, total_records, total_bytes, dt))
                    next_report_time = t + args.report_period_sec

                # Throttle byte rate.
                if throttle:
                    while not throttle.consume(num_bytes_for_step):
                        # print('sleeping')
                        time.sleep(args.throttle_sleep_sec)

                if stop_flag:
                    print('rank=%3d: %s: complete at step %d' % (rank, hostname, step))
                    break

                step += 1

            # Use Horovod to aggregate the final counters across all processes.
            num_steps_tensor = tf.constant(step)
            num_bytes_tensor = tf.constant(num_bytes)
            total_steps_tensor = hvd.allreduce(num_steps_tensor, average=False)
            total_bytes_tensor = hvd.allreduce(num_bytes_tensor, average=False)
            total_steps, total_bytes = session.run([total_steps_tensor, total_bytes_tensor])
            if rank == 0:
                dt = stop_time - start_time
                num_records = total_steps * args.batch_size
                records_per_sec = num_records / dt
                total_GB = total_bytes / 1e9
                bytes_per_sec = total_bytes / dt
                MB_per_sec = bytes_per_sec / 1e6
                print('FINAL: number of processes: %12d' % size)
                print('FINAL: batch size:          %12d' % args.batch_size)
                print('FINAL: sync:                %12s' % args.sync)
                print('FINAL: round robin files:   %12s' % args.round_robin_files)
                print('FINAL: number of records:   %12d' % num_records)
                print('FINAL: GB:                  %12.3f' % total_GB)
                print('FINAL: elapsed sec:         %12.3f' % dt)
                print('FINAL: records/sec:         %12.0f' % records_per_sec)
                print('FINAL: MB/sec:              %12.3f' % MB_per_sec)
                print('FINAL: options:             %s' % str(args))

        if rank == 0:
            print('storage_benchmark_tensorflow: END')


def driver(args):
    print('storage_benchmark_tensorflow: BEGIN')
    print(datetime.datetime.utcnow())

    mpi_hosts = ','.join(['%s:%d' % (h, args.npernode) for h in args.host])
    num_hosts = len(args.host)

    if args.install:
        cmd = [
            'mpirun',
            '--n', str(num_hosts),
            '-npernode', '1',
            '-allow-run-as-root',
            '--host', mpi_hosts,
            '-mca', 'plm_rsh_agent', 'ssh',
            '-mca', 'plm_rsh_args', '"-p 2222"',
            'pip', 'install', '--user', '--requirement', 'requirements.txt',
        ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

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
        cmd = [
            'mpirun',
            '--n', str(num_hosts),
            '-npernode', '1',
            '-allow-run-as-root',
            '--host', mpi_hosts,
            '-mca', 'plm_rsh_agent', 'ssh',
            '-mca', 'plm_rsh_args', '"-p 2222"',
            '/scripts/drop_caches.sh',
        ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

    # Run the same command that launched this driver but with mpirun.
    cmd = [
        'mpirun',
        '--allow-run-as-root',
        '--np', str(args.np),
        '--host', mpi_hosts,
        '-map-by', 'node',
        '-bind-to', 'socket',
        '--report-bindings',
        '-mca', 'plm_rsh_agent', 'ssh',
        '-mca', 'plm_rsh_args', '"-p 2222"',
        '-mca', 'pml', 'ob1',
        '-mca', 'btl', '^openib',
        '-mca', 'btl_tcp_if_exclude', 'lo,docker0',
        '-x', 'LD_LIBRARY_PATH',
        '-x', 'PATH',
        '-x', 'CUDA_VISIBLE_DEVICES=""',
    ]
    cmd += sys.argv
    cmd += ['--worker', 'true']
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

    print(datetime.datetime.utcnow())
    print('storage_benchmark_tensorflow: END')


def main():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=['storage_benchmark_tensorflow.yaml'],
    )
    parser.add('--batch_size', type=int, default=256, help='Number of records per batch')
    parser.add('--config', '-c', required=False, is_config_file=True, help='config file path')
    parser.add('--dataset_buffer_size', type=int, nargs='?')
    parser.add('--flush', type=parse_bool, default=False, help='Flush caches')
    parser.add('--host', '-H', action='append', required=True, help='List of hosts on which to invoke processes.')
    parser.add('--input_file_specs', '-i', action='append', help='Input file spec', required=True)
    parser.add('--install', type=parse_bool, default=False)
    parser.add('--isilon_host',
               help='IP address or hostname of an Isilon node. You must enable password-less SSH.')
    parser.add('--isilon_user', default='root',
               help='SSH user used to connect to Isilon.')
    parser.add('--metrics_directory', default='/tmp')
    parser.add('--noop', type=parse_bool, default=False)
    parser.add('--np', type=int, default=1, help='Run this many copies of the program on the given nodes.')
    parser.add('--npernode', type=int, default=80, help='On each node, launch this many processes.')
    parser.add('--num_threads', type=int, default=0, help='Number of threads')
    parser.add('--parallel_interleave_cycle_length', type=int, default=0)
    parser.add('--prefetch_records', type=int, nargs='?')
    parser.add('--report_period_sec', type=float, default=2.0, help='Report statistics with this period in seconds')
    parser.add('--round_robin_files', type=parse_bool, default=False)
    parser.add('--run_sec', type=float, default=60*60*4, help='Run time in seconds')
    parser.add('--sync', type=parse_bool, default=False, help='Synchronize workers after each batch')
    parser.add('--test_uuid', default=str(uuid.uuid4()))
    parser.add('--throttle_sleep_sec', type=float, default=0.01)
    parser.add('--throttle_total_rate_bytes_per_sec', type=float, default=0, help='If 0, unthrottled')
    parser.add('--warmup_sec', type=float, default=10.0, help='Warm-up time in seconds')
    parser.add('--warn_latency_sec', type=float, default=4.0, help='Warn if read latency exceeds this many seconds')
    parser.add('--worker', type=parse_bool, default=False)
    args = parser.parse_args()
    print('storage_benchmark_tensorflow: Options: %s' % str(args))

    if args.worker:
        if not args.noop:
            hvd.init()
            worker(hvd.rank(), hvd.size(), args)
    else:
        driver(args)


if __name__ == '__main__':
    main()
