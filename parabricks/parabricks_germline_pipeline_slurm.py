#!/usr/bin/python3 -u
#
# Copyright (c) 2019 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
This runs various pbrun commands and records details to a JSON file
for performance analysis.
This can be run standalone or it can be submitted as a job using sbatch.
"""

import configargparse
import datetime
import logging
import os
import shutil
import socket
import sys
import time
import uuid
from p3_test_driver.system_command import system_command, time_duration_to_seconds
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


def record_result(result, result_filename):
    now = datetime.datetime.utcnow()
    rec = result.copy()
    filename_timestamp = now.strftime('%Y%m%dT%H%M%S')
    var_dict = result.copy()
    var_dict['timestamp'] = filename_timestamp
    result_filename = result_filename % var_dict
    logging.info('Recording results to file %s' % result_filename)
    mkdir_for_file(result_filename)
    append_to_json_file([rec], result_filename, sort_keys=True, indent=True)


def run_system_command(cmd, rec, shell=False, env=None):
    t0 = datetime.datetime.utcnow()
    return_code, output, errors = system_command(
        cmd,
        print_command=True,
        print_output=True,
        raise_on_error=False,
        env=env,
        shell=shell,
    )
    t1 = datetime.datetime.utcnow()
    td = t1 - t0
    logging.info('exit_code=%d' % return_code)
    rec['command'] = cmd
    rec['utc_begin'] = t0.isoformat()
    rec['utc_end'] = t1.isoformat()
    rec['elapsed_sec'] = time_duration_to_seconds(td)
    rec['error'] = (return_code != 0)
    rec['exit_code'] = return_code
    rec['command_timed_out'] = (return_code == -1)
    rec['output'] = output
    rec['errors'] = errors


def process_sample(args):
    sample_id = args.sample_id
    hostname = socket.gethostname()

    def print_prefix():
        return '%s: %s' % (hostname, sample_id)

    logging.info('%s: BEGIN' % print_prefix())

    t0 = datetime.datetime.utcnow()

    rec = {}
    rec['record_uuid'] = str(uuid.uuid4())
    rec['sample_id'] = sample_id
    rec['hostname'] = hostname
    rec['args'] = args.__dict__

    input_dir = os.path.join(args.input_dir, sample_id)
    output_dir = os.path.join(args.output_dir, sample_id)
    temp_dir = os.path.join(args.temp_dir, sample_id)
    rec['input_dir'] = input_dir
    rec['output_dir'] = output_dir
    rec['temp_dir'] = output_dir

    logging.debug('%s: input_dir=%s' % (print_prefix, input_dir))
    logging.debug('%s: output_dir=%s' % (print_prefix, output_dir))
    logging.debug('%s: temp_dir=%s' % (print_prefix, temp_dir))

    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Slurm sets CUDA_VISIBLE_DEVICES but pbrun requires NVIDIA_VISIBLE_DEVICES.
    env = os.environ.copy()
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    logging.info('%s: cuda_visible_devices=%s' % (print_prefix(), cuda_visible_devices))
    num_gpus = len(cuda_visible_devices.split(','))
    logging.info('%s: num_gpus=%d' % (print_prefix(), num_gpus))
    env['NVIDIA_VISIBLE_DEVICES'] = cuda_visible_devices
    rec['env'] = env
    rec['cuda_visible_devices'] = cuda_visible_devices
    rec['num_gpus'] = num_gpus

    fq_pairs = []
    for i in range(args.max_num_fq_pairs):
        pair = []
        for j in range(1,3):
            filename = os.path.join(input_dir, '%d_%d.fq.gz' % (i, j))
            if os.path.isfile(filename):
                pair += [filename]
        if pair:
            fq_pairs += [pair]
    logging.debug('%s: fq_pairs=%s' % (print_prefix(), str(fq_pairs)))
    rec['fq_pairs'] = fq_pairs

    in_fq_cmd = []
    for i, fq_pair in enumerate(fq_pairs):
        header = '@RG\\tID:%d\\tLB:lib1\\tPL:bar\\tSM:%s\\tPU:%d' % (i, sample_id, i)
        in_fq_cmd += ['--in-fq'] + fq_pair + [header]

    logging.debug('%s: in_fq_cmd=%s' % (print_prefix, str(in_fq_cmd)))

    if args.fq2bam:
        cmd = [
            'pbrun',
            'fq2bam',
            '--ref', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.fasta'),
            '--out-bam', os.path.join(output_dir, '%s.bam' % sample_id),
            '--tmp-dir', temp_dir,
            '--num-gpus', '%d' % num_gpus,
        ]
        cmd += in_fq_cmd
        rec['fq2bam_result'] = {}
        run_system_command(cmd, rec['fq2bam_result'], env=env)

    if args.germline:
        cmd = [
            'pbrun',
            'germline',
            '--ref', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.fasta'),
            '--out-bam', os.path.join(output_dir, '%s.bam' % sample_id),
            '--out-recal-file', os.path.join(output_dir, '%s.txt' % sample_id),
            '--knownSites', os.path.join(args.reference_files_dir, 'Mills_and_1000G_gold_standard.indels.hg38.vcf.gz'),
            '--knownSites', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.dbsnp138.vcf'),
            '--out-variants', os.path.join(output_dir, '%s.g.vcf.gz' % sample_id),
            '--gvcf',
            '--tmp-dir', temp_dir,
            '--num-gpus', '%d' % num_gpus,
        ]
        cmd += in_fq_cmd
        rec['germline_result'] = {}
        run_system_command(cmd, rec['germline_result'], env=env)

    if args.deepvariant:
        # deepvariant uses the bam output of fq2bam or germline.
        cmd = [
            'pbrun',
            'deepvariant',
            '--ref', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.fasta'),
            '--in-bam', os.path.join(output_dir, '%s.bam' % sample_id),
            '--out-variants', os.path.join(output_dir, '%s_dv.g.vcf.gz' % sample_id),
            '--gvcf',
            '--tmp-dir', temp_dir,
            '--num-gpus', '%d' % num_gpus,
        ]
        # cmd += in_fq_cmd
        rec['deepvariant_result'] = {}
        run_system_command(cmd, rec['deepvariant_result'], env=env)

    t1 = datetime.datetime.utcnow()
    td = t1 - t0
    rec['utc_begin'] = t0.isoformat()
    rec['utc_end'] = t1.isoformat()
    rec['elapsed_sec'] = time_duration_to_seconds(td)

    if args.summary_file:
        record_result(rec, args.summary_file)

    logging.info('%s: END' % print_prefix())


def main():
    parser = configargparse.ArgParser(
        description='Run Parabricks germline pipeline.',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add('--config', '-c', default='parabricks_germline_pipeline.yaml',
               required=False, is_config_file=True, help='config file path')
    parser.add('--deepvariant', type=parse_bool, default=False)
    parser.add('--input_dir', help='Input directory', required=True)
    parser.add('--fq2bam', type=parse_bool, default=False)
    parser.add('--germline', type=parse_bool, default=False)
    parser.add('--log_level', type=int, default=logging.INFO, help='10=DEBUG,20=INFO')
    parser.add('--max_num_fq_pairs', default=55, type=int)
    parser.add('--output_dir', help='Output directory', required=True)
    parser.add('--reference_files_dir', required=True)
    parser.add('--summary_file', required=False)
    parser.add('--sample_id', required=True)
    parser.add('--temp_dir', default='/tmp', help='Temp directory', required=True)
    args = parser.parse_args()

    # Initialize logging
    root_logger = logging.getLogger()
    root_logger.setLevel(args.log_level)
    console_handler = logging.StreamHandler(sys.stdout)
    logging.Formatter.converter = time.gmtime
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    root_logger.addHandler(console_handler)

    logging.debug('args=%s' % str(args))
    process_sample(args)


if __name__ == '__main__':
    main()
