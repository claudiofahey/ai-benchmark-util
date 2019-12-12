#!/usr/bin/python3 -u
#
# Copyright (c) 2019 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
This runs various pbrun commands and records details to a JSON file
for performance analysis.
This can be run standalone or it can be submitted as a Slurm job using
submit_slurm_jobs.py or submit_slurm_job_1.sh
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


def run_system_command(cmd, rec, shell=False, env=None, noop=False, raise_on_error=True):
    t0 = datetime.datetime.utcnow()
    return_code, output, errors = system_command(
        cmd,
        print_command=True,
        print_output=True,
        raise_on_error=False,
        env=env,
        shell=shell,
        noop=noop,
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
    if not noop and raise_on_error and return_code != 0:
        raise Exception('System command returned %d: %s' % (return_code, cmd))


def process_sample(args):
    logging.info('BEGIN')
    record_uuid = str(uuid.uuid4())
    sample_id = args.sample_id
    hostname = socket.gethostname()
    logging.info('record_uuid=%s' % record_uuid)
    logging.info('sample_id=%s' % sample_id)
    logging.info('hostname=%s' % hostname)

    t0 = datetime.datetime.utcnow()

    rec = {}
    rec['batch_uuid'] = args.batch_uuid
    rec['record_uuid'] = record_uuid
    rec['sample_id'] = sample_id
    rec['hostname'] = hostname
    rec['args'] = args.__dict__

    exception = None

    try:
        input_dir = os.path.join(args.input_dir, sample_id)
        output_dir = os.path.join(args.output_dir, sample_id)
        temp_dir = os.path.join(args.temp_dir, sample_id)
        rec['input_dir'] = input_dir
        rec['output_dir'] = output_dir
        rec['temp_dir'] = temp_dir

        logging.debug('input_dir=%s' % input_dir)
        logging.debug('output_dir=%s' % output_dir)
        logging.debug('temp_dir=%s' % temp_dir)

        if not args.noop and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create copy of Parabricks installation just for this process.
        # If installation directories are used by different processes concurrently, corruption
        # in the Singularity image may occur.
        cmd = ['tar', '-xzvf', args.parabricks_install_tgz_file, '-C', temp_dir]
        system_command(
            cmd,
            print_command=True,
            print_output=True,
            raise_on_error=True,
            shell=False,
            noop=args.noop,
        )
        pbrun_file_name = os.path.join(temp_dir, 'parabricks', 'pbrun')
        logging.debug('pbrun_file_name=%s' % pbrun_file_name)
        assert os.path.exists(pbrun_file_name)

        # Slurm sets CUDA_VISIBLE_DEVICES but pbrun requires NVIDIA_VISIBLE_DEVICES.
        env = os.environ.copy()
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')
        logging.info('cuda_visible_devices=%s' % cuda_visible_devices)
        num_gpus = len(cuda_visible_devices.split(','))
        logging.info('num_gpus=%d' % num_gpus)
        env['NVIDIA_VISIBLE_DEVICES'] = cuda_visible_devices
        rec['env'] = env
        rec['cuda_visible_devices'] = cuda_visible_devices
        rec['num_gpus'] = num_gpus

        fq_pairs = []
        fq_file_sizes = []
        for i in range(args.max_num_fq_pairs):
            pair = []
            for j in range(1,3):
                filename = os.path.join(input_dir, '%d_%d.fq.gz' % (i, j))
                if os.path.isfile(filename):
                    pair += [filename]
                    fq_file_sizes += [os.path.getsize(filename)]
            if pair:
                fq_pairs += [pair]
        logging.debug('fq_pairs=%s' % str(fq_pairs))
        rec['fq_pairs'] = fq_pairs
        logging.info('fq_file_sizes=%s' % str(fq_file_sizes))
        rec['fq_file_sizes'] = fq_file_sizes

        in_fq_cmd = []
        for i, fq_pair in enumerate(fq_pairs):
            header = '@RG\\tID:%d\\tLB:lib1\\tPL:bar\\tSM:%s\\tPU:%d' % (i, sample_id, i)
            in_fq_cmd += ['--in-fq'] + fq_pair + [header]

        logging.debug('in_fq_cmd=%s' % str(in_fq_cmd))

        bam_file_name = os.path.join(output_dir, '%s.bam' % sample_id)
        gvcf_file_name = os.path.join(output_dir, '%s.g.vcf' % sample_id)
        dv_gvcf_file_name = os.path.join(output_dir, '%s_dv.g.vcf' % sample_id)

        if args.fq2bam:
            cmd = [
                pbrun_file_name,
                'fq2bam',
                '--ref', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.fasta'),
                '--out-bam', bam_file_name,
                '--out-recal-file', os.path.join(output_dir, '%s.txt' % sample_id),
                '--knownSites', os.path.join(args.reference_files_dir, 'Mills_and_1000G_gold_standard.indels.hg38.vcf.gz'),
                '--knownSites', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.dbsnp138.vcf'),
                '--tmp-dir', temp_dir,
                '--num-gpus', '%d' % num_gpus,
            ]
            cmd += in_fq_cmd
            rec['fq2bam_result'] = {}
            run_system_command(cmd, rec['fq2bam_result'], env=env, noop=args.noop)

        if args.germline:
            cmd = [
                pbrun_file_name,
                'germline',
                '--ref', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.fasta'),
                '--out-bam', bam_file_name,
                '--out-recal-file', os.path.join(output_dir, '%s.txt' % sample_id),
                '--knownSites', os.path.join(args.reference_files_dir, 'Mills_and_1000G_gold_standard.indels.hg38.vcf.gz'),
                '--knownSites', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.dbsnp138.vcf'),
                '--out-variants', gvcf_file_name,
                '--gvcf',
                '--tmp-dir', temp_dir,
                '--num-gpus', '%d' % num_gpus,
            ]
            cmd += in_fq_cmd
            rec['germline_result'] = {}
            run_system_command(cmd, rec['germline_result'], env=env, noop=args.noop)
            rec['haplotypecaller_gvcf_file_size_bytes'] = os.path.getsize(gvcf_file_name)

        rec['bam_file_size_bytes'] = os.path.getsize(bam_file_name)
        logging.debug('bam_file_size_bytes=%d' % rec['bam_file_size_bytes'])

        if args.haplotypecaller:
            cmd = [
                pbrun_file_name,
                'haplotypecaller',
                '--ref', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.fasta'),
                '--in-bam', bam_file_name,
                '--in-recal-file', os.path.join(output_dir, '%s.txt' % sample_id),
                '--out-variants', gvcf_file_name,
                '--gvcf',
                '--tmp-dir', temp_dir,
                '--num-gpus', '%d' % num_gpus,
            ]
            rec['haplotypecaller_result'] = {}
            run_system_command(cmd, rec['haplotypecaller_result'], env=env, noop=args.noop)
            rec['haplotypecaller_gvcf_file_size_bytes'] = os.path.getsize(gvcf_file_name)

        if args.deepvariant:
            # deepvariant uses the bam output of fq2bam or germline.
            cmd = [
                pbrun_file_name,
                'deepvariant',
                '--ref', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.fasta'),
                '--in-bam', bam_file_name,
                '--out-variants', dv_gvcf_file_name,
                '--gvcf',
                '--tmp-dir', temp_dir,
                '--num-gpus', '%d' % num_gpus,
            ]
            rec['deepvariant_result'] = {}
            run_system_command(cmd, rec['deepvariant_result'], env=env, noop=args.noop)
            rec['deepvariant_gvcf_file_size_bytes'] = os.path.getsize(dv_gvcf_file_name)

        if not args.noop and os.path.exists(temp_dir): shutil.rmtree(temp_dir)

    except Exception as e:
        exception = e
        rec['error'] = True

    t1 = datetime.datetime.utcnow()
    td = t1 - t0
    rec['utc_begin'] = t0.isoformat()
    rec['utc_end'] = t1.isoformat()
    rec['elapsed_sec'] = time_duration_to_seconds(td)

    if args.summary_file:
        record_result(rec, args.summary_file)

    logging.info('END')
    if exception: raise exception


def main():
    parser = configargparse.ArgParser(
        description='Run Parabricks germline pipeline.',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add('--config', '-c', default='parabricks_germline_pipeline.yaml',
               required=False, is_config_file=True, help='config file path')
    parser.add('--batch_uuid', required=False)
    parser.add('--deepvariant', type=parse_bool, default=False)
    parser.add('--input_dir', help='Input directory', required=True)
    parser.add('--fq2bam', type=parse_bool, default=False)
    parser.add('--germline', type=parse_bool, default=False)
    parser.add('--haplotypecaller', type=parse_bool, default=False)
    parser.add('--log_level', type=int, default=logging.INFO, help='10=DEBUG,20=INFO')
    parser.add('--max_num_fq_pairs', default=55, type=int)
    parser.add('--mem_per_cpu', type=int, default=0, help='for reporting only')
    parser.add('--noop', type=parse_bool, default=False)
    parser.add('--num_cpus', type=int, default=0, help='for reporting only')
    parser.add('--output_dir', help='Output directory', required=True)
    parser.add('--parabricks_install_tgz_file', default='parabricks_install.tar.gz')
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
