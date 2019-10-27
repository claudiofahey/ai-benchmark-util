#!/usr/bin/python3 -u
#
# Copyright (c) 2019 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#


"""
"""

import configargparse
import os
import shutil
import subprocess


def process_sample(args):
    sample_id = args.sample_id
    print_prefix = '%s' % (sample_id, )
    print('%s: BEGIN' % print_prefix)
    input_dir = os.path.join(args.input_dir, sample_id)
    # print('%s: input_dir=%s' % (print_prefix, input_dir))
    output_dir = os.path.join(args.output_dir, sample_id)
    # print('%s: output_dir=%s' % (print_prefix, output_dir))
    temp_dir = os.path.join(args.temp_dir, sample_id)
    # print('%s: temp_dir=%s' % (print_prefix, temp_dir))

    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    env = os.environ.copy()
    env['NVIDIA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    print('%s: NVIDIA_VISIBLE_DEVICES=%s' % (print_prefix, env['NVIDIA_VISIBLE_DEVICES']))

    fq_pairs = [[os.path.join(input_dir, '%d_%d.fq.gz' % (i, j)) for j in range(1,3)] for i in range(args.num_fq_pairs)]
    in_fq_cmd = [b for a in (['--in-fq'] + p for p in fq_pairs) for b in a]
    # print('%s: fq_pairs=%s' % (print_prefix, str(fq_pairs)))
    # print('%s: in_fq_cmd=%s' % (print_prefix, str(in_fq_cmd)))

    if True:
        cmd = [
            # 'echo',
            'pbrun',
            'fq2bam',
            '--ref', os.path.join(args.reference_files_dir, 'Homo_sapiens_assembly38.fasta'),
            '--out-bam', os.path.join(output_dir, '%s.bam' % sample_id),
            '--tmp-dir', temp_dir,
            '--num-gpus', '%d' % args.num_gpus,
        ]
        cmd += in_fq_cmd
        print(' '.join(cmd))
        subprocess.run(cmd, env=env, check=True)

    print('%s: END' % print_prefix)


def main():
    parser = configargparse.ArgParser(
        description='Run Parabricks germline pipeline.',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add('--config', '-c', default='parabricks_germline_pipeline.yaml',
               required=False, is_config_file=True, help='config file path')
    parser.add('--input_dir', help='Input directory', required=True)
    parser.add('--output_dir', help='Output directory', required=True)
    parser.add('--reference_files_dir', required=True)
    parser.add('--sample_id', required=True)
    parser.add('--temp_dir', default='/tmp', help='Temp directory', required=True)
    parser.add('--num_fq_pairs', default=55, type=int)
    parser.add('--num_gpus', default=4, type=int)
    args = parser.parse_args()
    print(args)
    process_sample(args)


if __name__ == '__main__':
    main()
