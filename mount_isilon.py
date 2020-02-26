#!/usr/bin/env python3
#
# Copyright (c) 2018 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#

"""
This script will SSH to multiple hosts and mount multiple Isilon interfaces on each host.
Isilon IP addresses are distributed in a round-robin fashion over hosts.

Installation of prerequisites:
  sudo apt install python3-pip
  pip3 install setuptools
  pip3 install --requirement requirements.txt
"""

import configargparse
import subprocess


def main():
    parser = configargparse.ArgParser(
        description='SSH to multiple hosts and mount all Isilon interfaces on each host.',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=['mount_isilon.yaml'],
    )
    parser.add_argument('--export', default='/ifs')
    parser.add_argument('--host', '-H', action='append', required=True,
                        help='List of hosts on which to run mount commands.')
    parser.add_argument('--isilon_ip', '-i', action='append', required=True,
                        help='Isilon IP address to use as a mount target.')
    parser.add_argument('--isilon_ip_count', '-c', type=int, default=1,
                        help='Total number of Isilon IP addresses. '
                        'If --isilon_ip specifies the first IP address only, consecutive IP addresses will be calculated.')
    parser.add_argument('--noop', action='store_true', default=False,
                        help='Print all commands but do not execute them.')
    parser.add_argument('--nomount', dest='mount', action='store_false', default=True,
                        help='Unmount existing but do not mount.')
    parser.add_argument('--skip_hosts', type=int, default=1,
                        help='Do not mount /mnt/isilon on the first skip_hosts hosts.')
    parser.add_argument('--user', default='')
    args = parser.parse_args()

    print('# Initial Arguments: ' + str(args))

    # Calculate consecutive Isilon IP addresses.
    if len(args.isilon_ip) == 1 and len(args.isilon_ip) < args.isilon_ip_count:
        first_isilon_ip = [int(octet) for octet in args.isilon_ip[0].split('.')]
        args.isilon_ip = []
        for i in range(args.isilon_ip_count):
            isilon_ip = '.'.join([str(octet) for octet in first_isilon_ip[0:3] + [first_isilon_ip[3] + i]])
            args.isilon_ip += [isilon_ip]

    print('# Expanded Arguments: ' + str(args))

    assert len(args.isilon_ip) == args.isilon_ip_count

    ssh_user = '' if args.user == '' else args.user + '@'

    # Mount /mnt/isilon on all except first host.
    # This will use default mount parameters.
    for host_index, host in enumerate(args.host):
        if host_index < args.skip_hosts: continue
        isilon_ip = args.isilon_ip[host_index % len(args.isilon_ip)]

        cmd = [
            'ssh',
            '%s%s' % (ssh_user, host),
            'sudo',
            'umount',
            '/mnt/isilon',
        ]
        print(' '.join(cmd))
        if not args.noop: subprocess.run(cmd, check=False)

        if args.mount:
            cmd = [
                'ssh',
                '%s%s' % (ssh_user, host),
                'sudo',
                'mkdir', '-p', '/mnt/isilon',
            ]
            print(' '.join(cmd))
            if not args.noop: subprocess.run(cmd, check=True)

            cmd = [
                'ssh',
                '%s%s' % (ssh_user, host),
                'sudo',
                'mount',
                '-t', 'nfs',
                '%s:%s' % (isilon_ip, args.export),
                '/mnt/isilon',
            ]
            print(' '.join(cmd))
            if not args.noop: subprocess.run(cmd, check=True)

    # Mount /mnt/isilon1, /mnt/isilon2, ...
    # This will use mount parameters optimized for high performance reads. nolock is used.
    for host_index, host in enumerate(args.host):
        for mount_index in range(len(args.isilon_ip)):
            mount_number = mount_index + 1
            isilon_ip = args.isilon_ip[(mount_index + host_index) % len(args.isilon_ip)]

            cmd = [
                'ssh',
                '%s%s' % (ssh_user, host),
                'sudo',
                'umount',
                '/mnt/isilon%d' % mount_number,
            ]
            print(' '.join(cmd))
            if not args.noop: subprocess.run(cmd, check=False)

            if args.mount:
                cmd = [
                    'ssh',
                    '%s%s' % (ssh_user, host),
                    'sudo',
                    'mkdir', '-p', '/mnt/isilon%d' % mount_number,
                ]
                print(' '.join(cmd))
                if not args.noop: subprocess.run(cmd, check=True)

                cmd = [
                    'ssh',
                    '%s%s' % (ssh_user, host),
                    'sudo',
                    'mount',
                    '-t', 'nfs',
                    '-o', 'rsize=524288,wsize=524288,nolock,soft,timeo=50,retrans=1,proto=tcp',
                    '%s:%s' % (isilon_ip, args.export),
                    '/mnt/isilon%d' % mount_number,
                ]
                print(' '.join(cmd))
                if not args.noop: subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
