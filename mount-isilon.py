#!/usr/bin/env python3

import subprocess


hosts = [
    "dgx2-1", 
    "dgx2-2", 
    "dgx2-3",
]
mount = True

for host in hosts[1:]:
        isilon_ip = "172.28.10.151"
        cmd = [
            "ssh",
            "root@%s" % host,
            "mkdir", "-p", "/mnt/isilon",
        ]
        subprocess.run(cmd, check=True)

        cmd = [
            "ssh",
            "root@%s" % host,
            "umount",
            "/mnt/isilon",
        ]
        print(cmd)
        subprocess.run(cmd, check=False)

        if mount:
            cmd = [
                "ssh",
                "root@%s" % host,
                "mount",
                "-t", "nfs",
                "%s:/ifs" % isilon_ip,
                "/mnt/isilon"
            ]
            print(cmd)
            subprocess.run(cmd, check=True)

for host in hosts:
    for mount_number in range(1, 16+1):
        isilon_ip = "10.200.10.%d" % (151 + mount_number - 1)

        cmd = [
            "ssh",
            "root@%s" % host,
            "mkdir", "-p", "/mnt/isilon%d" % mount_number,
        ]
        subprocess.run(cmd, check=True)

        cmd = [
            "ssh",
            "root@%s" % host,
            "umount",
            "/mnt/isilon%d" % mount_number,
        ]
        print(cmd)
        subprocess.run(cmd, check=False)

        if mount:
            cmd = [
                "ssh",
                "root@%s" % host,
                "mount",
                "-t", "nfs",
                "-o", "rsize=524288,wsize=524288,nolock,soft,timeo=50,retrans=1,proto=tcp",
                "%s:/ifs" % isilon_ip,
                "/mnt/isilon%d" % mount_number,
            ]
            print(cmd)
            subprocess.run(cmd, check=True)
