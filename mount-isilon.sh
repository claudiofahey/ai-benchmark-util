#!/usr/bin/env bash
set -x

cat hosts | xargs -i -P 0 ssh root@{} umount /mnt/isilon
cat hosts | xargs -i -P 0 ssh root@{} umount /mnt/isilon1
cat hosts | xargs -i -P 0 ssh root@{} umount /mnt/isilon2
cat hosts | xargs -i -P 0 ssh root@{} umount /mnt/isilon3
cat hosts | xargs -i -P 0 ssh root@{} umount /mnt/isilon4

cat hosts | xargs -i -P 0 ssh root@{} sudo mkdir -p /mnt/isilon
cat hosts | xargs -i -P 0 ssh root@{} sudo mkdir -p /mnt/isilon1
cat hosts | xargs -i -P 0 ssh root@{} sudo mkdir -p /mnt/isilon2
cat hosts | xargs -i -P 0 ssh root@{} sudo mkdir -p /mnt/isilon3
cat hosts | xargs -i -P 0 ssh root@{} sudo mkdir -p /mnt/isilon4

# /mnt/isilon uses default mount options.
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.200.10.151:/ifs /mnt/isilon

common_mount_opts="-o rsize=524288,wsize=524288,nolock,soft,timeo=50,retrans=1,proto=tcp"

cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.200.10.152:/ifs /mnt/isilon1

