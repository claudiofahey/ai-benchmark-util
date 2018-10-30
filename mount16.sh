#!/usr/bin/env bash
# This script mounts each DGX-1 to each Isilon interface on 8 Isilon nodes.

set -x

for mount in `seq 1 16`;
    do
    cat hosts | xargs -i -P 0 ssh root@{} "umount /mnt/isilon${mount} ; mkdir -p /mnt/isilon${mount}"
    done

common_mount_opts="-o rsize=524288,wsize=524288,nolock,soft,timeo=50,retrans=1,proto=tcp"

cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.66.211:/ifs "${common_mount_opts} /mnt/isilon1"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.66.212:/ifs "${common_mount_opts} /mnt/isilon2"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.66.213:/ifs "${common_mount_opts} /mnt/isilon3"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.66.214:/ifs "${common_mount_opts} /mnt/isilon4"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.66.215:/ifs "${common_mount_opts} /mnt/isilon5"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.66.216:/ifs "${common_mount_opts} /mnt/isilon6"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.66.217:/ifs "${common_mount_opts} /mnt/isilon7"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.66.218:/ifs "${common_mount_opts} /mnt/isilon8"

cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.5.211:/ifs  "${common_mount_opts} /mnt/isilon9"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.5.212:/ifs  "${common_mount_opts} /mnt/isilon10"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.5.213:/ifs  "${common_mount_opts} /mnt/isilon11"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.5.214:/ifs  "${common_mount_opts} /mnt/isilon12"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.5.215:/ifs  "${common_mount_opts} /mnt/isilon13"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.5.216:/ifs  "${common_mount_opts} /mnt/isilon14"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.5.217:/ifs  "${common_mount_opts} /mnt/isilon15"
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.5.218:/ifs  "${common_mount_opts} /mnt/isilon16"
