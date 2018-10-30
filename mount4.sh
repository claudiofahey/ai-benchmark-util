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
cat hosts | xargs -i -P 0 ssh root@{} mount -t nfs 10.55.66.211:/ifs /mnt/isilon

common_mount_opts="-o rsize=524288,wsize=524288,nolock,soft,timeo=50,retrans=1,proto=tcp"

# Mount 1 to Isilon 40gige-1, DGX1-1 to Isilon Node 1.
# DGX1-9 mounts to Isilon Node 1-4.
mount_opts="${common_mount_opts} /mnt/isilon1"
ssh root@DGX1-1 sudo mount -t nfs 10.55.66.211:/ifs ${mount_opts} &
ssh root@DGX1-2 sudo mount -t nfs 10.55.66.212:/ifs ${mount_opts} &
ssh root@DGX1-3 sudo mount -t nfs 10.55.66.213:/ifs ${mount_opts} &
ssh root@DGX1-4 sudo mount -t nfs 10.55.66.214:/ifs ${mount_opts} &
ssh root@DGX1-5 sudo mount -t nfs 10.55.66.215:/ifs ${mount_opts} &
ssh root@DGX1-6 sudo mount -t nfs 10.55.66.216:/ifs ${mount_opts} &
ssh root@DGX1-7 sudo mount -t nfs 10.55.66.217:/ifs ${mount_opts} &
ssh root@DGX1-8 sudo mount -t nfs 10.55.66.218:/ifs ${mount_opts} &
ssh root@DGX1-9 sudo mount -t nfs 10.55.66.211:/ifs ${mount_opts} &
wait

# Mount 2 to Isilon 40gige-2, DGX1-1 to Isilon Node 1.
mount_opts="${common_mount_opts} /mnt/isilon2"
ssh root@DGX1-1 sudo mount -t nfs 10.55.5.211:/ifs ${mount_opts} &
ssh root@DGX1-2 sudo mount -t nfs 10.55.5.212:/ifs ${mount_opts} &
ssh root@DGX1-3 sudo mount -t nfs 10.55.5.213:/ifs ${mount_opts} &
ssh root@DGX1-4 sudo mount -t nfs 10.55.5.214:/ifs ${mount_opts} &
ssh root@DGX1-5 sudo mount -t nfs 10.55.5.215:/ifs ${mount_opts} &
ssh root@DGX1-6 sudo mount -t nfs 10.55.5.216:/ifs ${mount_opts} &
ssh root@DGX1-7 sudo mount -t nfs 10.55.5.217:/ifs ${mount_opts} &
ssh root@DGX1-8 sudo mount -t nfs 10.55.5.218:/ifs ${mount_opts} &
ssh root@DGX1-9 sudo mount -t nfs 10.55.5.212:/ifs ${mount_opts} &
wait

# Mount 3 to Isilon 40gige-1, DGX1-1 to Isilon Node 3.
mount_opts="${common_mount_opts} /mnt/isilon3"
ssh root@DGX1-1 sudo mount -t nfs 10.55.66.213:/ifs ${mount_opts} &
ssh root@DGX1-2 sudo mount -t nfs 10.55.66.214:/ifs ${mount_opts} &
ssh root@DGX1-3 sudo mount -t nfs 10.55.66.215:/ifs ${mount_opts} &
ssh root@DGX1-4 sudo mount -t nfs 10.55.66.216:/ifs ${mount_opts} &
ssh root@DGX1-5 sudo mount -t nfs 10.55.66.217:/ifs ${mount_opts} &
ssh root@DGX1-6 sudo mount -t nfs 10.55.66.218:/ifs ${mount_opts} &
ssh root@DGX1-7 sudo mount -t nfs 10.55.66.211:/ifs ${mount_opts} &
ssh root@DGX1-8 sudo mount -t nfs 10.55.66.212:/ifs ${mount_opts} &
ssh root@DGX1-9 sudo mount -t nfs 10.55.66.213:/ifs ${mount_opts} &
wait

# Mount 4 to Isilon 40gige-2, DGX1-1 to Isilon Node 3.
mount_opts="${common_mount_opts} /mnt/isilon4"
ssh root@DGX1-1 sudo mount -t nfs 10.55.5.213:/ifs ${mount_opts} &
ssh root@DGX1-2 sudo mount -t nfs 10.55.5.214:/ifs ${mount_opts} &
ssh root@DGX1-3 sudo mount -t nfs 10.55.5.215:/ifs ${mount_opts} &
ssh root@DGX1-4 sudo mount -t nfs 10.55.5.216:/ifs ${mount_opts} &
ssh root@DGX1-5 sudo mount -t nfs 10.55.5.217:/ifs ${mount_opts} &
ssh root@DGX1-6 sudo mount -t nfs 10.55.5.218:/ifs ${mount_opts} &
ssh root@DGX1-7 sudo mount -t nfs 10.55.5.211:/ifs ${mount_opts} &
ssh root@DGX1-8 sudo mount -t nfs 10.55.5.212:/ifs ${mount_opts} &
ssh root@DGX1-9 sudo mount -t nfs 10.55.5.214:/ifs ${mount_opts} &
wait
