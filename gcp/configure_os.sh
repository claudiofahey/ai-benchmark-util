#!/usr/bin/env bash
set -ex

FIRST_WORKER=001
LAST_WORKER=018

#seq -w ${FIRST_WORKER} ${LAST_WORKER} | xargs -i -P 0 ssh dl-worker-{} \
#sudo /opt/deeplearning/install-driver.sh
#
#seq -w ${FIRST_WORKER} ${LAST_WORKER} | xargs -i -P 1 ssh dl-worker-{} \
#nvidia-smi

seq -w ${FIRST_WORKER} ${LAST_WORKER} | xargs -i -P 0 ssh dl-worker-{} \
sudo "apt-get -y update && sudo apt-get -y install nfs-common nmon fio"

seq -w ${FIRST_WORKER} ${LAST_WORKER} | xargs -i -P 0 ssh dl-worker-{} \
sudo mkdir -p /mnt/filestore1

seq -w ${FIRST_WORKER} ${LAST_WORKER} | xargs -i -P 0 ssh dl-worker-{} \
sudo mount -t nfs 10.108.10.90:/share1 /mnt/filestore1

seq -w ${FIRST_WORKER} ${LAST_WORKER} | xargs -i -P 0 ssh dl-worker-{} \
ls -lh /mnt/filestore1/THIS_IS_dl-nas-1-share1.txt | wc -l
