#!/usr/bin/env bash
set -ex
HOSTS_FILE=$1
JOB_FILE=$(readlink -f $2)

echo `date -u`: BEGIN
echo Number of clients: `wc -l ${HOSTS_FILE}`
cat ${HOSTS_FILE}
cat ${JOB_FILE}

cat ${HOSTS_FILE} | xargs -i -P 0 ssh {} fio ${JOB_FILE} --filename /mnt/isilon1/tmp/fio/{}.tmp

echo `date -u`: END
