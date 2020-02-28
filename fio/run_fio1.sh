#!/usr/bin/env bash
set -ex
HOSTS_FILE=$1
JOB_FILE=$2

echo `date -u`: BEGIN
echo Number of clients: `wc -l ${HOSTS_FILE}`
cat ${HOSTS_FILE}
cat ${JOB_FILE}

fio --client=${HOSTS_FILE} ${JOB_FILE}

echo `date -u`: END
