#!/usr/bin/env bash
set -x

AVAIL_ZONE=a
FIRST_WORKER=003
LAST_WORKER=003

seq -w ${FIRST_WORKER} ${LAST_WORKER} | \
xargs -i -P 0 \
gcloud -q beta compute \
--project=isilon-hdfs-project \
instances \
delete \
dl-worker-{} \
--zone=us-east4-${AVAIL_ZONE}

AVAIL_ZONE=b
FIRST_WORKER=004
LAST_WORKER=004

seq -w ${FIRST_WORKER} ${LAST_WORKER} | \
xargs -i -P 0 \
gcloud -q beta compute \
--project=isilon-hdfs-project \
instances \
delete \
dl-worker-{} \
--zone=us-east4-${AVAIL_ZONE}
