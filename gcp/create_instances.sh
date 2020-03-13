#!/usr/bin/env bash
set -ex

CREATE_COMMAND_1="\
gcloud beta compute \
--project=isilon-hdfs-project \
instances \
create"

CREATE_COMMAND_2="\
--machine-type=n1-standard-16 \
--subnet=hadoop-isilon-vpc \
--metadata=VmDnsSetting=GlobalOnly \
--maintenance-policy=TERMINATE \
--service-account=1043558379631-compute@developer.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/devstorage.read_only,\
https://www.googleapis.com/auth/logging.write,\
https://www.googleapis.com/auth/monitoring.write,\
https://www.googleapis.com/auth/servicecontrol,\
https://www.googleapis.com/auth/service.management.readonly,\
https://www.googleapis.com/auth/trace.append \
--image=c0-common-gce-gpu-image-20200128 \
--image-project=ml-images \
--boot-disk-size=50GB \
--boot-disk-type=pd-standard \
--labels=org=dl,type=worker \
--reservation-affinity=any"

#--accelerator=type=nvidia-tesla-p4,count=4

AVAIL_ZONE=b
FIRST_WORKER=019
LAST_WORKER=024

seq -w ${FIRST_WORKER} ${LAST_WORKER} | \
xargs -i -P 5 \
${CREATE_COMMAND_1} \
dl-worker-{} \
${CREATE_COMMAND_2} \
--min-cpu-platform=Intel\ Skylake \
--zone=us-east4-${AVAIL_ZONE} \
--boot-disk-device-name=dl-worker-{}


#AVAIL_ZONE=b
#FIRST_WORKER=024
#LAST_WORKER=028
#
#seq -w ${FIRST_WORKER} ${LAST_WORKER} | \
#xargs -i -P 4 \
#${CREATE_COMMAND_1} \
#dl-worker-{} \
#${CREATE_COMMAND_2} \
#--min-cpu-platform=Intel\ Skylake \
#--zone=us-east4-${AVAIL_ZONE} \
#--boot-disk-device-name=dl-worker-{}


#AVAIL_ZONE=c
#FIRST_WORKER=029
#LAST_WORKER=046
#
#seq -w ${FIRST_WORKER} ${LAST_WORKER} | \
#xargs -i -P 4 \
#${CREATE_COMMAND_1} \
#dl-worker-{} \
#${CREATE_COMMAND_2} \
#--min-cpu-platform=Intel\ Skylake \
#--zone=us-east4-${AVAIL_ZONE} \
#--boot-disk-device-name=dl-worker-{}
