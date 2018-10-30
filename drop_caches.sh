#!/bin/bash
set -ex
echo ${HOSTNAME}: Dropping caches
free -m -w
sync
echo 3 > /proc/sys/vm/drop_caches
free -m -w
