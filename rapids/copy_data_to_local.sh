#!/usr/bin/env bash
set -ex
cat ../hosts | xargs -i -P 0 ssh dgxuser@{} rsync -rvv --exclude=*.tgz --delete --times /mnt/isilon1/data/mortgage /raid/
