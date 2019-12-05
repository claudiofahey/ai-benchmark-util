#!/usr/bin/env bash
set -ex
file=perf-no-strings-6.00x-48p-2048MiB-snappy.orc
seq 0 49 | xargs -i -P 0 cp -rv /mnt/isilon1/data/mortgage/${file}.0 /raid/mortgage/${file}.{}
