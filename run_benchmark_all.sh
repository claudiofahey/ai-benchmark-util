#!/usr/bin/env bash
set -ex
LOGFILE="/imagenet-scratch/logs/run_benchmark_all_$(date "+%Y%m%dT%H%M%S").log"
echo -n run_benchmark_*.yaml run_benchmark_*.yaml run_benchmark_*.yaml | \
xargs -d " " -i -P 1 python -u run_benchmark.py -c {} \
|& tee -a "${LOGFILE}"
echo run_benchmark_all.sh: END: Log file is ${LOGFILE}
