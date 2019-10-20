#!/usr/bin/env bash
set -ex
LOGFILE="/imagenet-scratch/logs/run_benchmark_$(date "+%Y%m%dT%H%M%S").log"
python -u ./run_benchmark.py $* \
|& tee -a "${LOGFILE}"
echo run_benchmark.sh: END: Log file is ${LOGFILE}
