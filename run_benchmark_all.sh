#!/usr/bin/env bash
set -ex
LOGFILE="/imagenet-scratch/logs/run_benchmark_all_$(date "+%Y%m%dT%H%M%S").log"
for config in "$@"
do
    python -u run_benchmark.py -c "${config}" |& tee -a "${LOGFILE}"
done
echo run_benchmark_all.sh: END: Log file is ${LOGFILE}
