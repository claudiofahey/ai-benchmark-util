#!/usr/bin/env bash
set -ex
LOG_FILE=../data/fio.log
./run_fio_xargs1.sh $* |& tee -a ${LOG_FILE}
