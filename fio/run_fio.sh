#!/usr/bin/env bash
set -ex
LOG_FILE=../data/fio.log
./run_fio1.sh $* |& tee -a ${LOG_FILE}
