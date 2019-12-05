#!/usr/bin/env bash
set -ex
scriptdir=$(dirname "$(readlink -f "$0")")
cat ../hosts | xargs -i -P 0 ssh dgxuser@{} ${scriptdir}/copy_data_to_local2.sh
