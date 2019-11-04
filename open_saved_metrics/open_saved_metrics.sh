#!/usr/bin/env bash
set -e

: ${1?"Usage: $0 DATA_DIRECTORY"}

data_dir=$(readlink -f "$1")
echo Loading metrics from: ${data_dir}
export PROMETHEUS_DATA_DIR="/tmp/prometheus-data"
docker-compose -p open_saved_metrics -f $(dirname $0)/docker-compose.yml rm --stop --force prometheus grafana

#
# Prometheus
#

prometheus_tgz="${data_dir}/prometheus-data.tar.bz2"
ls -lh ${prometheus_tgz}
sudo rm -rf "${PROMETHEUS_DATA_DIR}"
mkdir "${PROMETHEUS_DATA_DIR}"
tar -xjvf "${prometheus_tgz}" --strip-components=1 -C "${PROMETHEUS_DATA_DIR}"
chmod -R a+rwX "${PROMETHEUS_DATA_DIR}"
docker-compose -p open_saved_metrics -f $(dirname $0)/docker-compose.yml up -d prometheus

#
# Grafana
#

docker-compose -p open_saved_metrics -f $(dirname $0)/docker-compose.yml up -d

echo
echo Prometheus URL: http://localhost:9094
echo Grafana URL: http://localhost:3000
echo
