#!/usr/bin/env bash
#
# This script assumes the Prometheus database was backed up using the following commands:
#   kubectl exec deployment/prometheus-server -c prometheus-server -- tar -cv /data > prometheus-data.tar
#   gzip prometheus-data.tar
#

set -e

: ${1?"Usage: $0 PROMETHEUS_DATA_TGZ"}

prometheus_tgz=$(readlink -f "$1")
echo Loading metrics from: ${prometheus_tgz}
export PROMETHEUS_DATA_DIR="/tmp/prometheus-data"
echo PROMETHEUS_DATA_DIR: ${PROMETHEUS_DATA_DIR}

ls -lh ${prometheus_tgz}
sudo rm -rf "${PROMETHEUS_DATA_DIR}"
mkdir "${PROMETHEUS_DATA_DIR}"
tar -xzvf "${prometheus_tgz}" --strip-components=1 -C "${PROMETHEUS_DATA_DIR}"
sudo chmod -R a+rwX "${PROMETHEUS_DATA_DIR}"

echo
echo Prometheus URL: http://localhost:9094
echo Grafana URL: http://localhost:3000
echo

docker run \
--rm \
--name prometheus \
-p 9094:9090 \
-v ${PROMETHEUS_DATA_DIR}:/prometheus \
prom/prometheus:v2.11.1 \
--config.file=/etc/prometheus/prometheus.yml \
--storage.tsdb.path=/prometheus \
--web.console.libraries=/usr/share/prometheus/console_libraries \
--web.console.templates=/usr/share/prometheus/consoles \
--storage.tsdb.retention.time 10000d
