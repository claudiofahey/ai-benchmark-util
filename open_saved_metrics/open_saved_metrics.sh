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
docker-compose -p open_saved_metrics -f $(dirname $0)/docker-compose.yml rm --stop --force prometheus grafana

#
# Prometheus
#

ls -lh ${prometheus_tgz}
sudo rm -rf "${PROMETHEUS_DATA_DIR}"
mkdir "${PROMETHEUS_DATA_DIR}"
tar -xzvf "${prometheus_tgz}" --strip-components=1 -C "${PROMETHEUS_DATA_DIR}"
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
