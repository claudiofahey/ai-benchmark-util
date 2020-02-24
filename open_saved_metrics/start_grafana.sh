#!/usr/bin/env bash

docker run \
-d \
--name grafana \
-p 3000:3000 \
grafana/grafana:6.3.5
