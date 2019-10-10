#!/usr/bin/env bash
# This script starts the TensorFlow Docker container on multiple hosts.
# You may add multiple -H parameters, one for each hostname.

./start_containers.py \
--nostart \
--user root \
-H dgx2-1 \
-H dgx2-2 \
-H dgx2-3
