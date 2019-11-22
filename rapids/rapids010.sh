#!/usr/bin/env bash

docker run \
-d \
--runtime=nvidia \
-it \
-p 8884:8888 \
-v /mnt:/mnt \
-v /mnt:/rapids/notebooks/mnt \
--name rapidsai010 \
nvcr.io/nvidia/rapidsai/rapidsai:0.10-cuda10.0-runtime-ubuntu18.04
