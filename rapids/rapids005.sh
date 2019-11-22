#!/usr/bin/env bash

docker run \
--runtime=nvidia \
-it \
-p 8881:8888 \
-v /mnt:/mnt \
--name rapidsai05 \
nvcr.io/nvidia/rapidsai/rapidsai:0.5-cuda10.0-runtime-ubuntu18.04-gcc7-py3.7
