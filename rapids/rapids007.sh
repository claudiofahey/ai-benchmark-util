#!/usr/bin/env bash

docker run \
--runtime=nvidia \
-it \
-p 8882:8888 \
-v /mnt:/mnt \
--name rapidsai07 \
nvcr.io/nvidia/rapidsai/rapidsai:0.7-cuda10.0-runtime-ubuntu18.04-gcc7-py3.7
