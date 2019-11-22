#!/usr/bin/env bash

docker run \
--runtime=nvidia \
-it \
-p 8883:8888 \
-v /mnt:/mnt \
--name rapidsai06 \
nvcr.io/nvidia/rapidsai/rapidsai:0.6-cuda10.0-runtime-ubuntu18.04-gcc7-py3.7
