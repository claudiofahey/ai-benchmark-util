#!/usr/bin/env bash
docker run -d -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes \
-v "$PWD":/home/jovyan/work \
-v /mnt:/home/jovyan/mnt \
--name jupyter jupyter/scipy-notebook:1386e2046833
