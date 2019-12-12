#!/usr/bin/env bash
docker run \
-d \
-p 8889:8888 \
-e JUPYTER_ENABLE_LAB=yes \
-v "$PWD":/home/jovyan/work \
-v /mnt:/home/jovyan/mnt \
--name jupyter-notebook \
jupyter/scipy-notebook:1386e2046833 \
jupyter-lab \
--ip=0.0.0.0 \
--no-browser \
--NotebookApp.token=""
