#!/usr/bin/env bash
set -ex
IMAGE_TAG=${IMAGE_TAG:-claudiofahey/rapidsai:0.10-cuda10.0-runtime-ubuntu18.04-custom}
docker build -t ${IMAGE_TAG} .
docker push ${IMAGE_TAG}
