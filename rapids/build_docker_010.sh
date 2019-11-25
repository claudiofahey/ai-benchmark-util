#!/usr/bin/env bash
set -ex
FROM_TAG=0.10-cuda10.0-runtime-ubuntu18.04
FROM_IMAGE=rapidsai/rapidsai:${FROM_TAG}
TO_IMAGE=claudiofahey/rapidsai:${FROM_TAG}-custom
docker build -t ${TO_IMAGE} --build-arg FROM_IMAGE=${FROM_IMAGE} .
docker push ${TO_IMAGE}
