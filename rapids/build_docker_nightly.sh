#!/usr/bin/env bash
set -ex
# Below digest is for cuda10.0-runtime-ubuntu18.04 dated 2019-12-03
FROM_SHA256=46ee5e319153ba1b29021aba56db9a47ab81f1b978ae7c03e73c402cbc9dcf4b
FROM_IMAGE=rapidsai/rapidsai-nightly@sha256:${FROM_SHA256}
TO_IMAGE=claudiofahey/rapidsai:${FROM_SHA256}
docker build -t ${TO_IMAGE} --build-arg FROM_IMAGE=${FROM_IMAGE} .
docker push ${TO_IMAGE}
