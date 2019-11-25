#!/usr/bin/env bash
set -ex
# Below digest is for cuda10.0-runtime-ubuntu18.04 dated 2019-11-22
FROM_SHA256=a359097c3c18a534b91557d5abe772c73ef57d11de3dfb632e1516b0a01745f1
FROM_IMAGE=rapidsai/rapidsai-nightly@sha256:${FROM_SHA256}
TO_IMAGE=claudiofahey/rapidsai:${FROM_SHA256}
docker build -t ${TO_IMAGE} --build-arg FROM_IMAGE=${FROM_IMAGE} .
docker push ${TO_IMAGE}
