#!/usr/bin/env bash
docker stop tensorboard

docker run \
-d \
--rm \
-v /mnt/isilon/data/imagenet-scratch:/imagenet-scratch \
-p 6006:6006 \
--name tensorboard \
tensorflow/tensorflow:1.11.0 \
tensorboard \
--logdir=/imagenet-scratch/train_dir/
