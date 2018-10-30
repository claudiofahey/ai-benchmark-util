#!/usr/bin/env bash
cat hosts | xargs -i -P 0 scp -r etc root@{}:/

cat hosts | xargs -i -P 0 ssh root@{} "docker stop nvidiagpubeat ; docker rm nvidiagpubeat"

cat hosts | xargs -i -P 0 ssh root@{} nvidia-docker run \
  --restart=always \
  -d \
  -v /etc/nvidiagpubeat/nvidiagpubeat.yml:/root/nvidiagpubeat.yml \
  --network host \
  --name nvidiagpubeat \
  claudiofahey/nvidiagpubeat:20180925 \
  /root/nvidiagpubeat \
  -e

#head -1 hosts | xargs -i -P 0 ssh root@{} docker exec nvidiagpubeat /root/nvidiagpubeat -setup -e
