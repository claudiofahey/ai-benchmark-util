#!/usr/bin/env bash
cat hosts | xargs -i -P 0 scp -r etc root@{}:/

cat hosts | xargs -i -P 0 ssh root@{} "docker stop metricbeat ; docker rm metricbeat"

cat hosts | xargs -i -P 0 ssh root@{} nvidia-docker run \
  -d \
  --restart=always \
  -v /etc/metricbeat/metricbeat.yml:/usr/share/metricbeat/metricbeat.yml \
  -v /etc/metricbeat/modules.d:/usr/share/metricbeat/modules.d \
  -v /proc:/hostfs/proc:ro \
  -v /sys/fs/cgroup:/hostfs/sys/fs/cgroup:ro \
  -v /:/hostfs:ro \
  --network host \
  --name metricbeat \
  docker.elastic.co/beats/metricbeat:6.4.1 \
  -system.hostfs=/hostfs \
  -e

head -1 hosts | xargs -i -P 0 ssh root@{} docker exec metricbeat metricbeat setup -v
