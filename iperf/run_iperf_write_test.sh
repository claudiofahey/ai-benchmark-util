#!/usr/bin/env bash
set -ex
client_opts="-t 60 --len 65536 --parallel 16"
ssh dl-worker1 iperf -c 10.3.0.5 ${client_opts} &
ssh dl-worker2 iperf -c 10.3.0.6 ${client_opts} &
ssh dl-worker3 iperf -c 10.3.0.7 ${client_opts} &
ssh dl-worker4 iperf -c 10.3.0.8 ${client_opts} &
ssh dl-worker5 iperf -c 10.3.0.5 ${client_opts} &
ssh dl-worker6 iperf -c 10.3.0.6 ${client_opts} &
ssh  dl-worker9.us-east4-b.c.isilon-hdfs-project.internal iperf -c 10.3.0.7 ${client_opts} &
ssh dl-worker10.us-east4-b.c.isilon-hdfs-project.internal iperf -c 10.3.0.8 ${client_opts} &

wait
