#!/usr/bin/env bash
set -ex
client_opts="-t 60 --len 65536 --parallel 32"
ssh -J root@10.3.0.4 root@r0103i2-1 iperf -c 10.10.10.15 ${client_opts} &
ssh -J root@10.3.0.4 root@r0103i2-2 iperf -c 10.10.10.17 ${client_opts} &
ssh -J root@10.3.0.4 root@r0103i2-3 iperf -c 10.10.10.18 ${client_opts} &
ssh -J root@10.3.0.4 root@r0103i2-4 iperf -c 10.10.10.19 ${client_opts} &
ssh -J root@10.3.0.4 root@r0103i2-1 iperf -c 10.10.10.20 ${client_opts} &
ssh -J root@10.3.0.4 root@r0103i2-2 iperf -c 10.10.10.21 ${client_opts} &
ssh -J root@10.3.0.4 root@r0103i2-3 iperf -c 10.10.10.23 ${client_opts} &
ssh -J root@10.3.0.4 root@r0103i2-4 iperf -c 10.10.10.24 ${client_opts} &

wait
