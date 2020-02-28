#!/usr/bin/env bash
set -x
cat hosts-isilon | xargs -i ssh -J root@10.3.0.4 root@{} pkill -9 iperf
cat hosts-isilon | xargs -i ssh -J root@10.3.0.4 root@{} "iperf --server --daemon > /dev/null 2>&1 &"
cat hosts-isilon | xargs -i ssh -J root@10.3.0.4 root@{} pgrep iperf
echo Done.
