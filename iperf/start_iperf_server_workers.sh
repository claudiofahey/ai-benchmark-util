#!/usr/bin/env bash
set -x
cat hosts | xargs -i -P 0 ssh {} sudo apt-get -y install iperf
cat hosts | xargs -i -P 0 ssh {} pkill -9 iperf
cat hosts | xargs -i ssh {} "iperf --server --daemon > /dev/null 2>&1 &"
cat hosts | xargs -i ssh {} pgrep iperf
echo Done.
