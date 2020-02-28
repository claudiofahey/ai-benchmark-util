#!/usr/bin/env bash
cat hosts | xargs -i -P 0 ssh {} pkill fio
cat hosts | xargs -i -P 0 ssh {} fio --server --daemonize=/tmp/fio.pid
