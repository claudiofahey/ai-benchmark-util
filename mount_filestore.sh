#!/usr/bin/env bash
cat hosts | xargs -i -P 0 ssh {} sudo mount -t nfs 10.108.10.90:/share1 /mnt/filestore1
