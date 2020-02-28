#!/usr/bin/env bash
set -ex
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,\
pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,\
memory.total,memory.free,memory.used --format=csv -l 5 | tee -a data/gpu_metrics_${HOSTNAME}.csv
