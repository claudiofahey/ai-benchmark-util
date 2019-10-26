#!/bin/bash

time mpirun \
-np 8 \
-H dgx2-2:12 \
-H dgx2-3:12 \
--map-by socket:PE=12 \
--bind-to core \
--report-bindings \
--tag-output \
--timestamp-output \
parabricks_germline_pipeline_mpi.py
