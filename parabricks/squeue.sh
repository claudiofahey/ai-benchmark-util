#!/usr/bin/env bash
squeue --format "%.18i %.9P %.27j %.8u %.2t %.10M %.6D %R" $*
