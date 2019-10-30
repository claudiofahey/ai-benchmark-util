#!/usr/bin/env bash
set -ex
cd /mnt/isilon/data/genomics
rm -rf parabricks-installer parabricks-installed
mkdir parabricks-installer

tar -xvzf parabricks.tar.gz -C parabricks-installer

./parabricks/installer.py \
--install-location ./parabricks-installed \
--container singularity \
--force

tar -czvf parabricks_install.tar.gz -C ./parabricks-installed parabricks
