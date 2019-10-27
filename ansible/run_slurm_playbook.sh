#!/usr/bin/env bash
ansible-playbook --user root --inventory inventory.yaml slurm_playbook.yaml -v $*
