#!/usr/bin/env bash
ansible-playbook --user root --inventory inventory.yaml monitoring_playbook.yaml -v $*
