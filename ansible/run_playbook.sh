#!/usr/bin/env bash
ansible-playbook --user root --inventory inventory.yaml playbook.yaml -v $*
