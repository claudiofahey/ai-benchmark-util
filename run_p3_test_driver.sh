#!/usr/bin/env bash
./testgen.py | p3_test_driver -t - -c p3_test_driver.config.yaml
