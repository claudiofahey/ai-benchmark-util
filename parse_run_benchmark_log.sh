#!/usr/bin/env bash
egrep -A1 "BEGIN|END|total images/sec:|args=" $* | less
