#!/usr/bin/env bash
egrep "BEGIN|END|total images/sec:|args=" $* | less
