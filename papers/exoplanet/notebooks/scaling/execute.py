#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import time
import subprocess as sp

os.environ["OMP_NUM_THREADS"] = "1"

cpus = os.cpu_count()
if cpus is None:
    cpus = 1
print(cpus)

if len(sys.argv) > 1:
    n_planets = int(sys.argv[1])
else:
    n_planets = 1

exe = [
    sys.executable,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaling.py"),
    "{0}".format(n_planets),
]
procs = []
for version in range(cpus):
    procs.append(sp.Popen(
        exe + ["{0}".format(version)],
        stdout=sp.DEVNULL, stderr=sp.PIPE))
    print(version, "started")

finished = [False for _ in procs]
try:
    while True:
        for i, proc in enumerate(procs):
            if finished[i]:
                continue
            code = proc.poll()
            if code is not None:
                out, err = proc.communicate()
                print(i, "finished")
                finished[i] = True
        if all(finished):
            break
        time.sleep(10)

finally:
    for proc in procs:
        proc.kill()
