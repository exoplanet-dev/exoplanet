#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


errors = []

for filename in ["_static/notebooks/pymc3-extras.ipynb"]:  # glob.glob("_static/notebooks/*.ipynb"):
    with open(filename) as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=-1)

    print("running: {0}".format(filename))
    try:
        ep.preprocess(notebook, {"metadata": {"path": "_static/notebooks/"}})
    except CellExecutionError as e:
        msg = "error while running: {0}\n\n".format(filename)
        msg += e.traceback
        print(msg)
        errors.append(msg)
    finally:
        with open(filename, mode="wt") as f:
            nbformat.write(notebook, f)


with open("notebook_errors.log", "wb") as f:
    f.write("\n\n".join(errors).encode("utf-8"))
