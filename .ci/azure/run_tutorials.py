#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import re

import jupytext
import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

pattern = "docs/notebooks/*.py"
filenames = [fn for fn in glob.glob(pattern) if "notebook_setup" not in fn]
filenames = list(sorted(filenames, reverse=True))


def process_notebook(filename):
    outfile = os.path.splitext(filename)[0] + ".ipynb"
    outfile = os.path.join("docs/_static/notebooks", os.path.split(outfile)[1])

    # path = os.path.join(
    #     os.path.abspath("theano_cache"), "p{0}".format(os.getpid())
    # )
    # os.makedirs(path, exist_ok=True)
    # os.environ["THEANO_FLAGS"] = "base_compiledir={0}".format(path)

    errors = []

    notebook = jupytext.read(filename)
    # with open(filename) as f:
    #     notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=-1)

    print("running: {0} -> {1}".format(filename, outfile))
    try:
        ep.preprocess(notebook, {"metadata": {"path": "docs/notebooks/"}})
    except CellExecutionError as e:
        msg = "error while running: {0}\n\n".format(filename)
        msg += e.traceback
        print(msg)
        errors.append(msg)
    finally:
        with open(outfile, mode="wt") as f:
            nbformat.write(notebook, f)

    return "\n\n".join(errors)


errors = list(map(process_notebook, filenames))

errors = [e for e in errors if len(e.strip())]
txt = "\n\n".join(errors)
ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
txt = ansi_escape.sub("", txt)

with open("docs/notebooks/notebook_errors.log", "wb") as f:
    f.write(txt.encode("utf-8"))
