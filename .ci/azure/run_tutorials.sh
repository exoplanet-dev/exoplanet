#!/bin/bash -x

set -e

. $CONDA/etc/profile.d/conda.sh
conda activate ./env

jupytext --to ipynb --execute docs/tutorials/*.py
ls docs/tutorials
