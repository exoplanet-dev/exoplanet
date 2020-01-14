#!/bin/bash -x

set -e

. $CONDA/etc/profile.d/conda.sh
conda activate ./env

env/bin/python .ci/azure/run_tutorials.py
