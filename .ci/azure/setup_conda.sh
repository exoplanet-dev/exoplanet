#!/bin/bash -x

set -e

sudo chown -R $USER $CONDA
. $CONDA/etc/profile.d/conda.sh
conda env create -q --prefix ./env -f environment.yml
conda activate ./env
env/bin/python -m pip install -e .
