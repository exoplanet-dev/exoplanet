#!/bin/bash -x

set -e

sudo chown -R $USER $CONDA
. $CONDA/etc/profile.d/conda.sh

if [[ -d env ]]; then
    conda env update -q --prefix ./env -f environment.yml --prune
else
    conda env create -q --prefix ./env -f environment.yml
fi

conda activate ./env
env/bin/python -m pip install -e .
