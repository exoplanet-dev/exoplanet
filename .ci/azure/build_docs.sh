#!/bin/bash -x

set -e

. $CONDA/etc/profile.d/conda.sh
conda activate ./env

cd docs

make dirhtml
