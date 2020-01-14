#!/bin/bash -x

set -e

. $CONDA/etc/profile.d/conda.sh
conda activate ./env

# Attempt to fix BLAS warnings
sudo apt-get update
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
export THEANO_FLAGS=blas.ldflags="-L/usr/lib/ -lblas"

# Execute the tutorials
env/bin/python .ci/azure/run_tutorials.py
