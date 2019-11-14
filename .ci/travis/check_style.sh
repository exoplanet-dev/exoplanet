#!/bin/bash -x

set -e

python -m pip install -U pip
python -m pip install -U black isort black_nbconvert

echo "Checking order of imports..."
isort -c -rc src

echo "Checking format of Python code..."
black --check src

echo "Checking format of Jupyter notebooks..."
black_nbconvert --check .
