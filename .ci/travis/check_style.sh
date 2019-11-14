#!/bin/bash -x

set -e

echo "Checking order of imports..."
isort -c -rc src

echo "Checking format of Python code..."
black --check src

echo "Checking format of Jupyter notebooks..."
black_nbconvert --check .
