# Miniconda (cached)
if ! command -v conda > /dev/null; then
    if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
    else
        export CXX=g++-4.8
        export CC=gcc-4.8
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    fi
    bash miniconda.sh -b -p $HOME/miniconda -u;
    . $HOME/miniconda/etc/profile.d/conda.sh
    conda activate
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda create -q --yes -n test python=$PYTHON_VERSION
    conda activate test
    conda install -q -c conda-forge pip numpy=$NUMPY_VERSION  scipy astropy setuptools Theano pymc3 pytest
    pip install parameterized nose coveralls pytest-cov>=2.6.1 pytest-env pybind11
    pip install celerite starry

    pip uninstall -y batman-package
    git clone https://github.com/lkreidberg/batman.git
    cd batman
    git apply $TRAVIS_BUILD_DIR/.ci/batman-patch.diff
    python setup.py install
    cd ..


fi

# Display some info
conda info -a
