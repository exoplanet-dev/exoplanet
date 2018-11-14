# Miniconda (cached)
export PATH="$HOME/miniconda/bin:$PATH"
if ! command -v conda > /dev/null; then
      wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
      bash miniconda.sh -b -p $HOME/miniconda -u;
      conda config --add channels conda-forge;
      conda config --set always_yes yes;
      conda update --all;
      conda create --yes -n test python=$PYTHON_VERSION
      conda activate test
      conda install tectonic;
      conda install -c conda-forge numpy=$NUMPY_VERSION scipy matplotlib setuptools pybind11 pytest pytest-cov pip healpy nbsphinx;
      pip install Pillow
      pip install batman-package
      pip install tqdm
      pip install ipython
      pip install jupyter
      pip install emcee
      pip install corner
      pip install wget
      pip install pandas
      pip install git+git://github.com/tomlouden/SPIDERMAN.git@69911b042bc46615ec9b39048a69e0d77c8542ad
      pip install sphinx
fi

# Display some info
conda info -a
