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
      conda install -c conda-forge numpy=$NUMPY_VERSION scipy astropy setuptools pytest pytest-cov pip;
      pip install batman-package
fi

# Display some info
conda info -a
