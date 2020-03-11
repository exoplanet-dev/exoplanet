.. _dev:

Developer documentation
=======================

Development of ``exoplanet`` is actively happening `on GitHub <https://github.com/exoplanet-dev/exoplanet>`_ and we would love your contributions.
There are a few different methods of contributing to ``exoplanet`` and the details are discussed below.

Reporting an issue
------------------

If you run into issues, bugs, or anything else, it is very useful if you can `post an issue on the GitHub repository <https://github.com/exoplanet-dev/exoplanet/issues>`_.
When you post an issue, please provide the details to reproduce the issue.
For example, if you find a bug, please provide a *standalone* and *executable* snippet of code that demonstrates the issue.
It's also useful to include details about your platform and the versions of key packages that your using.


Contributing code or documentation
----------------------------------

If you're not familiar with the workflow for contributing code to a GitHub repository, an excellent place to start is the `AstroPy developer docs <https://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_.


Set up your development environment
+++++++++++++++++++++++++++++++++++

After getting some familiarity with the workflow, you should fork `the exoplanet repository <https://github.com/exoplanet-dev/exoplanet>`_ and clone it to your local machine:

.. code-block:: bash

    git clone https://github.com/YOURUSERNAME/exoplanet.git
    cd exoplanet
    git checkout -b BRANCHNAME

for some name ``BRANCHNAME`` describing your contribution.

Then you should set up an isolated environment for development using a `Conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_, `virtualenv <https://virtualenv.pypa.io/>`_, `venv <https://docs.python.org/3/library/venv.html>`_, or similar.
If using ``conda``, you can get the current development environment from the ``environment.yml`` file:

.. code-block:: bash

    conda env create --prefix env -f environment.yml
    conda activate ./env

If you have an existing ``conda`` environment for ``exoplanet``, you can update it using:

.. code-block:: bash

    conda env update --prefix ./env -f environment.yml  --prune

If you're using a ``pip`` based environment, you can install the developer dependencies as follows:

.. code-block:: bash

    python -m pip install -U pip
    python -m pip install -U -r requirements-dev.txt

After your environment is set up, you can install the current development version of ``exoplanet``:

.. code-block:: bash

    python -m pip install -U setuptools setuptools_scm pep517
    python -m pip uninstall -y exoplanet
    python setup.py develop


Finding your way around the codebase
++++++++++++++++++++++++++++++++++++

``exoplanet`` is mostly arranged as a typical Python project with the module root in the ``src/exoplanet`` directory.
But there are a few directions that can be useful before diving in:

1. **Tutorials**: The tutorials are written using the `jupytext <https://github.com/mwouts/jupytext>`_ tool to keep the size of the repository from blowing up.
The tutorial files are saved (using jupytext) in the ``docs/notebooks`` directory as ``.py`` files in the ``py:percent`` format.
You can edit these files as if they are Jupyter notebooks by using the extension:

.. code-block:: bash

    jupyter nbextension install --py jupytext
    jupyter nbextension enable --py jupytext

If you are contributing a new tutorial, you should copy one of the existing ones and try to follow roughly the same format.

2. **Case studies**: These are more in depth tutorials that require more computational run time.
These can be found in the `case studies <https://github.com/exoplanet-dev/case-studies>`_ repo and there is more information there about how to contribute.

3. **Theano ops**: ``exoplanet`` comes bundled with a set of custom Theano ops that are implemented in ``src/exoplanet/theano_ops``.
As a user, you'll rarely interact with these directly and we haven't put a lot of work into making them user friendly, but if you are interested in diving in, here are some tips.
First, you should check out the Theano docs that describe how to develop new ops in `Python <http://deeplearning.net/software/theano/extending/extending_theano.html>`_ and `C/C++ <http://deeplearning.net/software/theano/extending/extending_theano_c.html>`_.
Most of the ``exoplanet`` ops are implemented in C++ for speed and we've made the design decision to separate the "science" code (which implements the actual operation without any dependency on Theano) and "wrapper" code (which sets up the interface).
The science code is implemented as a header-only C++ library in ``src/exoplanet/theano_ops/lib/include/exoplanet`` and then the wrappers are implemented as submodules in ``src/exoplanet/theano_ops``.
A good place to start is the ``KeplerOp`` implemented in ``src/exoplanet/theano_ops/kepler`` and ``src/exoplanet/theano_ops/lib/include/exoplanet/kepler.h``.


Testing your contribution
+++++++++++++++++++++++++

If you're contributing a change to the code (either a new feature or bug fix), make sure that you implement at least one test that checks the behavior of your code.
Then, you should run all of the unit tests before submitting a pull request using the following command:

.. code-block:: bash

    python -m pytest -sv src/exoplanet


Code style
----------

We have a pretty strict (but easy to implement!) set of style guidelines for the codebase.
For Python code, we use `isort <https://github.com/timothycrosley/isort>`_, `black <https://github.com/psf/black>`_, and `black_nbconvert <https://github.com/dfm/black_nbconvert>`_ (for Jupyter notebooks).
The custom settings for these projects can be found in ``pyproject.toml``.
Before opening a pull request, you can run the formatters as follows:

.. code-block:: bash

    isort -rc src
    black src
    black_nbconvert .

Or, you can use `pre-commit <https://pre-commit.com>`_ to automatically apply the formatting whenever you commit:

.. code-block:: bash

    python -m pip install -U pre-commit
    pre-commit install


Release management
------------------

.. note:: Most of this build process is based on the October 2019 update to `this blog post <https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/>`_ so you should check that out if you want more info.

This section is mainly internal, but these are the steps that should be executed to produce a new release.

1. Update citation date and version in ``src/exoplanet/citations.py``.
2. Update the notebooks in the `case studies <https://github.com/exoplanet-dev/case-studies>`_ repo and run a release there.
3. Update changelog date in ``HISTORY.rst``.
4. Tag a GitHub release.

After executing these steps, the release can be built as follows:

.. code-block:: bash

    python -m pip install -U pip pep517 twine setuptools_scm
    git pull
    rm -rf build dist
    python -m pep517.build .

Then you can test that the build installs as expected:

.. code-block:: bash

    rm -rf venv-test
    python -m venv venv-test
    venv-test/bin/python -m pip install dist/exoplanet*.tar.gz
    venv-test/bin/python -c "import exoplanet;print(exoplanet.__version__)"

This should print the current version number.
Similarly you can test the wheel:

.. code-block:: bash

    rm -rf venv-test
    python -m venv venv-test
    venv-test/bin/python -m pip install dist/exoplanet*.whl
    venv-test/bin/python -c "import exoplanet;print(exoplanet.__version__)"

Once you're satisfied with this build, you can upload it using ``twine``:

.. code-block:: bash

    twine upload dist/*
