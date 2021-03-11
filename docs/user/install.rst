.. _install:

Installation
============

.. note:: ``exoplanet`` requires Python 3.6 and later.

Using pip
---------

*exoplanet* can also be installed using `pip <https://pip.pypa.io>`_:

.. code-block:: bash

    python -m pip install -U exoplanet


Using conda
-----------

*exoplanet* can be installed using `conda <https://docs.conda.io>`_ from ``conda-forge``:

.. code-block:: bash

    conda install -c conda-forge exoplanet


.. _source:

From Source
-----------

The source code for *exoplanet* can be downloaded and installed `from GitHub
<https://github.com/exoplanet-dev/exoplanet>`_ by running

.. code-block:: bash

    git clone --recursive https://github.com/exoplanet-dev/exoplanet.git
    cd exoplanet
    python -m pip install -e .

Note: if you clone the repo without the ``--recursive`` flag, you will need to run

.. code-block:: bash

   git submodule init
   git submodule update


.. _windows:

Notes about running on Windows
------------------------------

Running exoplanet on Windows can be a little tricky because of the dependence on
runtime compilation, but it has been used on Windows successfully. The following
conda installation steps seem to work:

.. code-block:: bash

    conda create --name exoplanetTest python pip
    conda install --name exoplanetTest numpy scipy mkl-service libpython m2w64-toolchain astropy
    python -m pip install -e ".[test]"


Testing
-------

To run the unit tests, install the development dependencies using pip:

.. code-block:: bash

    python -m pip install ".[test]"

and then execute:

.. code-block:: bash

    python -m pytest -v tests

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/exoplanet-dev/exoplanet/issues>`_.
