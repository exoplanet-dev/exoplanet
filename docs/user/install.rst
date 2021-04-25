.. _install:

Installation
============

.. note:: ``exoplanet`` requires Python 3.6 and later.

Using pip
---------

The recommended method of installing *exoplanet* is with `pip
<https://pip.pypa.io>`_:

.. code-block:: bash

    python -m pip install -U exoplanet

If you want to install some recommended extras (`pymc3-ext
<https://github.com/exoplanet-dev/pymc3-ext>`_ and `celerite2
<https://celerite2.readthedocs.io>`_), you can use the following, or just follow
the instructions on the relevant documentation pages:

.. code-block:: bash

    python -m pip install -U "exoplanet[extras]"

Both of these installation methods will install the required dependencies:

- `numpy <https://numpy.org>`_
- `astropy <https://www.astropy.org>`_
- `pymc3 <https://docs.pymc.io>`_
- `exoplanet-core <https://github.com/exoplanet-dev/exoplanet-core>`_
- `aesara-theano-fallback <https://github.com/exoplanet-dev/aesara-theano-fallback>`_


Using conda
-----------

*exoplanet* can also be installed using `conda <https://docs.conda.io>`_ from
``conda-forge``:

.. code-block:: bash

    conda install -c conda-forge exoplanet


.. _source:

From Source
-----------

The source code for *exoplanet* can be downloaded and installed `from GitHub
<https://github.com/exoplanet-dev/exoplanet>`_ by running

.. code-block:: bash

    git clone https://github.com/exoplanet-dev/exoplanet.git
    cd exoplanet
    python -m pip install -e .


Testing
-------

To run the unit tests, install the development dependencies using pip:

.. code-block:: bash

    python -m pip install -e ".[test]"

and then execute:

.. code-block:: bash

    python -m pytest -v tests

All of the tests should (of course) pass. If any of the tests don't pass and if
you can't sort out why, `open an issue on GitHub
<https://github.com/exoplanet-dev/exoplanet/issues>`_.
