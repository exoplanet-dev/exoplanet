.. _install:

Installation
============

.. note:: ``exoplanet`` requires Python 3.6 and later.

Using pip
---------

The recommended method of installing *exoplanet* is with `pip
<https://pip.pypa.io>`_:

.. code-block:: bash

    python -m pip install -U "exoplanet[pymc]"

If you want to install some recommended extras (`pymc-ext
<https://github.com/exoplanet-dev/pymc-ext>`_ and `celerite2
<https://celerite2.readthedocs.io>`_), you can use the following, or just follow
the instructions on the relevant documentation pages:

.. code-block:: bash

    python -m pip install -U "exoplanet[extras]"

Both of these installation methods will install the required dependencies:

- `numpy <https://numpy.org>`_
- `astropy <https://www.astropy.org>`_
- `pymc <https://www.pymc.io>`_
- `exoplanet-core <https://github.com/exoplanet-dev/exoplanet-core>`_

Compatibility with PyMC3
------------------------

Although now updated to work with ``PyMC`` version 4, ``exoplanet`` is still
backwards compatible with ``PyMC3``. To install the ``PyMC3`` compatible version
of ``exoplanet``, use the following command:

.. code-block:: bash

    python -m pip install -U "exoplanet[pymc3]"

or first install ``PyMC3`` (however you want) and then ``exoplanet`` without any
extras:

.. code-block:: bash

    python -m pip install -U exoplanet

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
