.. _install:

Installation
============

.. note:: ``exoplanet`` requires Python 3.6 and later.

*exoplanet* doesn't have a compiled components so it can be easily installed from source or by using pip.

Dependencies
------------

The only required dependencies for *exoplanet* are `NumPy
<http://www.numpy.org/>`_, `PyMC3 <https://docs.pymc.io/>`_ and `AstroPy
<http://www.astropy.org/>`_.
These can be installed using conda or pip:

.. code-block:: bash

    conda install numpy pymc3 astropy

or

.. code-block:: bash

    python -m pip install -r requirements.txt


Using pip
---------

*exoplanet* can also be installed using `pip <https://pip.pypa.io>`_:

.. code-block:: bash

    python -m pip install -U exoplanet


.. _source:

From Source
-----------

The source code for *exoplanet* can be downloaded and installed `from GitHub
<https://github.com/exoplanet-dev/exoplanet>`_ by running

.. code-block:: bash

    git clone https://github.com/exoplanet-dev/exoplanet.git
    cd exoplanet
    python setup.py install


Testing
-------

To run the unit tests, install the development dependencies using pip:

.. code-block:: bash

    python -m pip install -r requirements-dev.txt

and then execute:

.. code-block:: bash

    python -m pytest -vs src/exoplanet

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/exoplanet-dev/exoplanet/issues>`_.
