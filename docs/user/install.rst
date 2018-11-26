.. _install:

Installation
============

*exoplanet* doesn't have a compiled components so it can be easily installed
from source or by using pip.

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

    pip install numpy pymc3 astropy


Using pip
---------

*exoplanet* can also be installed using `pip <https://pip.pypa.io>`_:

.. code-block:: bash

    pip install exoplanet


.. _source:

From Source
-----------

The source code for *exoplanet* can be downloaded and installed `from GitHub
<https://github.com/dfm/exoplanet>`_ by running

.. code-block:: bash

    git clone --recursive https://github.com/dfm/exoplanet.git
    cd exoplanet
    python setup.py install


Testing
-------

To run the unit tests, install the following dependencies using pip or conda
(you'll need to use the ``conda-forge`` channel to get starry):

.. code-block:: bash

    conda install -c conda-forge numpy scipy astropy pymc3 pytest starry pip
    pip install batman-package parameterized nose

and then execute:

.. code-block:: bash

    py.test -v

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/dfm/exoplanet/issues>`_.
