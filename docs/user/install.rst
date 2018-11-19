.. _install:

Installation
============

The core implementation of celerite is written in C++ so this will need to be
compiled to be called from Python. The easiest way for a new user to do this
will be by following the directions in the :ref:`using-conda` section below.
Power users might be able to eke out a bit more performance by tuning the
linear algebra library and installing :ref:`source`.

.. _using-conda:

Using conda
-----------

The easiest way to install celerite is using `conda
<http://continuum.io/downloads>`_ (via `conda-forge
<https://conda-forge.github.io/>`_) with the following command:

.. code-block:: bash

    conda install -c conda-forge celerite


Using pip
---------

celerite can also be installed using `pip <https://pip.pypa.io>`_ after
installing `Eigen <http://eigen.tuxfamily.org/>`_:

.. code-block:: bash

    pip install celerite


.. _source:

From Source
-----------

The source code for celerite can be downloaded `from GitHub
<https://github.com/dfm/celerite>`_ by running

.. code-block:: bash

    git clone https://github.com/dfm/celerite.git


.. _python-deps:

Dependencies
++++++++++++

For the Python interface, you'll (obviously) need a Python installation and I
recommend `conda <http://continuum.io/downloads>`_ if you don't already have
your own opinions.

After installing Python, the following dependencies are required to build
celerite:

1. `NumPy <http://www.numpy.org/>`_ for math and linear algebra in Python, and
2. `pybind11 <https://pybind11.readthedocs.io>`_ for the Python–C++ interface.
3. `autograd <https://github.com/HIPS/autograd>`_ (optional) for computing
   gradients in celerite models.

If you're using conda, you can install all of the dependencies with the
following command:

.. code-block:: bash

    conda install -c conda-forge numpy pybind11 autograd

Building
++++++++

After installing the dependencies, you can build the celerite module by
running:

.. code-block:: bash

    python setup.py install

in the root directory of the source tree.

Automatic differentiation
+++++++++++++++++++++++++

Gradients of celerite models are computed using automatic differentiation. By
default, this is implemented using the ``AutoDiffScalar`` in Eigen, but it is
possible to get faster performance by using the `stan-math library
<https://github.com/stan-dev/math>`_. To do this, download the library from
`the GitHub page <https://github.com/stan-dev/math>`_ and build celerite using
the following command:

.. code-block:: bash

    python setup.py build_ext -I${STAN}:${STAN}/lib/boost_1.62.0:${STAN}/lib/cvodes_2.9.0/include -DUSE_STAN_MATH install

where ``${STAN}`` is the directory where you downloaded the stan-math library.


Testing
-------

To run the unit tests, install `pytest <http://doc.pytest.org/>`_ and then
execute:

.. code-block:: bash

    py.test -v

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/dfm/celerite/issues>`_.
