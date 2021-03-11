exoplanet
=========

*exoplanet* is a toolkit for probabilistic modeling of transit and/or radial
velocity observations of `exoplanets <https://en.wikipedia.org/wiki/Exoplanet>`_
and other astronomical time series using `PyMC3 <https://docs.pymc.io>`_.
*PyMC3* is a flexible and high-performance model building language and
inference engine that scales well to problems with a large number of
parameters. *exoplanet* extends *PyMC3*'s language to support many of the
custom functions and distributions required when fitting exoplanet datasets.
These features include:

* A fast and robust solver for Kepler's equation.
* Scalable Gaussian Processes using `celerite
  <https://celerite2.readthedocs.io>`_.
* Fast and accurate limb darkened light curves using `starry
  <https://rodluger.github.io/starry>`_.
* Common reparameterizations for `limb darkening parameters
  <https://arxiv.org/abs/1308.0009>`_, and `planet radius and impact
  parameter <https://arxiv.org/abs/1811.04859>`_.
* And many others!

All of these functions and distributions include methods for efficiently
calculating their *gradients* so that they can be used with gradient-based
inference methods like `Hamiltonian Monte Carlo <https://arxiv.org/abs/1206.1901>`_,
`No U-Turns Sampling <https://arxiv.org/abs/1111.4246>`_, and `variational
inference <https://arxiv.org/abs/1603.00788>`_. These methods tend to be more
robust than the methods more commonly used in astronomy (like `ensemble
samplers <https://emcee.readthedocs.io>`_ and `nested sampling
<https://ccpforge.cse.rl.ac.uk/gf/project/multinest/>`_) especially when the
model has more than a few parameters. For many exoplanet applications,
*exoplanet* (the code) can improve the typical performance by orders of
magnitude.

*exoplanet* is being actively developed in `a public repository on GitHub
<https://github.com/exoplanet-dev/exoplanet>`_ so if you have any trouble, `open an issue
<https://github.com/exoplanet-dev/exoplanet/issues>`_ there.

Where to find what you need
---------------------------

1. For general installation and basic usage, continue scrolling to the table of
contents below.

2. For more in depth examples of *exoplanet* used for more realistic problems,
go to the `Case studies page <https://gallery.exoplanet.codes>`_.

3. For more information about scalable Gaussian Processes in PyMC3 (this was
previously implemented as part of *exoplanet*), see the `celerite2 documentation
page <httsp://celerite2.readthedocs.io>`_.

4. For helper functions and PyMC3 extras that used to be implemented as part of
*exoplanet*, see the `pymc3-ext project
<https://github.com/exoplanet-dev/pymc3-ext>`_.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/install
   tutorials/citation.ipynb
   user/api
   user/dev

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/intro-to-pymc3.ipynb
   tutorials/rv.ipynb
   tutorials/transit.ipynb
   tutorials/astrometric.ipynb
   tutorials/light-delay.ipynb
   Case studies <https://gallery.exoplanet.codes>


License & attribution
---------------------

Copyright 2018, 2019, 2020, 2021 Daniel Foreman-Mackey.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies.
You can find more information about how and what to cite in the
`citation <tutorials/citation.ipynb>`_ documentation.

These docs were made using `Sphinx <https://www.sphinx-doc.org>`_ and the
`Typlog theme <https://github.com/typlog/sphinx-typlog-theme>`_. They are
built and hosted on `Read the Docs <https://readthedocs.org>`_.


Changelog
---------

.. include:: ../HISTORY.rst
