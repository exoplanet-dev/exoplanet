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
  <https://celerite.readthedocs.io>`_.
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

.. note:: Some tutorials have been moved to the `case studies <https://gallery.exoplanet.codes>`_ page.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/install
   tutorials/citation
   user/api
   user/dev

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/intro-to-pymc3
   tutorials/pymc3-extras
   tutorials/rv
   tutorials/transit
   tutorials/astrometric
   tutorials/gp


License & attribution
---------------------

Copyright 2018, 2019, 2020 Daniel Foreman-Mackey.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies.
You can find more information about how and what to cite in the
:ref:`citation` documentation.

These docs were made using `Sphinx <https://www.sphinx-doc.org>`_ and the
`Typlog theme <https://github.com/typlog/sphinx-typlog-theme>`_. They are
built and hosted on `Read the Docs <https://readthedocs.org>`_.


Changelog
---------

.. include:: ../HISTORY.rst
