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
<https://github.com/dfm/exoplanet>`_ so if you have any trouble, `open an issue
<https://github.com/dfm/exoplanet/issues>`_ there.

.. raw:: html

    <p>
    <a href="https://travis-ci.org/dfm/exoplanet"><img src="https://img.shields.io/travis/dfm/exoplanet/master.svg?style=flat"/></a>
    <a href="https://travis-ci.org/dfm/exoplanet"><img src="https://img.shields.io/readthedocs/exoplanet.svg?style=flat"/></a>
    <br>
    <a href="https://rodluger.github.io/starry"><img src="https://img.shields.io/badge/powered_by-starry-EB5368.svg?style=flat"/></a>
    <a href="https://celerite.readthedocs.io"><img src="https://img.shields.io/badge/powered_by-celerite-EB5368.svg?style=flat"/></a>
    <a href="https://docs.pymc.io"><img src="https://img.shields.io/badge/powered_by-PyMC3-EB5368.svg?style=flat"/></a>
    <a href="http://www.astropy.org"><img src="https://img.shields.io/badge/powered_by-AstroPy-EB5368.svg?style=flat"/></a>
    </p>


User guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   user/install
   user/api


Tutorials
---------

.. toctree::
   :maxdepth: 2

   tutorials/intro-to-pymc3
   tutorials/pymc3-extras
   tutorials/gp


License & attribution
---------------------

Copyright 2018, Daniel Foreman-Mackey.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite the following paper:

.. code-block:: tex

    tbd


Changelog
---------

.. include:: ../HISTORY.rst
