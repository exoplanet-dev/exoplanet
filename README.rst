exoplanet
=========

.. image:: https://img.shields.io/badge/GitHub-dfm%2Fexoplanet-blue.svg?style=flat
   :target: https://github.com/dfm/exoplanet
.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
   :target: https://github.com/dfm/exoplanet/blob/master/LICENSE
.. image:: https://img.shields.io/travis/dfm/exoplanet/master.svg?style=flat
   :target: https://travis-ci.org/dfm/exoplanet
.. image:: https://img.shields.io/readthedocs/exoplanet.svg?style=flat
   :target: https://exoplanet.dfm.io
.. image:: https://zenodo.org/badge/138077978.svg
   :target: https://zenodo.org/badge/latestdoi/138077978

.. image:: https://img.shields.io/badge/powered_by-starry-EB5368.svg?style=flat
   :target: https://rodluger.github.io/starry
.. image:: https://img.shields.io/badge/powered_by-celerite-EB5368.svg?style=flat
   :target: https://celerite.readthedocs.io
.. image:: https://img.shields.io/badge/powered_by-PyMC3-EB5368.svg?style=flat
   :target: https://docs.pymc.io
.. image:: https://img.shields.io/badge/powered_by-AstroPy-EB5368.svg?style=flat
   :target: http://www.astropy.org

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
