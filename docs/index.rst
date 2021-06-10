exoplanet
=========

*exoplanet* is a toolkit for probabilistic modeling of time series data in
astronomy with a focus on observations of `exoplanets
<https://en.wikipedia.org/wiki/Exoplanet>`_, using `PyMC3
<https://docs.pymc.io>`_. *PyMC3* is a flexible and high-performance model
building language and inference engine that scales well to problems with a large
number of parameters. *exoplanet* extends *PyMC3*'s language to support many of
the custom functions and distributions required when fitting exoplanet datasets.
These features include:

* A fast and robust solver for Kepler's equation.
* Scalable Gaussian Processes using `celerite <https://celerite2.readthedocs.io>`_.
* Fast and accurate limb darkened light curves using `starry
  <https://rodluger.github.io/starry>`_.
* Common reparameterizations for exoplanet-specific parameters like `limb
  darkening <https://arxiv.org/abs/1308.0009>`_ and eccentricity.
* And many others!

All of these functions and distributions include methods for efficiently
calculating their *gradients* so that they can be used with gradient-based
inference methods like `Hamiltonian Monte Carlo
<https://arxiv.org/abs/1206.1901>`_, `No U-Turns Sampling
<https://arxiv.org/abs/1111.4246>`_, and `variational inference
<https://arxiv.org/abs/1603.00788>`_. These methods tend to be more robust than
the methods more commonly used in astronomy (like `ensemble samplers
<https://emcee.readthedocs.io>`_ and `nested sampling
<https://ccpforge.cse.rl.ac.uk/gf/project/multinest/>`_) especially when the
model has more than a few parameters. For many exoplanet applications,
*exoplanet* (the code) can improve the typical performance by orders of
magnitude.

*exoplanet* is being actively developed in `a public repository on GitHub
<https://github.com/exoplanet-dev/exoplanet>`_ so if you have any trouble, `open
an issue <https://github.com/exoplanet-dev/exoplanet/issues>`_ there.

.. admonition:: Where to find what you need
   :class: hint

   🖥 For general installation and basic usage, continue scrolling to the table of
   contents below.

   🖼 For more in depth examples of *exoplanet* used for more realistic problems,
   go to the `Case studies page <https://gallery.exoplanet.codes>`_.

   📈 For more information about scalable Gaussian Processes in PyMC3 (this was
   previously implemented as part of *exoplanet*), see the `celerite2 documentation
   page <https://celerite2.readthedocs.io>`_.

   👉 For helper functions and PyMC3 extras that used to be implemented as part of
   *exoplanet*, see the `pymc3-ext project
   <https://github.com/exoplanet-dev/pymc3-ext>`_.


Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user/install
   tutorials/citation.md
   user/theano
   user/multiprocessing
   user/api
   user/dev
   changes.rst

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/about.md
   tutorials/autodiff.md
   tutorials/intro-to-pymc3.md
   tutorials/data-and-models.md
   tutorials/light-delay.md
   tutorials/reparameterization.md
   Case studies <https://gallery.exoplanet.codes>
   celerite2 <https://celerite2.readthedocs.io>


License & attribution
---------------------

Copyright 2018, 2019, 2020, 2021 Daniel Foreman-Mackey.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies. You
can find more information about how and what to cite in the `citation
<tutorials/citation.ipynb>`_ documentation.
