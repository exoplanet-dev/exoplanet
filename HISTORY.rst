0.4.5 (2021-03-15)
++++++++++++++++++

- Adds support for recent versions of PyMC3 built on Aesara `#150 <https://github.com/exoplanet-dev/exoplanet/pull/150>`_
- Fixes bug in `SimpleTransitOrbit` contact point calculation `#148 <https://github.com/exoplanet-dev/exoplanet/pull/148>`_


0.4.4 (2021-01-04)
++++++++++++++++++

- Adds support for relativistic orbits using REBOUNDx `#98 <https://github.com/exoplanet-dev/exoplanet/pull/98>`_


0.4.3 (2020-12-08)
++++++++++++++++++

- Adds support for PyMC3 v3.10 and Theano-pymc


0.4.2 (2020-11-15)
++++++++++++++++++

- Fixes issue with automatic execution of tutorials


0.4.1 (2020-11-15)
++++++++++++++++++

- Fixes pickling error when sampling in parallel `#120 <https://github.com/exoplanet-dev/exoplanet/pull/120>`_


0.4.0 (2020-10-05)
++++++++++++++++++

- Adds faster solver for Kepler's equation
- Improves automation of documentation and release process
- Removes most C++ Ops and replaces them with pre-compiled functions which allows the distribution of binary wheels
- Removes ``gp`` submodule; moved to new `celerite2 <https://celerite2.readthedocs.io>`_ package
- Deprecates some non-exoplanet-specific functions and distributions; moved to `pymc3-ext <https://github.com/exoplanet-dev/pymc3-ext>`_ package


0.3.3 (2020-09-09)
++++++++++++++++++

- Fixes compatibility with PyMC3 version 3.9


0.3.2 (2020-05-04)
++++++++++++++++++

- Fixes a documentation bug introduced in v0.3.1


0.3.1 (2020-05-04)
++++++++++++++++++

- Adds support for light travel time when computing positions in Keplerian orbits
- Adds ``SecondaryEclipseLightCurve`` for modeling eclipsing binaries
- Adds ``UnitDisk`` distribution for fitting eccentricity vectors


0.3.0 (2020-04-03)
++++++++++++++++++

- Adds tests and support for Windows
- Adds a "Jacobian" interface to the orbit objects for reparameterization


0.2.6 (2020-03-23)
++++++++++++++++++

- Adds support for fitting circular orbits with duration
- Adds a ``bls_estimator`` for transit search using Astropy's ``BoxLeastSquares``


0.2.5 (2020-03-11)
++++++++++++++++++

- Improves infrastructure for generating documentation
- Adds an ``EclipsingBinaryLightCurve`` for building binary star models
- Adds ``DensityDist`` implementation for celerite GP likelihoods


0.2.4 (2019-12-30)
++++++++++++++++++

- Makes ``rebound`` an optional dependency


0.2.3 (2019-11-12)
++++++++++++++++++

- Adds ``ConditionalMeanOp`` and ``DotLOp`` for scalable conditional mean calculation
  and prior sampling with celerite
- Adds developer documentation
- Moves documentation to a separate repository


0.2.2 (2019-10-25)
++++++++++++++++++

- Adds ``TTVOrbit`` tutorial
- Switches tutorials to `lightkurve <https://docs.lightkurve.org>`_ for data access
- Improves packaging and code style features
- Fixes bugs and improves interface to ``TTVOrbit``


0.2.1 (2019-09-26)
++++++++++++++++++

- Adds a new interface for tuning dense mass matrices with less overhead
- Adds support for photodynamics using `rebound <https://rebound.rtfd.io>`_
- Adds a new interface for assigning units to Theano variables
- Adds new physically-motivated distributions for impact parameter and
  eccentricity
- Improves test coverage
- Fixes bug in diagonal elements of the ``IntegratedTerm`` model
- Fixes bug in indexing for ``TTVOrbit`` models with large TTVs


0.2.0 (2019-08-04)
++++++++++++++++++

- Updates ``starry`` to get much better performance for high order spherical
  harmonics
- Renames ``StarryLightCurve`` to ``LimbDarkLightCurve``
- Restructures the C++ backend to reduce code duplication
- Adds support for fitting of astrometric observations
- Adds support for exposure time integration in ``celerite`` models
- Adds new distributions for periodic parameters and U(0, 1).
- Fixes many small bugs


0.1.6 (2019-04-24)
++++++++++++++++++

- Fixes some edge case failures for Kepler solver
- Improves reliability of contact point solver and fails (more) gracefully
  when this doesn't work; this reduces the number of divergences when fitting
  a transit model


0.1.5 (2019-03-07)
++++++++++++++++++

- Improves contact point solver using companion matrix to solve quadratic
- Improves reliability of ``Angle`` distribution when the value of the angle
  is well constrained


0.1.4 (2019-02-10)
++++++++++++++++++

- Improves the reliability of the PyMC3Sampler
- Adds a new ``optimize`` function since the ``find_MAP`` method
  in PyMC3 is deprecated
- Adds cronjob script for automatically updating tutorials.


0.1.3 (2019-01-09)
++++++++++++++++++

- Adds a more robust and faster Kepler solver (`ref
  <http://adsabs.harvard.edu/abs/1991CeMDA..51..319N>`_)
- Fixes minor behavioral bugs in PyMC3 sampler wrapper


0.1.2 (2018-12-13)
++++++++++++++++++

- Adds regular grid interpolation Op for Theano
- Fixes major bug in handling of the stellar radius for transits
- Fixes small bugs in packaging and installation
- Fixes handling of diagonal covariances in ``PyMC3Sampler``


0.1.1 (IPO; 2018-12-06)
+++++++++++++++++++++++

- Initial public release
