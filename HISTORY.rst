0.2.0 (2019-08-02)
++++++++++++++++++

- Updates ``starry`` to get much better performance for high order spherical
  harmonics
- Renames ``StarryLightCurve`` to ``LimbDarkLightCurve``
- Adds support for fitting of astrometric observations
- Adds support for exposure time integration in ``celerite`` models
- Restructures the C++ backend to reduce code duplication
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
