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
