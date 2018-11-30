.. module:: exoplanet

.. _api:

API documentation
=================

Orbits
------

.. autoclass:: exoplanet.orbits.KeplerianOrbit
   :inherited-members:

.. autoclass:: exoplanet.orbits.TTVOrbit
   :inherited-members:

.. autoclass:: exoplanet.orbits.SimpleTransitOrbit
   :inherited-members:


Light curve models
------------------

.. autoclass:: exoplanet.StarryLightCurve
   :inherited-members:


Scalable Gaussian processes
---------------------------

.. autoclass:: exoplanet.gp.GP
   :inherited-members:

.. autoclass:: exoplanet.gp.terms.Term
   :inherited-members:

.. autoclass:: exoplanet.gp.terms.RealTerm
.. autoclass:: exoplanet.gp.terms.ComplexTerm
.. autoclass:: exoplanet.gp.terms.SHOTerm
.. autoclass:: exoplanet.gp.terms.Matern32Term
.. autoclass:: exoplanet.gp.terms.RotationTerm


Estimators
----------

.. autofunction:: exoplanet.estimate_semi_amplitude
.. autofunction:: exoplanet.estimate_minimum_mass
.. autofunction:: exoplanet.lomb_scargle_estimator
.. autofunction:: exoplanet.autocorr_estimator


Distributions
-------------

.. autoclass:: exoplanet.distributions.UnitVector
.. autoclass:: exoplanet.distributions.Angle
.. autoclass:: exoplanet.distributions.QuadLimbDark
.. autoclass:: exoplanet.distributions.RadiusImpact
.. autofunction:: exoplanet.distributions.get_joint_radius_impact


Utilities
---------

.. autofunction:: exoplanet.eval_in_model
.. autofunction:: exoplanet.get_samples_from_trace
.. autoclass:: exoplanet.PyMC3Sampler
   :inherited-members:


Citations
---------

.. autofunction:: exoplanet.citations.get_citations_for_model
