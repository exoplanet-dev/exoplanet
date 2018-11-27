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

.. autoclass:: exoplanet.light_curve.StarryLightCurve
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

.. autofunction:: exoplanet.estimators.estimate_minimum_mass
.. autofunction:: exoplanet.estimators.lomb_scargle_estimator
.. autofunction:: exoplanet.estimators.autocorr_estimator


Distributions
-------------

.. autoclass:: exoplanet.distributions.UnitVector
.. autoclass:: exoplanet.distributions.Angle
.. autoclass:: exoplanet.distributions.QuadLimbDark
.. autoclass:: exoplanet.distributions.RadiusImpact
.. autofunction:: exoplanet.distributions.get_joint_radius_impact


Utilities
---------

.. autofunction:: exoplanet.utils.eval_in_model

.. autoclass:: exoplanet.sampling.TuningSchedule
   :inherited-members:


Citations
---------

.. autofunction:: exoplanet.citations.get_citations_for_model
