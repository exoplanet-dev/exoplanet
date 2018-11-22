.. module:: exoplanet

.. _api:

API documentation
=================

Orbits
------

.. autoclass:: exoplanet.orbits.KeplerianOrbit
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


Distributions
-------------

.. autoclass:: exoplanet.distributions.UnitVector
.. autoclass:: exoplanet.distributions.Angle
.. autoclass:: exoplanet.distributions.Triangle
.. autoclass:: exoplanet.distributions.RadiusImpactParameter


Utilities
---------

.. autofunction:: exoplanet.utils.eval_in_model

.. autoclass:: exoplanet.sampling.TuningSchedule
   :inherited-members:


Citations
---------

.. autofunction:: exoplanet.citations.get_citations_for_model
