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

.. autoclass:: exoplanet.orbits.ReboundOrbit
   :inherited-members:

.. autoclass:: exoplanet.orbits.SimpleTransitOrbit
   :inherited-members:


Light curve models
------------------

.. autoclass:: exoplanet.LimbDarkLightCurve
   :inherited-members:

.. autoclass:: exoplanet.SecondaryEclipseLightCurve
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
.. autofunction:: exoplanet.bls_estimator


Distributions
-------------

.. autofunction:: exoplanet.distributions.get_log_abs_det_jacobian
.. autofunction:: exoplanet.distributions.estimate_inverse_gamma_parameters


Base distributions
~~~~~~~~~~~~~~~~~~

.. autoclass:: exoplanet.distributions.UnitUniform
.. autoclass:: exoplanet.distributions.UnitVector
.. autoclass:: exoplanet.distributions.UnitDisk
.. autoclass:: exoplanet.distributions.Angle
.. autoclass:: exoplanet.distributions.Periodic

Physical distributions
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: exoplanet.distributions.QuadLimbDark
.. autoclass:: exoplanet.distributions.ImpactParameter

Eccentricity distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: exoplanet.distributions.eccentricity.kipping13
.. autofunction:: exoplanet.distributions.eccentricity.vaneylen19


Utilities
---------

.. autofunction:: exoplanet.optimize
.. autofunction:: exoplanet.eval_in_model
.. autofunction:: exoplanet.get_theano_function_for_var
.. autofunction:: exoplanet.get_args_for_theano_function
.. autofunction:: exoplanet.get_samples_from_trace
.. autofunction:: exoplanet.get_dense_nuts_step

.. autofunction:: exoplanet.orbits.ttv.compute_expected_transit_times


Units
-----

.. autofunction:: exoplanet.units.with_unit
.. autofunction:: exoplanet.units.has_unit
.. autofunction:: exoplanet.units.to_unit


Citations
---------

.. autofunction:: exoplanet.citations.get_citations_for_model
