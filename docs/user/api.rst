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

.. autoclass:: exoplanet.LimbDarkLightCurve
   :inherited-members:

.. autoclass:: exoplanet.SecondaryEclipseLightCurve
   :inherited-members:


Estimators
----------

.. autofunction:: exoplanet.estimate_semi_amplitude
.. autofunction:: exoplanet.estimate_minimum_mass
.. autofunction:: exoplanet.lomb_scargle_estimator
.. autofunction:: exoplanet.autocorr_estimator
.. autofunction:: exoplanet.bls_estimator


Distributions
-------------

Physical distributions
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: exoplanet.distributions.angle
.. autofunction:: exoplanet.distributions.unit_disk
.. autofunction:: exoplanet.distributions.quad_limb_dark
.. autofunction:: exoplanet.distributions.impact_parameter

Eccentricity distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: exoplanet.distributions.eccentricity.kipping13
.. autofunction:: exoplanet.distributions.eccentricity.vaneylen19


Miscellaneous
-------------

.. autofunction:: exoplanet.orbits.ttv.compute_expected_transit_times

.. autofunction:: exoplanet.units.with_unit
.. autofunction:: exoplanet.units.has_unit
.. autofunction:: exoplanet.units.to_unit

.. autofunction:: exoplanet.citations.get_citations_for_model
