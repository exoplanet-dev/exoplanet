# -*- coding: utf-8 -*-

import astropy.units as u
import numpy as np

from exoplanet import units
from exoplanet.orbits import KeplerianOrbit


def test_mass_units():
    P_earth = 365.256
    Tper_earth = 2454115.5208333
    inclination_earth = np.radians(45.0)

    orbit1 = KeplerianOrbit(
        period=P_earth,
        t_periastron=Tper_earth,
        incl=inclination_earth,
        m_planet=units.with_unit(1.0, u.M_earth),
    )
    orbit2 = KeplerianOrbit(
        period=P_earth,
        t_periastron=Tper_earth,
        incl=inclination_earth,
        m_planet=1.0,
        m_planet_units=u.M_earth,
    )

    t = np.linspace(Tper_earth, Tper_earth + 1000, 1000)
    rv1 = orbit1.get_radial_velocity(t).eval()
    rv_diff = np.max(rv1) - np.min(rv1)
    assert rv_diff < 1.0, "with_unit"

    rv2 = orbit2.get_radial_velocity(t).eval()
    rv_diff = np.max(rv2) - np.min(rv2)
    assert rv_diff < 1.0, "m_planet_units"
    np.testing.assert_allclose(rv2, rv1)
