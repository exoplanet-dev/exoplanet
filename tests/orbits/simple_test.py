# -*- coding: utf-8 -*-

import numpy as np
import pytest
from aesara_theano_fallback import aesara as theano

from exoplanet.light_curves import LimbDarkLightCurve
from exoplanet.orbits import KeplerianOrbit, SimpleTransitOrbit


def test_simple():
    period = 3.456
    t0 = 1.45
    b = 0.5
    duration = 0.12
    r_star = 1.345

    t = t0 + np.linspace(-2 * period, 2 * period, 5000)
    m0 = (
        np.abs((t - t0 + 0.5 * period) % period - 0.5 * period)
        < 0.5 * duration
    )

    orbit = SimpleTransitOrbit(
        period=period, duration=duration, t0=t0, b=b, r_star=r_star
    )

    x, y, z = theano.function([], orbit.get_planet_position(t))()
    b_val = np.sqrt(x ** 2 + y ** 2)
    m = (b_val <= r_star) & (z > 0)

    assert np.all(m == m0)

    in_transit = orbit.in_transit(t).eval()
    assert np.all(b_val[in_transit] <= r_star)
    assert np.all(z[in_transit] > 0)

    x, y, z = theano.function([], orbit.get_star_position(t))()
    assert np.allclose(x, 0.0)
    assert np.allclose(y, 0.0)
    assert np.allclose(z, 0.0)

    with pytest.raises(NotImplementedError):
        orbit.get_planet_velocity(t)

    with pytest.raises(NotImplementedError):
        orbit.get_star_velocity(t)

    with pytest.raises(NotImplementedError):
        orbit.get_radial_velocity(t)


def test_simple_light_curve_compare_kepler():
    t = np.linspace(0.0, 1, 1000)
    # We use a long period, because at short periods there is a big difference
    # between a circular orbit and an object moving on a straight line.
    period = 1000
    t0 = 0.5
    r = 0.01
    r_star = 1
    b = 1 - r / r_star * 3

    star = LimbDarkLightCurve([0])
    orbit_keplerian = KeplerianOrbit(
        period=period, t0=t0, b=b, r_star=r_star, m_star=1
    )
    duration = (period / np.pi) * np.arcsin(
        ((r_star + r) ** 2 - (b * r_star) ** 2) ** 0.5 / orbit_keplerian.a
    ).eval()

    lc_keplerian = star.get_light_curve(orbit=orbit_keplerian, r=r, t=t)
    orbit_simple1 = SimpleTransitOrbit(
        period=period,
        t0=t0,
        b=b,
        duration=duration,
        r_star=r_star,
        ror=r / r_star,
    )
    lc_simple1 = star.get_light_curve(orbit=orbit_simple1, r=r, t=t)

    # Should look similar to Keplerian orbit
    assert np.allclose(lc_keplerian.eval(), lc_simple1.eval(), rtol=0.001)
