# -*- coding: utf-8 -*-

import numpy as np
import pytest
import theano

from ..light_curves import LimbDarkLightCurve
from .simple import SimpleTransitOrbit


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

    orbit = SimpleTransitOrbit(period, t0, b, duration, r_star)

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


def test_simple_light_curve():
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
    orbit = SimpleTransitOrbit(period, t0, b, duration, r_star)

    star = LimbDarkLightCurve([0.2, 0.3])
    lc1 = star.get_light_curve(
        orbit=orbit, r=0.01, t=t, use_in_transit=False
    ).eval()
    lc2 = star.get_light_curve(orbit=orbit, r=0.01, t=t).eval()
    assert np.allclose(lc1, lc2)
    assert np.allclose(lc2[~m0], 0.0)

    lc1 = star.get_light_curve(
        orbit=orbit, r=0.01, t=t, texp=0.01, use_in_transit=False
    ).eval()
    lc2 = star.get_light_curve(orbit=orbit, r=0.01, t=t, texp=0.01).eval()
    assert np.allclose(lc1, lc2)
