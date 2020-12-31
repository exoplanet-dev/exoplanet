# -*- coding: utf-8 -*-

import numpy as np

import exoplanet.pymc as xo


def test_secondary_eclipse():
    u1 = np.array([0.3, 0.2])
    lc1 = xo.LimbDarkLightCurve(*u1)

    u2 = np.array([0.4, 0.1])
    lc2 = xo.LimbDarkLightCurve(*u1)

    s = 0.3
    ror = 0.08
    f = ror ** 2 * s
    lc = xo.light_curves.SecondaryEclipseLightCurve(u1, u2, s)

    t = np.linspace(-6.435, 10.4934, 5000)
    orbit1 = xo.orbits.KeplerianOrbit(period=1.543, t0=-0.123, r_planet=ror)
    orbit2 = xo.orbits.KeplerianOrbit(
        period=orbit1.period,
        t0=orbit1.t0 + 0.5 * orbit1.period,
        r_star=ror,
        m_star=1.0,
        r_planet=1.0,
    )

    y1 = lc1.get_light_curve(orbit=orbit1, t=t).eval()
    y2 = lc2.get_light_curve(orbit=orbit2, t=t).eval()
    y = lc.get_light_curve(orbit=orbit1, t=t).eval()
    y_expect = (y1 + f * y2) / (1 + f)

    assert np.allclose(y_expect, y, atol=5e-6)
