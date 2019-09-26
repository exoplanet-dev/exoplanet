# -*- coding: utf-8 -*-

import numpy as np
import pytest
import theano

from ..light_curves import LimbDarkLightCurve
from .keplerian import KeplerianOrbit
from .rebound import ReboundOrbit


@pytest.mark.parametrize(
    "orbit",
    (
        ReboundOrbit(
            period=50.0,
            t0=1.55,
            ecc=0.28,
            omega=-4.56,
            m_planet=0.12,
            b=0.51,
            m_star=1.51,
            r_star=1.0,
        ),
        ReboundOrbit(
            period=[50.0, 87.5],
            t0=[1.55, 10.6],
            ecc=[0.28, 0.01],
            omega=[-4.56, 1.5],
            m_planet=[0.0, 0.0],
            b=[0.51, 0.21],
            m_star=1.51,
            r_star=1.0,
        ),
    ),
)
def test_keplerian(orbit):
    t = np.linspace(50, 1000, 1045)

    x, y, z = theano.function([], orbit.get_relative_position(t))()
    x0, y0, z0 = theano.function(
        [], super(ReboundOrbit, orbit).get_relative_position(t)
    )()
    assert np.allclose(x, x0)
    assert np.allclose(y, y0)
    assert np.allclose(z, z0)

    x, y, z = theano.function([], orbit.get_planet_position(t))()
    x0, y0, z0 = theano.function(
        [], super(ReboundOrbit, orbit).get_planet_position(t)
    )()
    assert np.allclose(x, x0)
    assert np.allclose(y, y0)
    assert np.allclose(z, z0)

    x, y, z = theano.function([], orbit.get_star_position(t))()
    x0, y0, z0 = theano.function(
        [], super(ReboundOrbit, orbit).get_star_position(t)
    )()
    if len(np.shape(x0)) > 1:
        assert np.allclose(x, np.sum(x0, axis=-1))
        assert np.allclose(y, np.sum(y0, axis=-1))
        assert np.allclose(z, np.sum(z0, axis=-1))
    else:
        assert np.allclose(x, x0)
        assert np.allclose(y, y0)
        assert np.allclose(z, z0)


@pytest.mark.xfail(reason="I don't understand Theano sometimes")
def test_tensor_bug():
    orbit = ReboundOrbit(
        period=50.0, t0=0.0, ecc=0.5, omega=0.1, b=0.2, m_planet=0.5
    )
    orbit2 = ReboundOrbit(
        period=50.0,
        t0=0.0,
        ecc=0.5 + 1e-9,
        omega=0.1,
        b=0.2,
        m_planet=0.5 - 1e-9,
    )
    t = np.linspace(50, 1000, 1045)

    x, y, z = theano.function([], orbit.get_relative_position(t))()
    x2, y2, z2 = theano.function([], orbit2.get_relative_position(t))()

    x0, y0, z0 = theano.function(
        [], super(ReboundOrbit, orbit).get_relative_position(t)
    )()
    x02, y02, z02 = theano.function(
        [], super(ReboundOrbit, orbit2).get_relative_position(t)
    )()

    assert np.allclose(x0, x02)
    assert np.allclose(y0, y02)
    assert np.allclose(z0, z02)

    assert np.allclose(x, x2)
    assert np.allclose(y, y2)
    assert np.allclose(z, z2)


def test_keplerian_light_curve():
    t = np.linspace(50, 1000, 1045)
    r = np.array([0.04, 0.02])
    args = dict(
        period=[50.0, 87.5],
        t0=[1.55, 10.6],
        ecc=[0.28, 0.01],
        omega=[-4.56, 1.5],
        m_planet=[0.0, 0.0],
        b=[0.51, 0.21],
        m_star=1.51,
        r_star=1.0,
    )
    orbit0 = KeplerianOrbit(**args)
    orbit = ReboundOrbit(**args)

    ld = LimbDarkLightCurve([0.2, 0.3])
    lc0 = ld.get_light_curve(orbit=orbit0, r=r, t=t).eval()
    lc = ld.get_light_curve(orbit=orbit, r=r, t=t).eval()

    assert np.allclose(lc0, lc)
