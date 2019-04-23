# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

import starry

from .orbits import KeplerianOrbit
from .light_curves import StarryLightCurve


def test_light_curve():
    u = tt.vector()
    b = tt.vector()
    r = tt.vector()
    lc = StarryLightCurve(u)
    f = lc._compute_light_curve(b, r)
    func = theano.function([u, b, r], f)

    u_val = np.array([0.2, 0.3, 0.1, 0.5])
    b_val = np.linspace(-1.5, 1.5, 100)
    r_val = 0.1 + np.zeros_like(b_val)

    m = starry.Map(lmax=len(u_val))
    m[:] = u_val
    expect = m.flux(xo=b_val, ro=r_val) - 1

    evaluated = func(u_val, b_val, r_val)

    utt.assert_allclose(expect, evaluated)


def test_light_curve_grad():
    u_val = np.array([0.2, 0.3, 0.1, 0.5])
    b_val = np.linspace(-1.5, 1.5, 20)
    r_val = 0.1 + np.zeros_like(b_val)

    lc = lambda u, b, r: StarryLightCurve(u)._compute_light_curve(b, r)  # NOQA
    utt.verify_grad(lc, [u_val, b_val, r_val])


def test_in_transit():
    t = np.linspace(-20, 20, 1000)
    m_planet = np.array([0.3, 0.5])
    m_star = 1.45
    orbit = KeplerianOrbit(
        m_star=m_star,
        r_star=1.5,
        t0=np.array([0.5, 17.4]),
        period=np.array([10.0, 5.3]),
        ecc=np.array([0.1, 0.8]),
        omega=np.array([0.5, 1.3]),
        m_planet=m_planet,
    )
    u = np.array([0.2, 0.3, 0.1, 0.5])
    r = np.array([0.1, 0.01])

    lc = StarryLightCurve(u)
    model1 = lc.get_light_curve(r=r, orbit=orbit, t=t)
    model2 = lc.get_light_curve(r=r, orbit=orbit, t=t, use_in_transit=False)
    vals = theano.function([], [model1, model2])()
    utt.assert_allclose(*vals)

    model1 = lc.get_light_curve(r=r, orbit=orbit, t=t, texp=0.1)
    model2 = lc.get_light_curve(r=r, orbit=orbit, t=t, texp=0.1,
                                use_in_transit=False)
    vals = theano.function([], [model1, model2])()
    utt.assert_allclose(*vals)


def test_contact_bug():
    orbit = KeplerianOrbit(period=3.456, ecc=0.6, omega=-1.5)
    t = np.linspace(-0.1, 0.1, 1000)
    u = [0.3, 0.2]
    y1 = StarryLightCurve(u).get_light_curve(
        orbit=orbit, r=0.1, t=t, texp=0.02).eval()
    y2 = StarryLightCurve(u).get_light_curve(
        orbit=orbit, r=0.1, t=t, texp=0.02, use_in_transit=False).eval()
    assert np.allclose(y1, y2)


def test_small_star():
    from batman.transitmodel import TransitModel, TransitParams
    u_star = [0.2, 0.1]
    r = 0.04221468

    m_star = 0.151
    r_star = 0.189
    period = 0.4626413
    t0 = 0.2
    b = 0.5
    ecc = 0.1
    omega = 0.1
    t = np.linspace(0, period, 500)

    r_pl = r * r_star

    orbit = KeplerianOrbit(
        r_star=r_star, m_star=m_star,
        period=period, t0=t0, b=b,
        ecc=ecc, omega=omega)
    a = orbit.a.eval()
    incl = orbit.incl.eval()

    lc = StarryLightCurve(u_star)

    model1 = lc.get_light_curve(r=r_pl, orbit=orbit, t=t)
    model2 = lc.get_light_curve(r=r_pl, orbit=orbit, t=t, use_in_transit=False)
    vals = theano.function([], [model1, model2])()
    utt.assert_allclose(*vals)

    params = TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = r
    params.a = a / r_star
    params.inc = np.degrees(incl)
    params.ecc = ecc
    params.w = np.degrees(omega)
    params.u = u_star
    params.limb_dark = "quadratic"

    model = TransitModel(params, t)
    flux = model.light_curve(params)
    utt.assert_allclose(vals[0][:, 0], flux - 1)
