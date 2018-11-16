# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

import starry

from .light_curve import StarryLightCurve


def test_light_curve():
    u = tt.vector()
    b = tt.vector()
    r = tt.vector()
    lc = StarryLightCurve(u)
    f = lc.compute_light_curve(b, r)
    func = theano.function([u, b, r], f)

    u_val = np.array([0.2, 0.3, 0.1, 0.5])
    b_val = np.linspace(-1.5, 1.5, 100)
    r_val = 0.1 + np.zeros_like(b_val)

    m = starry.Map(lmax=len(u_val))
    m[:] = u_val
    expect = m.flux(xo=b_val, ro=r_val)

    evaluated = func(u_val, b_val, r_val)

    utt.assert_allclose(expect, evaluated)


def test_light_curve_grad():
    u_val = np.array([0.2, 0.3, 0.1, 0.5])
    b_val = np.linspace(-1.5, 1.5, 20)
    r_val = 0.1 + np.zeros_like(b_val)

    lc = lambda u, b, r: StarryLightCurve(u).compute_light_curve(b, r)  # NOQA
    utt.verify_grad(lc, [u_val, b_val, r_val])
