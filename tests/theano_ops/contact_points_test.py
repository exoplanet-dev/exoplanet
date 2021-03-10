# -*- coding: utf-8 -*-

import aesara_theano_fallback.tensor as tt
import numpy as np
import pytest
from aesara_theano_fallback import aesara as theano

from exoplanet.theano_ops.contact_points import ContactPoints
from exoplanet.theano_ops.kepler import kepler
from exoplanet.theano_ops.test_tools import InferShapeTester


class TestContactPoints(InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = ContactPoints
        self.op = ContactPoints()

    def test_infer_shape(self):
        np.random.seed(42)
        args = [tt.dvector() for i in range(7)]
        vals = [np.random.rand(50) for i in range(7)]
        self._compile_and_check(args, self.op(*args), vals, self.op_class)

    def test_basic(self):
        a = np.float64(100.0)
        e = np.float64(0.3)
        w = 0.1
        cosw = np.float64(np.cos(w))
        sinw = np.float64(np.sin(w))
        i = 0.5 * np.pi - 1e-2
        cosi = np.float64(np.cos(i))
        sini = np.float64(np.sin(i))
        L = np.float64(1.1 + 0.1)

        M_expect = np.array([0.88809465, 0.90313756])
        M_calc = theano.function(
            [], self.op(a, e, cosw, sinw, cosi, sini, L)
        )()

        assert np.all(M_calc[2] == 0)
        assert np.allclose(M_calc[:2], M_expect)


class Solver:
    def __init__(self):
        self.op = ContactPoints()
        a = tt.dscalar()
        e = tt.dscalar()
        cosw = tt.dscalar()
        sinw = tt.dscalar()
        cosi = tt.dscalar()
        sini = tt.dscalar()
        L = tt.dscalar()
        M1, M2, flag = self.op(a, e, cosw, sinw, cosi, sini, L)

        sinf1, cosf1 = kepler(M1, e)
        sinf2, cosf2 = kepler(M2, e)

        self.func = theano.function(
            [a, e, cosw, sinw, cosi, sini, L],
            [sinf1, cosf1, sinf2, cosf2, flag],
        )

    def get_b2(self, sinf, cosf, a, e, cosw, sinw, cosi, sini):
        e2 = e ** 2
        factor = (a * (e2 - 1) / (e * cosf + 1)) ** 2
        return factor * (
            cosi ** 2 * (cosw * sinf + sinw * cosf) ** 2
            + (cosw * cosf - sinw * sinf) ** 2
        )

    def compute(self, L, a, b, e, w):
        target = L ** 2
        cosw = np.cos(w)
        sinw = np.sin(w)

        incl_factor = (1 + e * sinw) / (1 - e ** 2)
        cosi = incl_factor * b * L / a
        if np.abs(cosi) >= 1:
            return
        i = np.arccos(cosi)
        sini = np.sin(i)

        sinf1, cosf1, sinf2, cosf2, flag = self.func(
            a, e, cosw, sinw, cosi, sini, L
        )
        if np.any(flag):
            return

        fs = [(sinf1, cosf1), (sinf2, cosf2)]
        assert np.all(np.isfinite(fs))
        for sinf, cosf in fs:
            assert np.allclose(
                target, self.get_b2(sinf, cosf, a, e, cosw, sinw, cosi, sini)
            )


@pytest.mark.parametrize("a", [5.0, 12.1234, 100.0, 1000.0, 20000.0])
@pytest.mark.parametrize("L", [0.7, 0.9, 1.0, 1.1, 1.5])
def test_contact_point_impl(a, L):
    solver = Solver()
    es = np.linspace(0, 1, 25)[:-1]
    ws = np.linspace(-np.pi, np.pi, 51)
    bs = np.linspace(0, 1 - 1e-5, 5)
    for bi, b in enumerate(bs):
        for ei, e in enumerate(es):
            for wi, w in enumerate(ws):
                solver.compute(L, a, b, e, w)


def test_shape_segfault():
    args = [
        np.array(16.38049484),
        np.array([0.0002, 0.0002]),
        np.array([0.70710678, 0.70710678]),
        np.array([0.70710678, 0.70710678]),
        np.array([0.04579264, 0.04579264]),
        np.array([0.99895097, 0.99895097]),
        np.array([1.51741758]),
    ]

    with pytest.raises(ValueError):
        theano.function([], ContactPoints()(*args))()
