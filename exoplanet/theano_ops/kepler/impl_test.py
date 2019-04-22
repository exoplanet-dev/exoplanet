# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

from .impl import get_roots_general


def get_b2(f, a, e, cosw, sinw, cosi, sini):
    cosf = np.cos(f)
    sinf = np.sin(f)
    e2 = e**2
    return a**2*(e2 - 1)**2*(cosi**2*(cosw*sinf + sinw*cosf)**2 +
                             (cosw*cosf - sinw*sinf)**2)/(e*cosf + 1)**2


def compute(L, a, b, e, w):
    target = L**2
    cosw = np.cos(w)
    sinw = np.sin(w)

    incl_factor = (1 + e * sinw) / (1 - e**2)
    cosi = incl_factor * b * L / a
    if np.abs(cosi) >= 1:
        return
    i = np.arccos(cosi)
    sini = np.sin(i)

    fs, flag = get_roots_general(a, e, cosw, sinw, cosi, sini, L)
    if flag:
        return

    assert len(fs) == 2
    assert np.all(np.isfinite(fs))
    for f in fs:
        assert np.allclose(target, get_b2(f, a, e, cosw, sinw, cosi, sini))


@pytest.mark.parametrize("a", [5.0, 12.1234, 100.0, 1000.0, 20000.0])
@pytest.mark.parametrize("L", [0.7, 0.9, 1.0, 1.1, 1.5])
def test_contact_point_impl(a, L):
    es = np.linspace(0, 1, 25)[:-1]
    ws = np.linspace(-np.pi, np.pi, 51)
    bs = np.linspace(0, 1 - 1e-5, 5)
    for bi, b in enumerate(bs):
        for ei, e in enumerate(es):
            for wi, w in enumerate(ws):
                compute(L, a, b, e, w)
