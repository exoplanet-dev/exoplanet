# -*- coding: utf-8 -*-

from __future__ import division, print_function

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
    i = np.arccos(cosi)
    sini = np.sin(i)

    fs = get_roots_general(a, e, cosw, sinw, cosi, sini, L)

    assert len(fs) == 2
    assert np.all(np.isfinite(fs))
    for f in fs:
        assert np.allclose(target, get_b2(f, a, e, cosw, sinw, cosi, sini))


def test_contact_point_impl(L=1.1, a=100.0):
    es = np.linspace(0, 1, 50)[:-1]
    ws = np.linspace(-np.pi, np.pi, 71)
    bs = np.linspace(0, 1 - 1e-5, 5)
    for bi, b in enumerate(bs):
        for ei, e in enumerate(es):
            for wi, w in enumerate(ws):
                print(bi, ei, wi)
                compute(L, a, b, e, w)
