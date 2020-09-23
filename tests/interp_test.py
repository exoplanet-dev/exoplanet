# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import (
    RegularGridInterpolator as SpRegularGridInterpolator,
)

from exoplanet.interp import RegularGridInterpolator


def test_basic():
    def f(x, y, z):
        return 2 * x ** 3 + 3 * y ** 2 - z

    x = np.linspace(1, 4, 11)
    y = np.linspace(4, 7, 22)
    z = np.linspace(7, 9, 33)

    data = f(*np.meshgrid(x, y, z, indexing="ij", sparse=True))
    interpolator = SpRegularGridInterpolator((x, y, z), data)

    pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
    f0 = interpolator(pts)

    op = RegularGridInterpolator((x, y, z), data)
    f = np.squeeze(op.evaluate(pts).eval())
    assert np.allclose(f, f0)
