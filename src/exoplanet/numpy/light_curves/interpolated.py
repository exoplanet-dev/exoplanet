# -*- coding: utf-8 -*-

__all__ = ["InterpolatedLightCurve"]

from .. import compat
from ..compat import numpy as np


def interp(n, x, xmin, xmax, dx, func):
    xp = np.arange(xmin - dx, xmax + 2.5 * dx, dx)
    yp = func(xp)

    y0 = yp[:-3, n]
    y1 = yp[1:-2, n]
    y2 = yp[2:-1, n]
    y3 = yp[3:, n]

    a0 = y1
    a1 = -y0 / 3.0 - 0.5 * y1 + y2 - y3 / 6.0
    a2 = 0.5 * (y0 + y2) - y1
    a3 = 0.5 * ((y1 - y2) + (y3 - y0) / 3.0)

    inds = compat.as_tensor(np.floor((x - xmin) / dx), dtype="int64")
    x0 = (x - xp[inds + 1]) / dx
    return a0[inds] + a1[inds] * x0 + a2[inds] * x0 ** 2 + a3[inds] * x0 ** 3


class InterpolatedLightCurve:
    def __init__(self, base_light_curve, num_phase, num_planets, **kwargs):
        self.base_light_curve = base_light_curve
        self.num_phase = int(num_phase)
        self.num_planets = int(num_planets)

    def get_light_curve(
        self,
        orbit=None,
        t=None,
        texp=None,
        oversample=7,
        order=0,
        use_in_transit=None,
        light_delay=False,
    ):
        func = _wrapper(
            self.base_light_curve,
            orbit=orbit,
            texp=texp,
            oversample=oversample,
            order=order,
            use_in_transit=use_in_transit,
            light_delay=light_delay,
        )

        if self.num_planets <= 1:
            mn = orbit.t0
            mx = orbit.t0 + orbit.period
            return interp(
                0,
                np.mod(t - orbit.t0, orbit.period) + orbit.t0,
                mn,
                mx,
                (mx - mn) / (self.num_phase + 1),
                func,
            )[:, None]

        ys = []
        for n in range(self.num_planets):
            mn = orbit.t0[n]
            mx = orbit.t0[n] + orbit.period[n]
            ys.append(
                interp(
                    n,
                    np.mod(t - orbit.t0[n], orbit.period[n]) + orbit.t0[n],
                    mn,
                    mx,
                    (mx - mn) / (self.num_phase + 1),
                    func,
                )
            )

        return np.stack(ys, axis=-1)


class _wrapper:
    def __init__(self, base_light_curve, *args, **kwargs):
        self.base_light_curve = base_light_curve
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        kwargs = dict(t=x, **self.kwargs)
        return self.base_light_curve.get_light_curve(*self.args, **kwargs)
