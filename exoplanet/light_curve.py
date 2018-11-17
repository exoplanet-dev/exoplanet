# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StarryLightCurve"]

import numpy as np

import theano.tensor as tt

from .theano_ops.starry.get_cl import GetClOp
from .theano_ops.starry.limbdark import LimbDarkOp

get_cl = GetClOp()
limbdark = LimbDarkOp()


class StarryLightCurve(object):

    def __init__(self, u, r_star=1.0):
        self.r_star = tt.as_tensor_variable(r_star)
        self.u = tt.as_tensor_variable(u)
        u_ext = tt.concatenate([-1 + tt.zeros(1, dtype=self.u.dtype), self.u])
        self.c = get_cl(u_ext)
        self.c_norm = self.c / (np.pi * (self.c[0] + 2 * self.c[1] / 3))

    def get_light_curve(self, r, orbit, t, texp=None, oversample=7, order=2):
        r = tt.as_tensor_variable(r)
        t = tt.as_tensor_variable(t)

        if texp is None:
            tgrid = t
            rgrid = r
        else:
            texp = tt.as_tensor_variable(texp)

            oversample = int(oversample)
            oversample += 1 - oversample % 2
            stencil = np.ones(oversample)

            # Construct the exposure time integration stencil
            if order == 0:
                dt = np.linspace(-0.5, 0.5, 2*oversample+1)[1:-1:2]
            elif order == 1:
                dt = np.linspace(-0.5, 0.5, oversample)
                stencil[1:-1] = 2
            elif order == 2:
                dt = np.linspace(-0.5, 0.5, oversample)
                stencil[1:-1:2] = 4
                stencil[2:-1:2] = 2
            else:
                raise ValueError("order must be <= 2")
            stencil /= np.sum(stencil)

            if texp.ndim == 0:
                dt = texp * dt
            else:
                dt = tt.shape_padright(texp) * dt
            tgrid = tt.shape_padright(t) + dt

            # Madness to get the shapes to work out...
            rgrid = tt.shape_padleft(r, tgrid.ndim + 1) \
                + tt.shape_padright(tt.zeros_like(tgrid), r.ndim)

        coords = orbit.get_relative_position(tgrid)
        b = tt.sqrt(coords[0]**2 + coords[1]**2)

        lc = self.compute_light_curve(
            b/self.r_star, rgrid/self.r_star, coords[2]/self.r_star)

        if texp is not None:
            stencil = tt.shape_padright(tt.shape_padleft(stencil, t.ndim),
                                        r.ndim)
            lc = tt.sum(stencil * lc, axis=t.ndim)

        return lc

    def compute_light_curve(self, b, r, los=None):
        if los is None:
            los = -tt.ones_like(b)
        return limbdark(self.c_norm, b, r, los)
