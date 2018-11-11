# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["light_curve"]

import numpy as np

import theano.tensor as tt

from .get_cl import GetClOp
from .limbdark import LimbDarkOp

get_cl = GetClOp()
limbdark = LimbDarkOp()


def light_curve(u, b, r):
    u_ext = tt.concatenate([-1 + tt.zeros(1, dtype=u.dtype), u])
    c = get_cl(u_ext)
    c_norm = c / (np.pi * (c[0] + 2 * c[1] / 3))
    return limbdark(c_norm, b, r)
