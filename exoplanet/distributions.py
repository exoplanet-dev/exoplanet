# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["UnitVector"]

import pymc3 as pm

from . import transforms as tr


class UnitVector(pm.Normal):
    """A vector where the sum of squares is fixed to unity

    For a multidimensional shape, the normalization is performed along the
    last dimension.

    """

    def __init__(self, *args, **kwargs):
        kwargs["transform"] = tr.unit_vector
        super(UnitVector, self).__init__(*args, **kwargs)


class Angle(pm.Flat):
    """An angle constrained to be in the range -pi to pi

    The actual sampling is performed in the two dimensional vector space
    ``(sin(theta), cos(theta))`` so that the sampler doesn't see a
    discontinuity at pi.

    """

    def __init__(self, *args, **kwargs):
        kwargs["transform"] = tr.angle
        super(Angle, self).__init__(*args, **kwargs)
