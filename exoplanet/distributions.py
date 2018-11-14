# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["UnitVector", "Angle", "RadiusImpactParameter"]

import numpy as np

import pymc3 as pm
from pymc3.distributions import draw_values

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
        self._default = np.zeros(self.shape)


class Triangle(pm.Flat):
    """

    """

    def __init__(self, *args, **kwargs):
        # Make sure that the shape is compatible
        shape = kwargs.get("shape", 2)
        try:
            if list(shape)[0] != 2:
                raise ValueError("the first dimension should be exactly 2")
        except TypeError:
            if shape != 2:
                raise ValueError("the first dimension should be exactly 2")

        kwargs["shape"] = shape
        kwargs["transform"] = tr.triangle

        super(Triangle, self).__init__(*args, **kwargs)

        # Work out some reasonable starting values for the parameters
        default = np.zeros(shape)
        default[0] = np.sqrt(0.5)
        default[1] = 0.0
        self._default = default


class RadiusImpactParameter(pm.Flat):
    """The Espinoza (2018) distribution over radius and impact parameter

    This is an implementation of `Espinoza (2018)
    <http://iopscience.iop.org/article/10.3847/2515-5172/aaef38/meta>`_
    The first axis of the shape of the parameter should be exactly 2. The
    radius ratio will be in the zeroth entry in the first dimension and
    the impact parameter will be in the first.

    """

    def __init__(self, *args, **kwargs):
        # Make sure that the shape is compatible
        shape = kwargs.get("shape", 2)
        try:
            if list(shape)[0] != 2:
                raise ValueError("the first dimension should be exactly 2")
        except TypeError:
            if shape != 2:
                raise ValueError("the first dimension should be exactly 2")

        min_radius = kwargs.pop("min_radius", 0)
        max_radius = kwargs.pop("max_radius", 1)
        transform = tr.radius_impact(min_radius, max_radius)
        kwargs["shape"] = shape
        kwargs["transform"] = transform

        super(RadiusImpactParameter, self).__init__(*args, **kwargs)

        # Work out some reasonable starting values for the parameters
        default = np.zeros(shape)
        mn, mx = draw_values([min_radius-0., max_radius-0.])
        default[0] = 0.5 * (mn + mx)
        default[1] = 0.5
        self._default = default
