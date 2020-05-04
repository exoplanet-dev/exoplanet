# -*- coding: utf-8 -*-

__all__ = ["UnitUniform", "UnitVector", "UnitDisk", "Angle", "Periodic"]

import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions import generate_samples

from . import transforms as tr


class UnitUniform(pm.Flat):
    """A uniform distribution between zero and one

    This can sometimes get better performance than ``pm.Uniform.dist(0, 1)``.

    """

    def __init__(self, *args, **kwargs):
        kwargs["transform"] = kwargs.pop(
            "transform", pm.distributions.transforms.logodds
        )

        shape = kwargs.get("shape", None)
        if shape is None:
            testval = 0.5
        else:
            testval = 0.5 + np.zeros(shape)
        kwargs["testval"] = kwargs.pop("testval", testval)

        super(UnitUniform, self).__init__(*args, **kwargs)

    def _random(self, size=None):
        return np.random.uniform(0, 1, size)

    def random(self, point=None, size=None):
        return generate_samples(
            self._random,
            dist_shape=self.shape,
            broadcast_shape=self.shape,
            size=size,
        )

    def logp(self, value):
        return tt.zeros_like(tt.as_tensor_variable(value))


class UnitVector(pm.Normal):
    """A vector where the sum of squares is fixed to unity

    For a multidimensional shape, the normalization is performed along the
    last dimension.

    """

    def __init__(self, *args, **kwargs):
        kwargs["transform"] = tr.unit_vector
        super(UnitVector, self).__init__(*args, **kwargs)

    def _random(self, size=None):
        x = np.random.normal(size=size)
        return x / np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True))

    def random(self, point=None, size=None):
        return generate_samples(
            self._random,
            dist_shape=self.shape,
            broadcast_shape=self.shape,
            size=size,
        )


class UnitDisk(pm.Flat):
    """Two dimensional parameters constrianed to live within the unit disk

    This distribution is constrained such that the sum of squares along the
    zeroth axis will always be less than one. For example, in this code block:

    .. code-block:: python

        import theano.tensor as tt
        disk = UnitDisk("disk")
        radius = tt.sum(disk ** 2, axis=0)

    the tensor ``radius`` will always have a value in the range ``[0, 1)``.

    Note that the shape of this distribution must be two in the zeroth axis.

    """

    def __init__(self, *args, **kwargs):
        kwargs["transform"] = kwargs.pop("transform", tr.unit_disk)

        # Make sure that the shape is compatible
        shape = kwargs["shape"] = kwargs.get("shape", 2)
        try:
            if list(shape)[0] != 2:
                raise ValueError("the first dimension should be exactly 2")
        except TypeError:
            if shape != 2:
                raise ValueError("the first dimension should be exactly 2")

        super(UnitDisk, self).__init__(*args, **kwargs)

        # Work out some reasonable starting values for the parameters
        self._default = np.zeros(shape)

    def _random(self, size=None):
        sr = np.sqrt(np.random.uniform(0, 1, size))
        theta = np.random.uniform(-np.pi, np.pi, size)
        return np.moveaxis(
            np.stack((sr * np.cos(theta), sr * np.sin(theta))), 0, 1
        )

    def random(self, point=None, size=None):
        return generate_samples(
            self._random,
            dist_shape=self.shape[1:],
            broadcast_shape=self.shape[1:],
            size=size,
        )

    def logp(self, value):
        return tt.zeros_like(tt.sum(value, axis=0))


class Angle(pm.Continuous):
    """An angle constrained to be in the range -pi to pi

    The actual sampling is performed in the two dimensional vector space
    ``(sin(theta), cos(theta))`` so that the sampler doesn't see a
    discontinuity at pi.

    """

    def __init__(self, *args, **kwargs):
        transform = kwargs.pop("transform", None)
        if transform is None:
            if "regularized" in kwargs:
                transform = tr.AngleTransform(
                    regularized=kwargs.pop("regularized")
                )
            else:
                transform = tr.angle
        kwargs["transform"] = transform

        shape = kwargs.get("shape", None)
        if shape is None:
            testval = 0.0
        else:
            testval = np.zeros(shape)
        kwargs["testval"] = kwargs.pop("testval", testval)
        super(Angle, self).__init__(*args, **kwargs)

    def _random(self, size=None):
        return np.random.uniform(-np.pi, np.pi, size)

    def random(self, point=None, size=None):
        return generate_samples(
            self._random,
            dist_shape=self.shape,
            broadcast_shape=self.shape,
            size=size,
        )

    def logp(self, value):
        return tt.zeros_like(tt.as_tensor_variable(value))


class Periodic(pm.Continuous):
    """An periodic parameter in a given range

    Like the :class:`Angle` distribution, the actual sampling is performed in
    a two dimensional vector space ``(sin(theta), cos(theta))`` and then
    transformed into the range ``[lower, upper)``.

    Args:
        lower: The lower bound on the range.
        upper: The upper bound on the range.

    """

    def __init__(self, lower=0, upper=1, **kwargs):
        self.lower = lower
        self.upper = upper

        transform = kwargs.pop("transform", None)
        if transform is None:
            transform = tr.PeriodicTransform(
                lower=lower,
                upper=upper,
                regularized=kwargs.pop("regularized", 10.0),
            )
        kwargs["transform"] = transform

        shape = kwargs.get("shape", None)
        if shape is None:
            testval = 0.5 * (lower + upper)
        else:
            testval = 0.5 * (lower + upper) + np.zeros(shape)
        kwargs["testval"] = kwargs.pop("testval", testval)
        super(Periodic, self).__init__(**kwargs)

    def _random(self, size=None):
        return np.random.uniform(self.lower, self.upper, size)

    def random(self, point=None, size=None):
        return generate_samples(
            self._random,
            dist_shape=self.shape,
            broadcast_shape=self.shape,
            size=size,
        )

    def logp(self, value):
        return tt.zeros_like(tt.as_tensor_variable(value))
