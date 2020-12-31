# -*- coding: utf-8 -*-

__all__ = ["QuadLimbDark", "ImpactParameter"]

import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions import draw_values, generate_samples

from ..citations import add_citations_to_model
from . import transforms as tr


class QuadLimbDark(pm.Flat):
    """An uninformative prior for quadratic limb darkening parameters

    This is an implementation of the `Kipping (2013)
    <https://arxiv.org/abs/1308.0009>`_ reparameterization of the
    two-parameter limb darkening model to allow for efficient and
    uninformative sampling.

    """

    __citations__ = ("kipping13",)

    def __init__(self, *args, **kwargs):
        add_citations_to_model(self.__citations__, kwargs.get("model", None))

        # Make sure that the shape is compatible
        shape = kwargs.get("shape", 2)
        try:
            if list(shape)[0] != 2:
                raise ValueError("the first dimension should be exactly 2")
        except TypeError:
            if shape != 2:
                raise ValueError("the first dimension should be exactly 2")

        kwargs["shape"] = shape
        kwargs["transform"] = tr.quad_limb_dark

        super(QuadLimbDark, self).__init__(*args, **kwargs)

        # Work out some reasonable starting values for the parameters
        default = np.zeros(shape)
        default[0] = np.sqrt(0.5)
        default[1] = 0.0
        self._default = default

    def _random(self, size=None):
        q = np.moveaxis(
            np.random.uniform(0, 1, size=size), 0, -len(self.shape)
        )
        sqrtq1 = np.sqrt(q[0])
        twoq2 = 2 * q[1]
        u = np.stack([sqrtq1 * twoq2, sqrtq1 * (1 - twoq2)], axis=0)
        return np.moveaxis(u, 0, -len(self.shape))

    def random(self, point=None, size=None):
        return generate_samples(
            self._random,
            dist_shape=self.shape,
            broadcast_shape=self.shape,
            size=size,
        )

    def logp(self, value):
        return tt.zeros_like(tt.as_tensor_variable(value))


class ImpactParameter(pm.Flat):
    """The impact parameter distribution for a transiting planet

    Args:
        ror: A scalar, tensor, or PyMC3 distribution representing the radius
            ratio between the planet and star. Conditioned on a value of
            ``ror``, this will be uniformly distributed between ``0`` and
            ``1+ror``.

    """

    def __init__(self, ror=None, **kwargs):
        if ror is None:
            raise ValueError("missing required parameter 'ror'")
        self.ror = tt.as_tensor_variable(ror)
        kwargs["transform"] = kwargs.pop(
            "transform", tr.ImpactParameterTransform(self.ror)
        )

        try:
            shape = kwargs.get("shape", self.ror.distribution.shape)
        except AttributeError:
            shape = None
        if shape is None:
            testval = 0.5
        else:
            testval = 0.5 + np.zeros(shape)
            kwargs["shape"] = shape
        kwargs["testval"] = kwargs.pop("testval", testval)

        super(ImpactParameter, self).__init__(**kwargs)

    def _random(self, ror=0.0, size=None):
        return np.random.uniform(0, 1 + ror, size)

    def random(self, point=None, size=None):
        (ror,) = draw_values([self.ror], point=point, size=size)
        return generate_samples(
            self._random,
            dist_shape=self.shape,
            broadcast_shape=self.shape,
            size=size,
        )

    def logp(self, value):
        return tt.zeros_like(tt.as_tensor_variable(value))
