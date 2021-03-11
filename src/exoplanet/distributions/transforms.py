# -*- coding: utf-8 -*-

__all__ = [
    "quad_limb_dark",
    "radius_impact",
    "impact_parameter",
]

import aesara_theano_fallback.tensor as tt
import numpy as np
import pymc3.distributions.transforms as tr
from pymc3.distributions import draw_values


class QuadLimbDarkTransform(tr.Transform):
    """A triangle transformation for PyMC3

    Ref: https://arxiv.org/abs/1308.0009

    """

    name = "quadlimbdark"

    def backward(self, y):
        q = tt.nnet.sigmoid(y)
        sqrtq1 = tt.sqrt(q[0])
        twoq2 = 2 * q[1]
        u = tt.stack([sqrtq1 * twoq2, sqrtq1 * (1 - twoq2)])

        return u

    def forward(self, x):
        usum = tt.sum(x, axis=0)
        q = tt.stack([usum ** 2, 0.5 * x[0] / usum])
        return tt.log(q) - tt.log(1 - q)

    def forward_val(self, x, point=None):
        usum = np.sum(x, axis=0)
        q = np.array([usum ** 2, 0.5 * x[0] / usum])
        return np.log(q) - np.log(1 - q)

    def jacobian_det(self, y):
        return -2 * tt.nnet.softplus(-y) - y


quad_limb_dark = QuadLimbDarkTransform()


class RadiusImpactTransform(tr.Transform):
    """A reparameterization of the radius-impact parameter plane

    This is an implementation of `Espinoza (2018)
    <http://iopscience.iop.org/article/10.3847/2515-5172/aaef38/meta>`_

    The last axis of the shape of the parameter should be exactly 2. The
    radius ratio will be in the zeroth entry in the last dimension and
    the impact parameter will be in the first.

    """

    name = "radiusimpact"

    def __init__(self, min_radius, max_radius):
        self.min_radius = tt.as_tensor_variable(min_radius)
        self.max_radius = tt.as_tensor_variable(max_radius)

        # Compute Ar from Espinoza
        self.dr = self.max_radius - self.min_radius
        denom = 2 + self.min_radius + self.max_radius
        self.Ar = self.dr / denom

    def backward(self, y):
        y = tt.nnet.sigmoid(y)
        r1 = y[0]
        r2 = y[1]
        pl, pu = self.min_radius, self.max_radius

        b1 = (1 + pl) * (1 + (r1 - 1) / (1 - self.Ar))
        p1 = pl + r2 * self.dr

        q1 = r1 / self.Ar
        q2 = r2
        b2 = (1 + pl) + tt.sqrt(q1) * q2 * self.dr
        p2 = pu - self.dr * tt.sqrt(q1) * (1 - q2)

        pb = tt.switch(
            r1 > self.Ar,
            tt.stack((p1, b1), axis=0),
            tt.stack((p2, b2), axis=0),
        )
        return pb

    def forward(self, x):
        p = x[0]
        b = x[1]
        pl = self.min_radius

        r11 = (b / (1 + pl) - 1) * (1 - self.Ar) + 1
        r21 = (p - pl) / self.dr

        arg = p - b - self.dr + 1
        q1 = (arg / self.dr) ** 2
        q2 = (pl - b + 1) / arg
        r12 = q1 * self.Ar
        r22 = q2

        y = tt.switch(
            b <= 1, tt.stack((r11, r21), axis=0), tt.stack((r12, r22), axis=0)
        )

        return tt.log(y) - tt.log(1 - y)

    def forward_val(self, x, point=None):
        p = x[0]
        b = x[1]
        pl, Ar, dr = draw_values(
            [self.min_radius - 0.0, self.Ar - 0.0, self.dr - 0.0], point=point
        )

        m = b <= 1
        r = np.empty_like(x)
        r[0, m] = (b[m] / (1 + pl) - 1) * (1 - Ar) + 1
        r[1, m] = (p[m] - pl) / dr

        arg = p[~m] - b[~m] - dr + 1
        q1 = (arg / dr) ** 2
        q2 = (pl - b[~m] + 1) / arg
        r[0, ~m] = q1 * Ar
        r[1, ~m] = q2

        return np.log(r) - np.log(1 - r)

    def jacobian_det(self, y):
        return -2 * tt.nnet.softplus(-y) - y


radius_impact = RadiusImpactTransform


class ImpactParameterTransform(tr.LogOdds):

    name = "impact"

    def __init__(self, ror):
        self.one_plus_ror = 1 + tt.as_tensor_variable(ror)

    def backward(self, x):
        bhat = super(ImpactParameterTransform, self).backward(x)
        return bhat * self.one_plus_ror

    def backward_val(self, x):
        raise NotImplementedError(
            "backward_val isn't implemented for the impact parameter transform"
        )

    def forward(self, x):
        return super(ImpactParameterTransform, self).forward(
            x / self.one_plus_ror
        )

    def forward_val(self, x, point=None):
        (opror,) = draw_values([self.one_plus_ror - 0.0], point=point)
        return super(ImpactParameterTransform, self).forward_val(
            x / opror, point=point
        )

    def jacobian_det(self, y):
        # This is y here, not y / (1 + ror) because the jacobian is computed
        # using the 'backward' op *of this transform* not its super.
        jac = super(ImpactParameterTransform, self).jacobian_det(y)
        return jac - tt.log(self.one_plus_ror)


impact_parameter = ImpactParameterTransform
