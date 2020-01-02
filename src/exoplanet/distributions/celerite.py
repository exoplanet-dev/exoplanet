# -*- coding: utf-8 -*-

__all__ = ["CeleriteNormal"]

import numpy as np
import pymc3 as pm
import theano.tensor as tt

from ..citations import add_citations_to_model
from ..theano_ops.celerite.dot_l import DotLOp
from ..theano_ops.celerite.factor import FactorOp
from ..theano_ops.celerite.solve import SolveOp


class CeleriteNormal(pm.distributions.Continuous):
    """A multivariate normal distribution for semi-separable matrices

    Args:
        kernel: A :class:`exoplanet.gp.terms.Term` object the specifies the
            GP kernel.
        x: The input coordinates. This should be a 1D array and the elements
            must be sorted. Otherwise the results are undefined.
        diag (Optional): The extra diagonal to add to the covariance matrix.
            This should have the same length as ``x`` and correspond to the
            excess *variance* for each data point. **Note:** this is different
            from the usage in the ``celerite`` package where the standard
            deviation (instead of variance) is provided.
        mu (Optional): The mean of the normal.
        J (Optional): The width of the system. This is the ``J`` parameter
            from `Foreman-Mackey (2018) <https://arxiv.org/abs/1801.10156>`_
            (not the original paper) so a real term contributes ``J += 1`` and
            a complex term contributes ``J += 2``. If you know this value in
            advance, you can provide it. Otherwise, the code will try to work
            it out.

    """

    __citations__ = ("celerite",)

    def __init__(self, kernel, x, diag=None, mu=None, *args, **kwargs):
        add_citations_to_model(
            self.__citations__, model=kwargs.get("model", None)
        )
        J = kwargs.pop("J", -1)

        self.x = tt.as_tensor_variable(x)
        kwargs["shape"] = kwargs.get("shape", x.shape)

        super(CeleriteNormal, self).__init__(*args, **kwargs)
        if len(self.shape) != 1:
            raise ValueError("Shape must have exactly one dimension")

        # Parse the inputs
        self.kernel = kernel
        if self.x.ndim != 1:
            raise ValueError("Invalid dimensions")
        if diag is None:
            self.diag = tt.zeros_like(self.x)
        else:
            self.diag = tt.as_tensor_variable(diag)
        if mu is None:
            self.mu = tt.zeros_like(self.x)
        else:
            self.mu = tt.as_tensor_variable(mu) + tt.zeros_like(self.x)
        self.mean = self.median = self.mode = self.mu

        # Work out the width of the kernel
        if J < 0:
            J = self.kernel.J
            if J > 32:
                J = -1
        self.J = J

        # Set up the ops that we need
        self.a, self.U, self.V, self.P = self.kernel.get_celerite_matrices(
            self.x, self.diag
        )
        self.factor_op = FactorOp(J=self.J)
        self.d, self.W, _, self.flag = self.factor_op(
            self.a, self.U, self.V, self.P
        )
        self.log_det = tt.sum(tt.log(self.d))
        self.norm = -0.5 * (self.log_det + self.x.size * np.log(2 * np.pi))
        self.vector_solve_op = SolveOp(J=self.J, n_rhs=1)
        self.dot_tril_op = DotLOp(J=self.J)
        self.vector_dot_tril_op = DotLOp(J=self.J, n_rhs=1)

    def random(self, point=None, size=None):
        if size is None:
            size = tuple()
        else:
            if not isinstance(size, tuple):
                try:
                    size = tuple(size)
                except TypeError:
                    size = (size,)

        # Sample all the parameters from the prior
        mu, U, P, d, W = pm.distributions.draw_values(
            [self.mu, self.U, self.P, self.d, self.W], point=point, size=size
        )

        # Handle things differently if U is also sampled
        if len(U.shape) == len(size) + 2:
            raise NotImplementedError(
                "Prior sampling with a stochastic is not (yet) implemented"
            )

        # Sample a standard normal
        norm = np.random.standard_normal((d.shape[-1],) + size)

        # Rotate into the correct frame, then swap the axes to get 'size'
        # to be the first axes
        res = np.moveaxis(self.dot_tril_op(U, P, d, W, norm).eval(), 0, -1)
        return mu + res

    def logp(self, value):
        resid = value - self.mu
        if resid.ndim != 1:
            raise ValueError("Invalid dimensions")
        alpha, _, _ = self.vector_solve_op(
            self.U, self.P, self.d, self.W, tt.reshape(resid, (resid.size, 1))
        )
        alpha = tt.reshape(alpha, (resid.size,))
        logp = self.norm - 0.5 * tt.sum(resid * alpha)
        return pm.distributions.dist_math.bound(logp, tt.eq(self.flag, 0))
