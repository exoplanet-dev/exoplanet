# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GP"]

import numpy as np
import theano.tensor as tt

from ..citations import add_citations_to_model
from ..theano_ops.celerite.solve import SolveOp
from ..theano_ops.celerite.factor import FactorOp
from ..theano_ops.celerite.diag_dot import DiagDotOp


diag_dot = DiagDotOp()


class GP(object):
    """The interface for computing Gaussian Process models with celerite

    This class implements the method described in `Foreman-Mackey et al. (2017)
    <https://arxiv.org/abs/1703.09710>`_ and `Foreman-Mackey (2018)
    <https://arxiv.org/abs/1801.10156>`_ for scalable evaluation of Gaussian
    Process (GP) models in 1D.

    .. note:: The input coordinates ``x`` must be sorted in ascending order,
        but this is not checked in the code. If the values are not sorted, the
        behavior of the algorithm is undefined.

    Args:
        kernel: A :class:`exoplanet.gp.terms.Term` object the specifies the
            GP kernel.
        x: The input coordinates. This should be a 1D array and the elements
            must be sorted. Otherwise the results are undefined.
        diag: The extra diagonal to add to the covariance matrix. This should
            have the same length as ``x`` and correspond to the excess
            *variance* for each data point. **Note:** this is different from
            the usage in the ``celerite`` package where the standard deviation
            (instead of variance) is provided.
        J (Optional): The width of the system. This is the ``J`` parameter
            from `Foreman-Mackey (2018) <https://arxiv.org/abs/1801.10156>`_
            (not the original paper) so a real term contributes ``J += 1`` and
            a complex term contributes ``J += 2``. If you know this value in
            advance, you can provide it. Otherwise, the code will try to work
            it out.

    """

    __citations__ = ("celerite", )

    def __init__(self, kernel, x, diag, J=-1, model=None):
        add_citations_to_model(self.__citations__, model=model)

        self.kernel = kernel
        if J < 0:
            J = self.kernel.J
            if J > 32:
                J = -1
        self.J = J

        self.x = tt.as_tensor_variable(x)
        self.diag = tt.as_tensor_variable(diag)
        self.a, self.U, self.V, self.P = self.kernel.get_celerite_matrices(
            self.x, self.diag)
        self.factor_op = FactorOp(J=self.J)
        self.d, self.W, _, self.flag = self.factor_op(
            self.a, self.U, self.V, self.P)

        self.vector_solve_op = SolveOp(J=self.J, n_rhs=1)
        self.general_solve_op = SolveOp(J=self.J)

    def log_likelihood(self, y):
        self.y = y
        self.z, _, _ = self.vector_solve_op(self.U, self.P, self.d, self.W,
                                            tt.reshape(self.y,
                                                       (self.y.size, 1)))
        loglike = -0.5 * tt.sum(self.y * self.z[:, 0] + tt.log(self.d))
        loglike -= 0.5 * self.y.size * np.log(2*np.pi)
        return tt.switch(tt.eq(self.flag, 0), loglike, -np.inf)

    def apply_inverse(self, rhs):
        return self.general_solve_op(self.U, self.P, self.d, self.W, rhs)[0]

    def predict(self, t=None, return_var=False, return_cov=False, kernel=None):
        mu = None
        if t is None and kernel is None:
            mu = self.y - self.diag * self.z[:, 0]
            if not (return_var or return_cov):
                return mu

        if kernel is None:
            kernel = self.kernel

        if t is None:
            t = self.x
            Kxs = kernel.value(self.x[:, None] - self.x[None, :])
            KxsT = Kxs
            Kss = Kxs
        else:
            t = tt.as_tensor_variable(t)
            KxsT = kernel.value(t[None, :] - self.x[:, None])
            Kxs = tt.transpose(KxsT)
            Kss = kernel.value(t[:, None] - t[None, :])

        if mu is None:
            mu = tt.dot(Kxs, self.z)[:, 0]

        if not (return_var or return_cov):
            return mu

        KinvKxsT = self.apply_inverse(KxsT)
        if return_var:
            var = -diag_dot(Kxs, KinvKxsT)  # tt.sum(KxsT*KinvKxsT, axis=0)
            var += kernel.value(0)
            return mu, var

        cov = Kss - tt.dot(Kxs, KinvKxsT)
        return mu, cov
