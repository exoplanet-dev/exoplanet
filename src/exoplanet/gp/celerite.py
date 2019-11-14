# -*- coding: utf-8 -*-

__all__ = ["GP"]

import numpy as np
import theano.tensor as tt

from ..citations import add_citations_to_model
from ..theano_ops.celerite.conditional_mean import ConditionalMeanOp
from ..theano_ops.celerite.diag_dot import DiagDotOp
from ..theano_ops.celerite.dot_l import DotLOp
from ..theano_ops.celerite.factor import FactorOp
from ..theano_ops.celerite.solve import SolveOp

diag_dot = DiagDotOp()


class GP:
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

    __citations__ = ("celerite",)

    def __init__(self, kernel, x, diag, J=-1, model=None):
        add_citations_to_model(self.__citations__, model=model)

        self.kernel = kernel
        if J < 0:
            J = self.kernel.J
            if J > 32:
                J = -1
        self.J = J

        self.z = None
        self.x = tt.as_tensor_variable(x)
        self.diag = tt.as_tensor_variable(diag)
        self.a, self.U, self.V, self.P = self.kernel.get_celerite_matrices(
            self.x, self.diag
        )
        self.factor_op = FactorOp(J=self.J)
        self.d, self.W, _, self.flag = self.factor_op(
            self.a, self.U, self.V, self.P
        )
        self.log_det = tt.sum(tt.log(self.d))
        self.norm = 0.5 * (self.log_det + self.x.size * np.log(2 * np.pi))

        self.vector_solve_op = SolveOp(J=self.J, n_rhs=1)
        self.general_solve_op = SolveOp(J=self.J)

        self.conditional_mean_op = ConditionalMeanOp(J=self.J)
        self.dot_l_op = DotLOp(J=self.J)

    def condition(self, y):
        self.y = y
        z, _, _ = self.vector_solve_op(
            self.U,
            self.P,
            self.d,
            self.W,
            tt.reshape(self.y, (self.y.size, 1)),
        )
        self.z = tt.reshape(z, (self.y.size,))
        self.loglike = tt.switch(
            tt.eq(self.flag, 0),
            -0.5 * tt.sum(self.y * self.z) - self.norm,
            -np.inf,
        )
        return self.z

    def log_likelihood(self, y=None):
        if y is not None:
            self.condition(y)
        return self.loglike

    def dot_l(self, n):
        n = tt.as_tensor_variable(n)
        return self.dot_l_op(self.U, self.P, self.d, self.W, n)

    def apply_inverse(self, rhs):
        return self.general_solve_op(self.U, self.P, self.d, self.W, rhs)[0]

    def predict(
        self,
        t=None,
        return_var=False,
        return_cov=False,
        kernel=None,
        _fast_mean=True,
    ):
        if self.z is None:
            raise RuntimeError("'condition' must be called before 'predict'")

        mu = None
        if t is None and kernel is None:
            mu = self.y - self.diag * self.z
            if not (return_var or return_cov):
                return mu

        if kernel is None:
            kernel = self.kernel
        else:
            _fast_mean = False

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
            if _fast_mean:
                U_star, V_star, inds = kernel.get_conditional_mean_matrices(
                    self.x, t
                )
                mu = self.conditional_mean_op(
                    self.U, self.V, self.P, self.z, U_star, V_star, inds
                )
            else:
                mu = tt.dot(Kxs, self.z)

        if not (return_var or return_cov):
            return mu

        KinvKxsT = self.apply_inverse(KxsT)
        if return_var:
            var = -diag_dot(Kxs, KinvKxsT)  # tt.sum(KxsT*KinvKxsT, axis=0)
            var += kernel.value(0)
            return mu, var

        cov = Kss - tt.dot(Kxs, KinvKxsT)
        return mu, cov
