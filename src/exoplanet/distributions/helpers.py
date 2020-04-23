# -*- coding: utf-8 -*-

__all__ = ["get_log_abs_det_jacobian", "estimate_inverse_gamma_parameters"]

import numpy as np
import theano
import theano.tensor as tt
from scipy.optimize import root
from scipy.special import gammaincc


def get_log_abs_det_jacobian(in_vars, out_vars, **kwargs):
    r"""Get the log absolute determinant of the Jacobian between parameter sets

    This returns a Theano tensor representation of
    :math:`\log \left|\mathrm{det} J \right|` where the elements of :math:`J`
    are :math:`J_{nm} = \partial y_n / \partial x_m`. Here, :math:`y_n` and
    :math:`x_m` should be Theano tensors or PyMC3 variables.

    Args:
        in_vars: A list of input parameters :math:`x_m`
        out_vars: A list of output parameters :math:`y_n`

    """
    out_vec = []
    for v in out_vars:
        out_vec.append(tt.as_tensor_variable(v).flatten(ndim=1))
    out_vec = tt.concatenate(out_vec)

    kwargs["disconnected_inputs"] = kwargs.pop("disconnected_inputs", "ignore")
    jac = theano.gradient.jacobian(out_vec, in_vars, **kwargs)
    jac = tt.concatenate([j.reshape((out_vec.size, -1)) for j in jac], axis=-1)

    return tt.log(tt.abs_(tt.nlinalg.det(jac)))


def estimate_inverse_gamma_parameters(
    lower, upper, target=0.01, initial=None, **kwargs
):
    r"""Estimate an inverse Gamma with desired tail probabilities

    This method numerically solves for the parameters of an inverse Gamma
    distribution where the tails have a given probability. In other words
    :math:`P(x < \mathrm{lower}) = \mathrm{target}` and similarly for the
    upper bound. More information can be found in `part 4 of this blog post
    <https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html>`_.

    Args:
        lower (float): The location of the lower tail
        upper (float): The location of the upper tail
        target (float, optional): The desired tail probability
        initial (ndarray, optional): An initial guess for the parameters
            ``alpha`` and ``beta``

    Raises:
        RuntimeError: If the solver does not converge.

    Returns:
        dict: A dictionary with the keys ``alpha`` and ``beta`` for the
        parameters of the distribution.

    """
    lower, upper = np.sort([lower, upper])
    if initial is None:
        initial = np.array([2.0, 0.5 * (lower + upper)])
    if np.shape(initial) != (2,) or np.any(np.asarray(initial) <= 0.0):
        raise ValueError("invalid initial guess")

    def obj(x):
        a, b = np.exp(x)
        return np.array(
            [
                gammaincc(a, b / lower) - target,
                1 - gammaincc(a, b / upper) - target,
            ]
        )

    result = root(obj, np.log(initial), method="hybr", **kwargs)
    if not result.success:
        raise RuntimeError(
            "failed to find parameter estimates: \n{0}".format(result.message)
        )
    return dict(zip(("alpha", "beta"), np.exp(result.x)))
