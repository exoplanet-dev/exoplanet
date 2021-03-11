# -*- coding: utf-8 -*-

__all__ = ["get_log_abs_det_jacobian", "estimate_inverse_gamma_parameters"]

from pymc3_ext.distributions import (
    estimate_inverse_gamma_parameters,
    get_log_abs_det_jacobian,
)

from ..utils import deprecated

get_log_abs_det_jacobian = deprecated(
    "pymc3_ext.distributions.get_log_abs_det_jacobian"
)(get_log_abs_det_jacobian)
estimate_inverse_gamma_parameters = deprecated(
    "pymc3_ext.distributions.estimate_inverse_gamma_parameters"
)(estimate_inverse_gamma_parameters)
