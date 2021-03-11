# -*- coding: utf-8 -*-

__all__ = [
    "logger",
    "eval_in_model",
    "get_samples_from_trace",
    "get_args_for_theano_function",
    "get_theano_function_for_var",
    "deprecation_warning",
    "deprecated",
]

import logging
import warnings
from functools import wraps

from aesara_theano_fallback import aesara as theano
from pymc3_ext import (
    eval_in_model,
    get_args_for_theano_function,
    get_samples_from_trace,
    get_theano_function_for_var,
)

logger = logging.getLogger("exoplanet")


def as_tensor_variable(x, dtype="float64", **kwargs):
    t = theano.tensor.as_tensor_variable(x, **kwargs)
    if dtype is None:
        return t
    return t.astype(dtype)


def deprecation_warning(msg):
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


def deprecated(alternate=None):
    def wrapper(func, alternate=alternate):
        msg = "'{0}' is deprecated.".format(func.__name__)
        if alternate is not None:
            msg += " Use '{0}' instead.".format(alternate)

        @wraps(func)
        def f(*args, **kwargs):
            deprecation_warning(msg)
            return func(*args, **kwargs)

        return f

    return wrapper


eval_in_model = deprecated("pymc3_ext.eval_in_model")(eval_in_model)
get_samples_from_trace = deprecated("pymc3_ext.get_samples_from_trace")(
    get_samples_from_trace
)
get_args_for_theano_function = deprecated(
    "pymc3_ext.get_args_for_theano_function"
)(get_args_for_theano_function)
get_theano_function_for_var = deprecated(
    "pymc3_ext.get_theano_function_for_var"
)(get_theano_function_for_var)
