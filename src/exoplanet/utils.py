# -*- coding: utf-8 -*-

__all__ = ["logger", "as_tensor_variable", "deprecation_warning", "deprecated"]

import logging
import warnings
from functools import wraps

from aesara_theano_fallback import aesara as theano

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
