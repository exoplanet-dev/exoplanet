__all__ = [
    "logger",
    "as_tensor_variable",
    "deprecation_warning",
    "deprecated",
    "docs_setup",
]

import logging
import warnings
from functools import wraps

from exoplanet.compat import tensor

logger = logging.getLogger("exoplanet")


def as_tensor_variable(x, dtype="float64", **kwargs):
    t = tensor.as_tensor_variable(x, **kwargs)
    if dtype is None:
        return t
    return t.astype(dtype)


def deprecation_warning(msg):
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


def deprecated(alternate=None):  # pragma: no cover
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


def docs_setup():
    """Set some environment variables and ignore some warnings for the docs"""
    import logging

    import matplotlib.pyplot as plt

    logger = logging.getLogger("theano.gof.compilelock")
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("pytensor.tensor.rewriting")
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("pytensor.tensor.blas")
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("matplotlib.font_manager")
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("exoplanet")
    logger.setLevel(logging.DEBUG)

    plt.style.use("default")
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
    plt.rcParams["font.cursive"] = ["Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
