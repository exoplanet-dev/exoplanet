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

import numpy as np
import pymc3 as pm
import theano

logger = logging.getLogger("exoplanet")

if theano.config.floatX != "float64":
    warnings.warn(
        "exoplanet should only be used with 'float64' precision, "
        "but theano.config.floatX == '{0}'".format(theano.config.floatX)
    )


def as_tensor_variable(x, dtype="float64", **kwargs):
    t = theano.tensor.as_tensor_variable(x, **kwargs)
    if dtype is None:
        return t
    return t.astype(dtype)


def get_args_for_theano_function(point=None, model=None):
    """Get the arguments required to evaluate a PyMC3 model component

    Use the result as the arguments for the callable returned by the
    :func:`get_theano_function_for_var` function.

    Args:
        point (dict, optional): The point in parameter space where the model
            componenet should be evaluated
        model (optional): The PyMC3 model object. By default, the current
            context will be used.

    """
    model = pm.modelcontext(model)
    if point is None:
        point = model.test_point
    return [point[k.name] for k in model.vars]


def get_theano_function_for_var(var, model=None, **kwargs):
    """Get a callable function to evaluate a component of a PyMC3 model

    This should then be called using the arguments returned by the
    :func:`get_args_for_theano_function` function.

    Args:
        var: The model component to evaluate. This can be a Theano tensor, a
            PyMC3 variable, or a list of these
        model (optional): The PyMC3 model object. By default, the current
            context will be used.

    """
    model = pm.modelcontext(model)
    kwargs["on_unused_input"] = kwargs.get("on_unused_input", "ignore")
    return theano.function(model.vars, var, **kwargs)


def eval_in_model(var, point=None, return_func=False, model=None, **kwargs):
    """Evaluate a Theano tensor or PyMC3 variable in a PyMC3 model

    This method builds a Theano function for evaluating a node in the graph
    given the required parameters. This will also cache the compiled Theano
    function in the current ``pymc3.Model`` to reduce the overhead of calling
    this function many times.

    Args:
        var: The variable or tensor to evaluate.
        point (Optional): A ``dict`` of input parameter values. This can be
            ``model.test_point`` (default), the result of ``pymc3.find_MAP``,
            a point in a ``pymc3.MultiTrace`` or any other representation of
            the input parameters.
        return_func (Optional[bool]): If ``False`` (default), return the
            evaluated variable. If ``True``, return the result, the Theano
            function and the list of arguments for that function.

    Returns:
        Depending on ``return_func``, either the value of ``var`` at ``point``,
        or this value, the Theano function, and the input arguments.

    """
    func = get_theano_function_for_var(var, model=model, **kwargs)
    args = get_args_for_theano_function(point=point, model=model)
    if return_func:
        return func(*args), func, args
    return func(*args)


def get_samples_from_trace(trace, size=1):
    """Generate random samples from a PyMC3 MultiTrace

    Args:
        trace: The ``MultiTrace``.
        size: The number of samples to generate.

    """
    for i in range(size):
        chain_idx = np.random.randint(len(trace.chains))
        sample_idx = np.random.randint(len(trace))
        yield trace._straces[chain_idx][sample_idx]


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
