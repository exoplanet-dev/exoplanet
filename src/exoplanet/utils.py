# -*- coding: utf-8 -*-

__all__ = [
    "logger",
    "eval_in_model",
    "get_samples_from_trace",
    "optimize",
    "get_args_for_theano_function",
    "get_theano_function_for_var",
    "deprecation_warning",
    "deprecated",
]

import logging
import os
import sys
import warnings
from functools import wraps

import numpy as np
import pymc3 as pm
import theano
from pymc3.blocking import ArrayOrdering, DictToArrayBijection
from pymc3.model import Point
from pymc3.theanof import inputvars
from pymc3.util import (
    get_default_varnames,
    get_untransformed_name,
    is_transformed_name,
    update_start_vals,
)

logger = logging.getLogger("exoplanet")


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


def optimize(
    start=None,
    vars=None,
    model=None,
    return_info=False,
    verbose=True,
    progress_bar=True,
    **kwargs
):
    """Maximize the log prob of a PyMC3 model using scipy

    All extra arguments are passed directly to the ``scipy.optimize.minimize``
    function.

    Args:
        start: The PyMC3 coordinate dictionary of the starting position
        vars: The variables to optimize
        model: The PyMC3 model
        return_info: Return both the coordinate dictionary and the result of
            ``scipy.optimize.minimize``
        verbose: Print the success flag and log probability to the screen
        progress_bar: A ``tqdm`` progress bar instance. Set to ``True``
            (default) to use ``tqdm.auto.tqdm()``. Set to ``False`` to disable.

    """
    from scipy.optimize import minimize

    model = pm.modelcontext(model)

    # Work out the full starting coordinates
    if start is None:
        start = model.test_point
    else:
        update_start_vals(start, model.test_point, model)

    # Fit all the parameters by default
    if vars is None:
        vars = model.cont_vars
    vars = inputvars(vars)
    allinmodel(vars, model)

    # Work out the relevant bijection map
    start = Point(start, model=model)
    bij = DictToArrayBijection(ArrayOrdering(vars), start)

    # Pre-compile the theano model and gradient
    nlp = -model.logpt
    grad = theano.grad(nlp, vars, disconnected_inputs="ignore")
    func = get_theano_function_for_var([nlp] + grad, model=model)

    if verbose:
        names = [
            get_untransformed_name(v.name)
            if is_transformed_name(v.name)
            else v.name
            for v in vars
        ]
        sys.stderr.write(
            "optimizing logp for variables: [{0}]\n".format(", ".join(names))
        )

        if progress_bar is True:
            if "EXOPLANET_NO_AUTO_PBAR" in os.environ:
                from tqdm import tqdm
            else:
                from tqdm.auto import tqdm

            progress_bar = tqdm()

    # Check whether the input progress bar has the expected methods
    has_progress_bar = (
        hasattr(progress_bar, "set_postfix")
        and hasattr(progress_bar, "update")
        and hasattr(progress_bar, "close")
    )

    # This returns the objective function and its derivatives
    def objective(vec):
        try:
            res = func(
                *get_args_for_theano_function(bij.rmap(vec), model=model)
            )
        except Exception:
            import traceback

            print("array:", vec)
            print("point:", bij.rmap(vec))
            traceback.print_exc()
            raise

        d = dict(zip((v.name for v in vars), res[1:]))
        g = bij.map(d)
        if verbose and has_progress_bar:
            progress_bar.set_postfix(logp="{0:e}".format(-res[0]))
            progress_bar.update()
        return res[0], g

    # Optimize using scipy.optimize
    x0 = bij.map(start)
    initial = objective(x0)[0]
    kwargs["jac"] = True
    info = minimize(objective, x0, **kwargs)

    # Only accept the output if it is better than it was
    x = info.x if (np.isfinite(info.fun) and info.fun < initial) else x0

    # Coerce the output into the right format
    vars = get_default_varnames(model.unobserved_RVs, True)
    point = {
        var.name: value
        for var, value in zip(vars, model.fastfn(vars)(bij.rmap(x)))
    }

    if verbose:
        if has_progress_bar:
            progress_bar.close()

        sys.stderr.write("message: {0}\n".format(info.message))
        sys.stderr.write("logp: {0} -> {1}\n".format(-initial, -info.fun))
        if not np.isfinite(info.fun):
            logger.warning("final logp not finite, returning initial point")
            logger.warning(
                "this suggests that something is wrong with the model"
            )
            logger.debug("{0}".format(info))

    if return_info:
        return point, info
    return point


def allinmodel(vars, model):
    notin = [v for v in vars if v not in model.vars]
    if notin:
        raise ValueError("Some variables not in the model: " + str(notin))


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
