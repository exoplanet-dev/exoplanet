# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["eval_in_model", "get_samples_from_trace", "optimize",
           "get_args_for_theano_function", "get_theano_function_for_var"]

import numpy as np

import theano

import pymc3 as pm

from pymc3.model import Point
from pymc3.theanof import inputvars
from pymc3.util import update_start_vals, get_default_varnames
from pymc3.blocking import DictToArrayBijection, ArrayOrdering


def get_args_for_theano_function(point=None, model=None):
    model = pm.modelcontext(model)
    if point is None:
        point = model.test_point
    return [point[k.name] for k in model.vars]


def get_theano_function_for_var(var, model=None, **kwargs):
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


def optimize(start=None, vars=None, model=None, return_info=False,
             verbose=True, **kwargs):
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
    grad = theano.grad(nlp, vars, disconnected_inputs='ignore')
    func = get_theano_function_for_var([nlp] + grad, model=model)

    # This returns the objective function and its derivatives
    def objective(vec):
        res = func(*get_args_for_theano_function(bij.rmap(vec), model=model))
        d = dict(zip((v.name for v in vars), res[1:]))
        g = bij.map(d)
        return res[0], g

    if verbose:
        print("optimizing logp for variables: {0}"
              .format([v.name for v in vars]))

    # Optimize using scipy.optimize
    x0 = bij.map(start)
    initial = objective(x0)[0]
    kwargs["jac"] = True
    info = minimize(objective, x0, **kwargs)

    # Only accept the output if it is better than it was
    x = info.x if (np.isfinite(info.fun) and info.fun < initial) else x0

    # Coerce the output into the right format
    vars = get_default_varnames(model.unobserved_RVs, True)
    point = {var.name: value
             for var, value in zip(vars, model.fastfn(vars)(bij.rmap(x)))}

    if verbose:
        print("message: {0}".format(info.message))
        print("logp: {0} -> {1}".format(-initial, -info.fun))

    if return_info:
        return point, info
    return point


def allinmodel(vars, model):
    notin = [v for v in vars if v not in model.vars]
    if notin:
        raise ValueError("Some variables not in the model: " + str(notin))
