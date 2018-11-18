# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["eval_in_model"]

import theano

from pymc3.model import modelcontext


def eval_in_model(var, point=None, model=None, return_func=False, **kwargs):
    model = modelcontext(model)
    if point is None:
        point = model.test_point
    kwargs["on_unused_input"] = kwargs.get("on_unused_input", "ignore")
    func = theano.function(model.vars, var, **kwargs)
    args = [point[k.name] for k in model.vars]
    if return_func:
        return func(*args), func, args
    return func(*args)
