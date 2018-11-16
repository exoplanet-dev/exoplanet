# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["eval_in_model"]

import theano

from pymc3.model import modelcontext


def eval_in_model(var, point=None, model=None):
    model = modelcontext(model)
    if point is None:
        point = model.test_point
    func = theano.function(model.vars, var, on_unused_input="ignore")
    args = (point[k.name] for k in model.vars)
    return func(*args)
