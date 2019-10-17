# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm

from .utils import eval_in_model, optimize


def test_eval_in_model(seed=123409):
    np.random.seed(seed)
    x_val = np.random.randn(5, 3)
    x_val2 = np.random.randn(5, 3)
    with pm.Model():
        x = pm.Normal("x", shape=x_val.shape, testval=x_val)
        assert np.allclose(eval_in_model(x), x_val)
        assert np.allclose(eval_in_model(x, {"x": x_val2}), x_val2)


def test_optimize(seed=1234):
    np.random.seed(seed)
    x_val = np.random.randn(5, 3)
    with pm.Model():
        pm.Normal("x", shape=x_val.shape, testval=x_val)
        soln1 = optimize()
        soln2, info = optimize(soln1, return_info=True)

    assert np.allclose(soln1["x"], 0.0)
    assert np.allclose(soln2["x"], 0.0)
    assert info.success
