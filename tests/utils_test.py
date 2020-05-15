# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm
import pytest
import theano.tensor as tt

from exoplanet.utils import eval_in_model, optimize


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
        soln1 = optimize(verbose=False)
        soln2, info = optimize(soln1, return_info=True, verbose=False)

    assert np.allclose(soln1["x"], 0.0)
    assert np.allclose(soln2["x"], 0.0)
    assert info.success


def test_optimize_exception(capsys):
    with pm.Model():
        cov = pm.Normal("cov", mu=np.eye(5), shape=(5, 5))
        chol = tt.slinalg.Cholesky(on_error="raise")(cov)
        pm.MvNormal("x", mu=np.zeros(5), chol=chol, shape=5)
        with pytest.raises(np.linalg.LinAlgError):
            optimize({"cov": np.zeros((5, 5))}, verbose=False)
        captured = capsys.readouterr()
        assert "array:" in captured.out
        assert "point:" in captured.out
